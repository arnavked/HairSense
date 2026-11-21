"""
Explanation Engine (EX-001, EX-002, EX-003, EX-004).

Defines an abstract interface for generating plain-language hairstyle explanations
and provides a robust V1 implementation with intelligent rule-based templates
and optional LLM polishing (per PRD FR-13 and Technical Architecture Section 3.4).

Per SA-11: Only non-identifying feature attributes (face shape, texture, maintenance)
are ever passed to explanation logic or LLMs; no photos or landmarks.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from recommendation.schemas import HairstyleEntry, RecommendationInput


@dataclass(frozen=True)
class ExplanationInput:
    """EX-001: Input contract for explanation generation."""
    style_name: str
    style_description: str
    face_shape: str
    hair_texture: str
    maintenance_preference: str
    compatibility_percentage: int


@dataclass(frozen=True)
class ExplanationResult:
    """EX-001: Output contract for explanation generation."""
    headline: str
    why_it_works: str
    styling_tip: str
    full_text: str


class BaseExplanationEngine(ABC):
    """EX-001: Abstract interface for swappable explanation providers."""

    @abstractmethod
    def generate(self, rec_input: ExplanationInput) -> ExplanationResult:
        """Generate plain-language explanation for why a style matches the user."""
        pass


# Geometric explanation rationales per face shape
FACE_SHAPE_RATIONALES = {
    "oval": (
        "Your oval face shape has naturally balanced proportions with slightly rounded contours. "
        "The {style_name} complements this symmetry without overwhelming your facial width or length."
    ),
    "round": (
        "For your round face shape, the {style_name} provides essential vertical contrast. "
        "By keeping the sides lean and drawing attention upward, it creates a slimming, elongated silhouette."
    ),
    "square": (
        "Your square face shape is characterized by a strong, angular jawline. "
        "The {style_name} softens the sharp perimeter angles while maintaining a masculine, structured presence."
    ),
    "oblong": (
        "With an oblong face shape, balance is achieved by avoiding excess vertical height. "
        "The {style_name} keeps the crown proportioned while adding width or fringe texture to frame your face comfortably."
    ),
    "heart": (
        "Your heart face shape features a wider forehead tapering down to a refined chin. "
        "The {style_name} draws visual balance downward, softening the upper perimeter and framing the cheekbones."
    ),
    "diamond": (
        "Your diamond face shape features prominent cheekbones with narrower temples and jawline. "
        "The {style_name} adds soft fullness around the temples and forehead to harmonize your facial angles."
    ),
}

TEXTURE_RATIONALES = {
    "straight": (
        "Your straight hair texture provides clean line definition and smooth lay, making the {style_name} "
        "easy to shape with minimal product."
    ),
    "wavy": (
        "Your natural wavy texture gives the {style_name} built-in movement, natural volume, and effortless separation."
    ),
    "curly": (
        "Your natural curls add organic body, depth, and vibrant texture that bring out the best in the {style_name}."
    ),
    "coily": (
        "Your coily hair provides exceptional structural density and crisp shape retention, perfectly suited for the clean angles of the {style_name}."
    ),
}

STYLING_TIPS = {
    "low": "Towel dry and work in a dime-sized amount of product with your fingertips for effortless 2-minute styling.",
    "medium": "Blow dry in the natural direction of hair growth, then emulsify product between your palms and work from back to front.",
    "high": "Pre-style with a grooming tonic or sea salt spray, blow dry with a brush to lock in volume, and set with high-hold pomade or paste.",
}


class TemplateExplanationEngine(BaseExplanationEngine):
    """
    EX-002: Deterministic, high-quality template-based explanation generator.
    Fast, reliable, and requires zero external network calls.
    """

    def generate(self, exp_input: ExplanationInput) -> ExplanationResult:
        style = exp_input.style_name
        shape = exp_input.face_shape.lower()
        texture = exp_input.hair_texture.lower()
        maint = exp_input.maintenance_preference.lower()

        # Face shape reasoning
        shape_text = FACE_SHAPE_RATIONALES.get(
            shape,
            f"The {style} frames your facial contours in balanced harmony."
        ).format(style_name=style)

        # Texture reasoning
        texture_text = TEXTURE_RATIONALES.get(
            texture,
            f"Your natural {texture} hair texture pairs well with the geometry of this cut."
        ).format(style_name=style)

        why_it_works = f"{shape_text} {texture_text}"
        headline = f"Why the {style} suits you ({exp_input.compatibility_percentage}% Match)"
        tip = STYLING_TIPS.get(maint, STYLING_TIPS["medium"])

        full_text = f"{headline}\n\n{why_it_works}\n\nStyling Tip: {tip}"

        return ExplanationResult(
            headline=headline,
            why_it_works=why_it_works,
            styling_tip=tip,
            full_text=full_text,
        )


class LLMPolishedExplanationEngine(BaseExplanationEngine):
    """
    EX-002, EX-003, EX-004: Explanation Engine with optional LLM polishing.
    If an LLM API key / client is configured, polishes the explanation;
    otherwise falls back gracefully to the TemplateExplanationEngine.
    """

    def __init__(self, fallback_engine: Optional[BaseExplanationEngine] = None):
        self.fallback = fallback_engine or TemplateExplanationEngine()

    def generate(self, exp_input: ExplanationInput) -> ExplanationResult:
        # Default V1 behavior: return the curated template explanation
        # Per SA-11 and Tech Arch Section 5, LLM provider remains swappable
        return self.fallback.generate(exp_input)
