"""
Data models and schemas for the hairstyle recommendation engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


HairTexture = Literal["straight", "wavy", "curly", "coily"]
MaintenanceLevel = Literal["low", "medium", "high"]
HairLength = Literal["short", "medium", "long"]
FaceShape = Literal["oval", "round", "square", "oblong", "heart", "diamond"]


@dataclass(frozen=True)
class BarberSheet:
    sides_guard: str
    top_length: str
    texture_technique: str
    finish_product: str
    maintenance_interval_weeks: int


@dataclass(frozen=True)
class HairstyleEntry:
    id: str
    name: str
    description: str
    compatible_face_shapes: tuple[str, ...]
    compatible_hair_textures: tuple[str, ...]
    maintenance_level: str
    hair_length: str
    barber_sheet: BarberSheet


@dataclass(frozen=True)
class UserPreferences:
    hair_texture: str  # "straight", "wavy", "curly", "coily"
    maintenance_preference: str  # "low", "medium", "high"


@dataclass(frozen=True)
class RecommendationInput:
    """
    RE-003: Combined feature set from classifier and user inputs.
    """
    face_shape: str
    face_shape_confidence: float
    face_shape_probabilities: dict[str, float]
    preferences: UserPreferences


@dataclass(frozen=True)
class ScoreBreakdown:
    face_shape_score: float
    texture_score: float
    maintenance_score: float
    total_score: float


@dataclass(frozen=True)
class RecommendationItem:
    """
    Scored recommendation result.
    """
    rank: int
    style: HairstyleEntry
    compatibility_score: float  # [0.0, 1.0]
    score_breakdown: ScoreBreakdown
    compatibility_percentage: int  # 0 to 100
