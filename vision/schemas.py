"""
Shared data structures for the face detection / landmark extraction stage.

These are intentionally plain dataclasses (not pydantic/etc.) so this module
has zero web-framework dependency — Bucket 6 (API layer) will wrap these,
and Bucket 3 (quality checks) / Bucket 4 (feature extraction) will consume
them directly without needing to know anything about mediapipe.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    """Normalized [0,1] bounding box in image coordinates."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class Landmark:
    """A single 3D landmark point, normalized to [0,1] in x/y, z is relative depth."""
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FaceResult:
    """Detection + landmark output for a single detected face."""
    bounding_box: BoundingBox
    landmarks: tuple  # tuple[Landmark, ...] — 478 points if refine_landmarks=True
    detection_confidence: Optional[float]  # None if the backend doesn't expose one
    landmark_index_map: dict = field(default_factory=dict)  # named subsets, see LANDMARK_GROUPS


@dataclass(frozen=True)
class FaceDetectionResult:
    """Top-level result for one input image. May contain 0, 1, or many faces."""
    image_width: int
    image_height: int
    faces: tuple  # tuple[FaceResult, ...]

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def single_face(self) -> Optional[FaceResult]:
        """Convenience accessor for the common case. None if face_count != 1."""
        return self.faces[0] if self.face_count == 1 else None