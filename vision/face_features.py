"""
Bucket 2 / Ticket CV-003: Anthropometric Feature Extraction from Face Landmarks.

Extracts scale-, translation-, and rotation-robust geometric ratios from 
MediaPipe Face Mesh 3D landmarks (478 points).

Feature set:
1. aspect_ratio: Face length (forehead center to chin) / Bizygomatic width (cheekbone to cheekbone)
2. jaw_to_cheek_ratio: Bigonial width (jawline) / Bizygomatic width
3. forehead_to_cheek_ratio: Forehead width / Bizygomatic width
4. jaw_to_forehead_ratio: Bigonial width / Forehead width
5. chin_angle_deg: Angle between left jaw, chin, and right jaw in degrees
6. cheek_prominence: Ratio of cheek width to average of forehead and jaw widths
"""

from __future__ import annotations

import math
from typing import Sequence, Union
import numpy as np

from vision.landmark_groups import (
    CHIN,
    CHEEKBONE_LEFT,
    CHEEKBONE_RIGHT,
    FOREHEAD_CENTER,
)
from vision.schemas import FaceResult, Landmark

# Additional key landmark indices for anthropometry
FOREHEAD_LEFT = 54      # Left temporal / forehead outer
FOREHEAD_RIGHT = 284    # Right temporal / forehead outer
JAW_LEFT_ANGLE = 58     # Left gonion / jaw angle
JAW_RIGHT_ANGLE = 288   # Right gonion / jaw angle

FEATURE_NAMES = (
    "aspect_ratio",
    "jaw_to_cheek_ratio",
    "forehead_to_cheek_ratio",
    "jaw_to_forehead_ratio",
    "chin_angle_deg",
    "cheek_prominence",
)


def _distance_2d(
    p1: Landmark, p2: Landmark, image_width: int, image_height: int
) -> float:
    """Euclidean distance after restoring normalized points to pixel space."""
    dx = (p1.x - p2.x) * image_width
    dy = (p1.y - p2.y) * image_height
    return math.sqrt(dx * dx + dy * dy)


def _calculate_angle_2d(
    p_left: Landmark,
    p_center: Landmark,
    p_right: Landmark,
    image_width: int,
    image_height: int,
) -> float:
    """Calculate the interior angle at p_center formed by p_left and p_right (in degrees)."""
    v1 = np.array([
        (p_left.x - p_center.x) * image_width,
        (p_left.y - p_center.y) * image_height,
    ])
    v2 = np.array([
        (p_right.x - p_center.x) * image_width,
        (p_right.y - p_center.y) * image_height,
    ])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 90.0
    
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def extract_face_shape_features(
    face_or_landmarks: Union[FaceResult, Sequence[Landmark]],
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict[str, float]:
    """
    Extract normalized anthropometric facial ratios from landmarks.
    
    Parameters
    ----------
    face_or_landmarks:
        Either a FaceResult instance or a sequence/tuple of Landmark objects.
    image_width, image_height:
        Required for a raw landmark sequence. FaceResult inputs carry these
        dimensions from the image used for landmark extraction.
        
    Returns
    -------
    dict[str, float]
        Dictionary of computed scale-invariant geometric features.
    """
    if isinstance(face_or_landmarks, FaceResult):
        landmarks = face_or_landmarks.landmarks
        if image_width is None:
            image_width = face_or_landmarks.image_width
        if image_height is None:
            image_height = face_or_landmarks.image_height
    else:
        landmarks = face_or_landmarks

    if len(landmarks) < 468:
        raise ValueError(
            f"Expected at least 468 MediaPipe landmarks, got {len(landmarks)}"
        )
    if (
        image_width is None
        or image_height is None
        or image_width <= 0
        or image_height <= 0
    ):
        raise ValueError(
            "Positive image_width and image_height are required to convert "
            "normalized landmarks to pixel-space geometry."
        )

    # Key anthropometric landmarks
    pt_forehead = landmarks[FOREHEAD_CENTER]
    pt_chin = landmarks[CHIN]
    pt_cheek_l = landmarks[CHEEKBONE_LEFT]
    pt_cheek_r = landmarks[CHEEKBONE_RIGHT]
    pt_jaw_l = landmarks[JAW_LEFT_ANGLE]
    pt_jaw_r = landmarks[JAW_RIGHT_ANGLE]
    pt_forehead_l = landmarks[FOREHEAD_LEFT]
    pt_forehead_r = landmarks[FOREHEAD_RIGHT]

    # Distances
    face_length = _distance_2d(pt_forehead, pt_chin, image_width, image_height)
    cheek_width = max(
        _distance_2d(pt_cheek_l, pt_cheek_r, image_width, image_height), 1e-6
    )
    jaw_width = max(
        _distance_2d(pt_jaw_l, pt_jaw_r, image_width, image_height), 1e-6
    )
    forehead_width = max(
        _distance_2d(pt_forehead_l, pt_forehead_r, image_width, image_height), 1e-6
    )

    # Ratios (scale-invariant)
    aspect_ratio = face_length / cheek_width
    jaw_to_cheek = jaw_width / cheek_width
    forehead_to_cheek = forehead_width / cheek_width
    jaw_to_forehead = jaw_width / forehead_width
    
    # Chin angle formed at the gnathion (p_chin) between jaw angles
    chin_angle = _calculate_angle_2d(
        pt_jaw_l, pt_chin, pt_jaw_r, image_width, image_height
    )

    # Cheek prominence compared to forehead & jaw
    avg_extremities = (forehead_width + jaw_width) / 2.0
    cheek_prominence = cheek_width / max(avg_extremities, 1e-6)

    return {
        "aspect_ratio": float(aspect_ratio),
        "jaw_to_cheek_ratio": float(jaw_to_cheek),
        "forehead_to_cheek_ratio": float(forehead_to_cheek),
        "jaw_to_forehead_ratio": float(jaw_to_forehead),
        "chin_angle_deg": float(chin_angle),
        "cheek_prominence": float(cheek_prominence),
    }


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Convert features dictionary to a 1D float32 numpy array ordered by FEATURE_NAMES."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
