"""
Bucket 3: Image Quality Assessment Pre-Check Stage.

Fulfills:
- CV-006: Image quality assessment: blur detection (Laplacian variance)
- CV-007: Image quality assessment: lighting/exposure check (histogram luminance)
- CV-008: Image quality assessment: yaw angle check (from 3D landmarks / head pose)
- CV-009: Image quality assessment: occlusion + multiple-face detection
- CV-010: Wire quality assessment as a pre-check stage before classification

Per PRD (FR-2, FR-3): Returns actionable error messages rather than silent bad predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import cv2
import numpy as np

from vision.face_landmarks import FaceLandmarkExtractor
from vision.landmark_groups import (
    CHIN,
    EYE_LEFT_OUTER,
    EYE_RIGHT_OUTER,
    NOSE_TIP,
)
from vision.schemas import BoundingBox, FaceDetectionResult, FaceResult, Landmark


class QualityIssueSeverity(str, Enum):
    ERROR = "ERROR"        # Blocks classification
    WARNING = "WARNING"    # Advisory, doesn't block classification


class QualityIssueCode(str, Enum):
    BLUR = "BLUR"
    TOO_DARK = "TOO_DARK"
    TOO_BRIGHT = "TOO_BRIGHT"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    FACE_YAW_EXCESSIVE = "FACE_YAW_EXCESSIVE"
    FACE_PITCH_EXCESSIVE = "FACE_PITCH_EXCESSIVE"
    FACE_ROLL_EXCESSIVE = "FACE_ROLL_EXCESSIVE"
    FACE_CUT_OFF = "FACE_CUT_OFF"
    FACE_TOO_SMALL = "FACE_TOO_SMALL"


@dataclass(frozen=True)
class QualityIssue:
    code: QualityIssueCode
    message: str
    severity: QualityIssueSeverity = QualityIssueSeverity.ERROR
    measured_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass(frozen=True)
class HeadPose:
    """Estimated head rotation angles in degrees."""
    yaw: float    # Left (-), Right (+)
    pitch: float  # Down (-), Up (+)
    roll: float   # Tilt left (-), Tilt right (+)


@dataclass(frozen=True)
class QualityAssessmentResult:
    """Consolidated output of all quality checks."""
    passed: bool
    issues: tuple[QualityIssue, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)
    pose: Optional[HeadPose] = None

    @property
    def primary_error(self) -> Optional[str]:
        """First error-level actionable message, or None if passed."""
        for issue in self.issues:
            if issue.severity == QualityIssueSeverity.ERROR:
                return issue.message
        return None

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            issue.message
            for issue in self.issues
            if issue.severity == QualityIssueSeverity.WARNING
        )


# Canonical 3D facial model points (in mm) for solvePnP pose estimation
# In standard camera coordinates: X right, Y down, Z forward
# Corresponding landmarks:
# lm 1: Nose tip
# lm 152: Chin
# lm 33: Right eye corner (person's right, left in image)
# lm 263: Left eye corner (person's left, right in image)
# lm 61: Right mouth corner (left in image)
# lm 291: Left mouth corner (right in image)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (lm 1)
    (0.0, 330.0, -65.0),      # Chin (lm 152)
    (-225.0, -170.0, -135.0), # Right eye outer corner (lm 33, left in image)
    (225.0, -170.0, -135.0),  # Left eye outer corner (lm 263, right in image)
    (-150.0, 150.0, -125.0),  # Right mouth corner (lm 61, left in image)
    (150.0, 150.0, -125.0),   # Left mouth corner (lm 291, right in image)
], dtype=np.float64)

KEYPOINT_INDICES = (1, 152, 33, 263, 61, 291)


def compute_blur_score(image_bgr: np.ndarray) -> float:
    """
    CV-006: Blur detection using the variance of the Laplacian.
    Higher value means sharper image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_lighting_metrics(image_bgr: np.ndarray) -> dict[str, float]:
    """
    CV-007: Lighting and exposure metrics from image luminance.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    underexposed_ratio = float(np.sum(gray < 20) / gray.size)
    overexposed_ratio = float(np.sum(gray > 245) / gray.size)

    return {
        "mean_brightness": mean_val,
        "underexposed_ratio": underexposed_ratio,
        "overexposed_ratio": overexposed_ratio,
    }


def estimate_head_pose(
    face: FaceResult,
    image_width: int,
    image_height: int,
) -> HeadPose:
    """
    CV-008: Estimate yaw, pitch, and roll in degrees using solvePnP
    with MediaPipe landmarks and canonical 3D facial model points.
    """
    landmarks = face.landmarks
    if len(landmarks) < 468:
        raise ValueError(f"Expected at least 468 landmarks, got {len(landmarks)}")

    # Extract 2D image coordinates for keypoints
    image_points = []
    for idx in KEYPOINT_INDICES:
        lm = landmarks[idx]
        image_points.append([lm.x * image_width, lm.y * image_height])
    image_points = np.array(image_points, dtype=np.float64)

    # Camera matrix approximation
    focal_length = float(image_width)
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return _fallback_pose_estimation(landmarks)

    rmat, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles (pitch, yaw)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)

    # Roll is best calculated directly from the eye corners in 2D image coordinates
    p_left_eye = landmarks[EYE_LEFT_OUTER]
    p_right_eye = landmarks[EYE_RIGHT_OUTER]
    dx = p_left_eye.x - p_right_eye.x
    dy = p_left_eye.y - p_right_eye.y
    roll_deg = math.degrees(math.atan2(dy, dx)) if dx != 0 else 0.0

    return HeadPose(
        yaw=float(np.degrees(yaw)),
        pitch=float(np.degrees(pitch)),
        roll=float(roll_deg),
    )


def _fallback_pose_estimation(landmarks: Sequence[Landmark]) -> HeadPose:
    """Fallback geometric estimation if solvePnP fails."""
    p_left_eye = landmarks[EYE_LEFT_OUTER]
    p_right_eye = landmarks[EYE_RIGHT_OUTER]
    p_nose = landmarks[NOSE_TIP]

    # Roll from eye slope
    dx = p_left_eye.x - p_right_eye.x
    dy = p_left_eye.y - p_right_eye.y
    roll = math.degrees(math.atan2(dy, dx)) if dx != 0 else 0.0

    # Yaw from nose symmetry between eyes
    dist_left = abs(p_left_eye.x - p_nose.x)
    dist_right = abs(p_right_eye.x - p_nose.x)
    total_dist = dist_left + dist_right
    if total_dist > 0:
        ratio = (dist_right - dist_left) / total_dist
        yaw = ratio * 50.0  # approximate scaling
    else:
        yaw = 0.0

    return HeadPose(yaw=yaw, pitch=0.0, roll=roll)


class QualityChecker:
    """
    CV-010: Wire quality assessment as a pre-check stage before classification.
    """

    def __init__(
        self,
        min_blur_score: float = 80.0,
        min_brightness: float = 38.0,
        max_brightness: float = 225.0,
        max_underexposed_ratio: float = 0.35,
        max_overexposed_ratio: float = 0.30,
        max_yaw_deg: float = 20.0,
        max_pitch_deg: float = 25.0,
        max_roll_deg: float = 20.0,
        min_face_ratio: float = 0.05,
        margin_threshold: float = 0.015,
    ):
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_underexposed_ratio = max_underexposed_ratio
        self.max_overexposed_ratio = max_overexposed_ratio
        self.max_yaw_deg = max_yaw_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_roll_deg = max_roll_deg
        self.min_face_ratio = min_face_ratio
        self.margin_threshold = margin_threshold

    def check_image_level(self, image_bgr: np.ndarray) -> tuple[list[QualityIssue], dict[str, float]]:
        """Run fast image-level checks (blur and lighting) before running detector."""
        issues: list[QualityIssue] = []
        metrics: dict[str, float] = {}

        # CV-006: Blur check
        blur_score = compute_blur_score(image_bgr)
        metrics["blur_score"] = blur_score
        if blur_score < self.min_blur_score:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.BLUR,
                    message="Photo is too blurry. Please hold the camera steady in good lighting.",
                    severity=QualityIssueSeverity.ERROR,
                    measured_value=blur_score,
                    threshold=self.min_blur_score,
                )
            )

        # CV-007: Lighting check (global image)
        lighting = compute_lighting_metrics(image_bgr)
        metrics.update(lighting)
        if lighting["mean_brightness"] < self.min_brightness or lighting["underexposed_ratio"] > self.max_underexposed_ratio:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.TOO_DARK,
                    message="Photo is too dark / underexposed. Please move to a brighter area or turn on a light.",
                    severity=QualityIssueSeverity.ERROR,
                    measured_value=lighting["mean_brightness"],
                    threshold=self.min_brightness,
                )
            )
        elif (
            lighting["mean_brightness"] > self.max_brightness
            or (lighting["mean_brightness"] > 210.0 and lighting["overexposed_ratio"] > 0.65)
        ):
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.TOO_BRIGHT,
                    message="Photo is overexposed / too bright. Please avoid harsh backlighting or direct glare.",
                    severity=QualityIssueSeverity.ERROR,
                    measured_value=lighting["mean_brightness"],
                    threshold=self.max_brightness,
                )
            )

        return issues, metrics

    def check_face_detection(
        self,
        detection: FaceDetectionResult,
        image_bgr: Optional[np.ndarray] = None,
    ) -> tuple[list[QualityIssue], dict[str, float], Optional[HeadPose]]:
        """
        CV-008 & CV-009: Check face detection counts, boundary cutoff, framing, and head pose angles.
        Optionally evaluates face-specific illumination if image_bgr is provided.
        """
        issues: list[QualityIssue] = []
        metrics: dict[str, float] = {
            "face_count": float(detection.face_count),
        }
        pose: Optional[HeadPose] = None

        # CV-009: Multiple faces / No face
        if detection.face_count == 0:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.NO_FACE,
                    message="No face detected. Please ensure your face is clearly visible and looking at the camera.",
                    severity=QualityIssueSeverity.ERROR,
                )
            )
            return issues, metrics, None

        if detection.face_count > 1:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.MULTIPLE_FACES,
                    message=f"Multiple faces detected ({detection.face_count}). Please upload a photo with only one person.",
                    severity=QualityIssueSeverity.ERROR,
                    measured_value=float(detection.face_count),
                    threshold=1.0,
                )
            )
            return issues, metrics, None

        # Single face checks
        face = detection.faces[0]
        bbox = face.bounding_box
        face_area_ratio = bbox.width * bbox.height
        metrics["face_area_ratio"] = float(face_area_ratio)

        # CV-009: Occlusion / cut-off at edge
        if (
            bbox.x_min <= self.margin_threshold
            or bbox.y_min <= self.margin_threshold
            or bbox.x_max >= (1.0 - self.margin_threshold)
            or bbox.y_max >= (1.0 - self.margin_threshold)
        ):
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.FACE_CUT_OFF,
                    message="Face is partially cut off at the edge of the photo. Please center your face in the frame.",
                    severity=QualityIssueSeverity.ERROR,
                )
            )

        # Face too small
        if face_area_ratio < self.min_face_ratio:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.FACE_TOO_SMALL,
                    message="Face is too far away. Please take the photo closer to your face or crop closer.",
                    severity=QualityIssueSeverity.ERROR,
                    measured_value=face_area_ratio,
                    threshold=self.min_face_ratio,
                )
            )

        # Face crop lighting check (if image provided)
        if image_bgr is not None and image_bgr.size > 0:
            h, w = image_bgr.shape[:2]
            y1 = max(0, int(bbox.y_min * h))
            y2 = min(h, int(bbox.y_max * h))
            x1 = max(0, int(bbox.x_min * w))
            x2 = min(w, int(bbox.x_max * w))
            if y2 > y1 and x2 > x1:
                face_crop = image_bgr[y1:y2, x1:x2]
                face_lighting = compute_lighting_metrics(face_crop)
                metrics["face_mean_brightness"] = face_lighting["mean_brightness"]
                if face_lighting["mean_brightness"] < 30.0:
                    issues.append(
                        QualityIssue(
                            code=QualityIssueCode.TOO_DARK,
                            message="Face is in deep shadow / too dark. Please face towards a light source.",
                            severity=QualityIssueSeverity.ERROR,
                            measured_value=face_lighting["mean_brightness"],
                            threshold=30.0,
                        )
                    )
                elif face_lighting["mean_brightness"] > 230.0 or face_lighting["overexposed_ratio"] > 0.40:
                    issues.append(
                        QualityIssue(
                            code=QualityIssueCode.TOO_BRIGHT,
                            message="Face is washed out / overexposed. Please avoid direct harsh lighting.",
                            severity=QualityIssueSeverity.ERROR,
                            measured_value=face_lighting["mean_brightness"],
                            threshold=230.0,
                        )
                    )

        # CV-008: Yaw / Pitch / Roll pose angles
        try:
            pose = estimate_head_pose(face, detection.image_width, detection.image_height)
            metrics["yaw_deg"] = pose.yaw
            metrics["pitch_deg"] = pose.pitch
            metrics["roll_deg"] = pose.roll

            if abs(pose.yaw) > self.max_yaw_deg:
                issues.append(
                    QualityIssue(
                        code=QualityIssueCode.FACE_YAW_EXCESSIVE,
                        message=f"Face is turned at too steep an angle ({abs(pose.yaw):.1f}°). Please face directly forward toward the camera.",
                        severity=QualityIssueSeverity.ERROR,
                        measured_value=abs(pose.yaw),
                        threshold=self.max_yaw_deg,
                    )
                )

            if abs(pose.pitch) > self.max_pitch_deg:
                issues.append(
                    QualityIssue(
                        code=QualityIssueCode.FACE_PITCH_EXCESSIVE,
                        message=f"Head is tilted too high or low ({abs(pose.pitch):.1f}°). Please keep your head level with the camera.",
                        severity=QualityIssueSeverity.ERROR,
                        measured_value=abs(pose.pitch),
                        threshold=self.max_pitch_deg,
                    )
                )

            if abs(pose.roll) > self.max_roll_deg:
                issues.append(
                    QualityIssue(
                        code=QualityIssueCode.FACE_ROLL_EXCESSIVE,
                        message=f"Head is tilted sideways ({abs(pose.roll):.1f}°). Please keep your head straight.",
                        severity=QualityIssueSeverity.ERROR,
                        measured_value=abs(pose.roll),
                        threshold=self.max_roll_deg,
                    )
                )
        except Exception:
            # If pose calculation fails, don't crash
            pass

        return issues, metrics, pose

    def assess(
        self,
        image_bgr: np.ndarray,
        detection: Optional[FaceDetectionResult] = None,
        extractor: Optional[FaceLandmarkExtractor] = None,
    ) -> QualityAssessmentResult:
        """
        Orchestrate all quality pre-checks.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("assess() received an empty or None image")

        # Step 1: Fast image-level checks
        issues, metrics = self.check_image_level(image_bgr)

        # If image fails basic quality (e.g. pitch dark or completely blurry), return early
        has_critical_image_error = any(
            issue.severity == QualityIssueSeverity.ERROR for issue in issues
        )
        if has_critical_image_error and detection is None and extractor is None:
            return QualityAssessmentResult(
                passed=False,
                issues=tuple(issues),
                metrics=metrics,
                pose=None,
            )

        # Step 2: Face detection checks
        if detection is None:
            if extractor is None:
                extractor = FaceLandmarkExtractor()
            detection = extractor.extract(image_bgr)

        det_issues, det_metrics, pose = self.check_face_detection(
            detection, image_bgr=image_bgr
        )
        issues.extend(det_issues)
        metrics.update(det_metrics)

        has_error = any(issue.severity == QualityIssueSeverity.ERROR for issue in issues)

        return QualityAssessmentResult(
            passed=not has_error,
            issues=tuple(issues),
            metrics=metrics,
            pose=pose,
        )
