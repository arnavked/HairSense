"""
Bucket 1: Face Detection & Landmarks.

B1-001 (Face detection) + B1-002 (Landmark extraction) live in one class
because MediaPipe's FaceLandmarker task does both in a single pass — there's
no separate detector step to wire up. FaceLandmarkExtractor is the single
entry point later buckets should import:

    from vision.face_landmarks import FaceLandmarkExtractor
    extractor = FaceLandmarkExtractor()
    result = extractor.extract(image_bgr)

REQUIRES a downloaded model file — see models/README.md / scripts/download_models.sh.
This module targets the mediapipe Tasks API (mediapipe>=0.10), NOT the older
`mp.solutions.face_mesh` API, which this mediapipe build no longer ships.

Deliberately NOT included here (out of scope for Bucket 1, see PRD Section
4 / ticket buckets 2-3):
- face-shape classification (Bucket 2)
- blur/lighting/angle/occlusion quality checks (Bucket 3) — though this
  module gives Bucket 3 everything it needs (landmarks + bbox)
"""

import os
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from vision.schemas import BoundingBox, FaceDetectionResult, FaceResult, Landmark

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_landmarker.task",
)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the FaceLandmarker .task model file isn't on disk yet."""


class FaceLandmarkExtractor:
    """
    Wraps mediapipe's Tasks-API FaceLandmarker (IMAGE running mode).

    IMAGE running mode is deliberate: V1 is a static-photo pipeline (see
    PRD Non-Goals — no live camera/AR), so we don't want VIDEO/LIVE_STREAM
    mode's frame-to-frame tracking heuristics.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_num_faces: int = 5,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
    ):
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"FaceLandmarker model not found at {model_path}. "
                f"Run scripts/download_models.sh (requires network) to fetch it, "
                f"or pass model_path= to point at a model file you already have."
            )
        self._model_path = model_path
        self._max_num_faces = max_num_faces
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._landmarker = self._make_landmarker()

    def _make_landmarker(self) -> FaceLandmarker:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=self._max_num_faces,
            min_face_detection_confidence=self._min_detection_confidence,
            min_face_presence_confidence=self._min_presence_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return FaceLandmarker.create_from_options(options)

    def extract(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        """
        Run detection + landmark extraction on a single BGR image (as
        returned by cv2.imread).

        Returns a FaceDetectionResult with zero, one, or many FaceResult
        entries. Zero faces and multiple faces are both valid, non-error
        outcomes here — Bucket 3's quality-check stage decides what to do
        about them (reject / warn), not this module.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("extract() received an empty or None image")

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        mp_result = self._landmarker.detect(mp_image)

        faces = []
        for face_landmark_list in mp_result.face_landmarks:
            landmarks = tuple(
                Landmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmark_list
            )
            bbox = _bounding_box_from_landmarks(landmarks)
            faces.append(
                FaceResult(
                    bounding_box=bbox,
                    landmarks=landmarks,
                    # FaceLandmarkerResult doesn't expose a single scalar
                    # detection-confidence per face (it exposes landmarks
                    # + optional blendshapes/transform matrices). Flagging
                    # None rather than fabricating a number — revisit if
                    # Bucket 3/6 needs one (could derive from blendshape
                    # presence scores, or run the separate FaceDetector
                    # task alongside this for its bbox-level score).
                    detection_confidence=None,
                )
            )

        return FaceDetectionResult(
            image_width=width,
            image_height=height,
            faces=tuple(faces),
        )

    def extract_from_path(self, image_path: str) -> FaceDetectionResult:
        """Convenience wrapper: load an image from disk and extract."""
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        return self.extract(image_bgr)

def close(self):
    """Release MediaPipe resources."""
    if self._landmarker:
        self._landmarker.close()
def _bounding_box_from_landmarks(landmarks: tuple) -> BoundingBox:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return BoundingBox(
        x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys),
    )