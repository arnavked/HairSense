"""
Bucket 2 / Ticket CV-004: Face Shape Classifier (ONNX Runtime inference).

Consumes landmarks from Bucket 1 (FaceResult), extracts anthropometric features,
and runs inference via the exported ONNX model (`models/face_shape_classifier.onnx`).

Returns:
- predicted_shape: str (e.g. "oval", "round", "square", "oblong", "heart", "diamond")
- confidence: float (0.0 to 1.0)
- class_probabilities: dict[str, float]
- features: dict[str, float]
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from vision.face_features import extract_face_shape_features, features_to_vector
from vision.schemas import FaceResult, Landmark

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_shape_classifier.onnx",
)

DEFAULT_META_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "face_shape_classifier_meta.json",
)


@dataclass(frozen=True)
class FaceShapeClassificationResult:
    """Output of face shape classification."""
    shape: str
    confidence: float
    probabilities: dict[str, float]
    features: dict[str, float]


class FaceShapeClassifier:
    """ONNX-based face shape classifier."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        meta_path: str = DEFAULT_META_PATH,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face shape classifier model not found at {model_path}. "
                f"Run scripts/train_face_shape_classifier.py to generate it."
            )
        self._model_path = model_path
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        # Load classes from metadata if present, else fallback
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._classes = meta.get(
                "classes",
                ["oval", "round", "square", "oblong", "heart", "diamond"],
            )
        else:
            self._classes = ["oval", "round", "square", "oblong", "heart", "diamond"]

    @property
    def classes(self) -> list[str]:
        return list(self._classes)

    def classify(
        self,
        face_or_landmarks: Union[FaceResult, Sequence[Landmark]],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> FaceShapeClassificationResult:
        """
        Classify a dimension-carrying FaceResult or landmarks with their
        source image dimensions.
        """
        features_dict = extract_face_shape_features(
            face_or_landmarks,
            image_width=image_width,
            image_height=image_height,
        )
        feat_vec = features_to_vector(features_dict).reshape(1, -1)

        outputs = self._session.run(None, {self._input_name: feat_vec})
        class_idx = int(outputs[0][0])
        probs = outputs[1][0]  # shape: (num_classes,)

        predicted_shape = self._classes[class_idx]
        confidence = float(probs[class_idx])
        probabilities = {
            self._classes[i]: float(probs[i]) for i in range(len(self._classes))
        }

        return FaceShapeClassificationResult(
            shape=predicted_shape,
            confidence=confidence,
            probabilities=probabilities,
            features=features_dict,
        )
