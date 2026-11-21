"""
Unit tests for Bucket 2: Face Shape Features & Classification (CV-003, CV-004, CV-005).
"""

import os
import unittest
import numpy as np

from vision.face_classifier import FaceShapeClassifier, FaceShapeClassificationResult
from vision.face_features import (
    FEATURE_NAMES,
    extract_face_shape_features,
    features_to_vector,
)
from vision.face_landmarks import FaceLandmarkExtractor
from vision.schemas import Landmark, BoundingBox, FaceResult


class TestFaceFeaturesAndClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = FaceShapeClassifier()
        cls.extractor = FaceLandmarkExtractor()
        cls.sample_image_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_images",
            "indian_male.jpg",
        )

    def test_feature_names_and_ordering(self):
        self.assertEqual(len(FEATURE_NAMES), 6)
        self.assertIn("aspect_ratio", FEATURE_NAMES)
        self.assertIn("jaw_to_cheek_ratio", FEATURE_NAMES)
        self.assertIn("forehead_to_cheek_ratio", FEATURE_NAMES)
        self.assertIn("jaw_to_forehead_ratio", FEATURE_NAMES)
        self.assertIn("chin_angle_deg", FEATURE_NAMES)
        self.assertIn("cheek_prominence", FEATURE_NAMES)

    def test_features_to_vector(self):
        feats = {name: float(i + 1) for i, name in enumerate(FEATURE_NAMES)}
        vec = features_to_vector(feats)
        self.assertEqual(vec.shape, (6,))
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(list(vec), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_feature_extraction_raises_on_few_landmarks(self):
        few_landmarks = tuple(Landmark(0.5, 0.5, 0.0) for _ in range(100))
        with self.assertRaises(ValueError):
            extract_face_shape_features(few_landmarks)

    def test_classify_sample_image(self):
        res = self.extractor.extract_from_path(self.sample_image_path)
        self.assertIsNotNone(res.single_face)
        out = self.classifier.classify(res.single_face)

        self.assertIsInstance(out, FaceShapeClassificationResult)
        self.assertIn(out.shape, self.classifier.classes)
        self.assertGreaterEqual(out.confidence, 0.0)
        self.assertLessEqual(out.confidence, 1.0)

        # Probabilities sum to ~1.0
        prob_sum = sum(out.probabilities.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

        # Feature vector contains all required features
        for name in FEATURE_NAMES:
            self.assertIn(name, out.features)
            self.assertGreater(out.features[name], 0.0)

    def test_classifier_classes_match_spec(self):
        expected_classes = {"oval", "round", "square", "oblong", "heart", "diamond"}
        self.assertEqual(set(self.classifier.classes), expected_classes)


if __name__ == "__main__":
    unittest.main()
