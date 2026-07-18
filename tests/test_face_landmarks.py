import os
import unittest

import numpy as np

from vision.face_landmarks import DEFAULT_MODEL_PATH, FaceLandmarkExtractor, ModelNotFoundError
from vision.schemas import BoundingBox, FaceDetectionResult, FaceResult, Landmark

MODEL_AVAILABLE = os.path.exists(DEFAULT_MODEL_PATH)
SKIP_REASON = (
    f"FaceLandmarker model not present at {DEFAULT_MODEL_PATH}. "
    f"Run scripts/download_models.sh (needs network) then re-run tests."
)


class TestModelNotFoundHandling(unittest.TestCase):
    """These run regardless of whether the model file is present."""

    def test_missing_model_raises_actionable_error(self):
        with self.assertRaises(ModelNotFoundError) as ctx:
            FaceLandmarkExtractor(model_path="/definitely/not/a/real/path.task")
        # FR-3-style requirement: error should be specific/actionable, not silent.
        self.assertIn("not found", str(ctx.exception).lower())
        self.assertIn("download_models.sh", str(ctx.exception))


@unittest.skipUnless(MODEL_AVAILABLE, SKIP_REASON)
class TestFaceLandmarkExtractor(unittest.TestCase):
    """
    Requires the real FaceLandmarker model. Not runnable in this sandbox
    (no network) — run these after scripts/download_models.sh in an
    environment with network access, ideally against real test photos too
    (see vision/debug_draw.py for visual verification).
    """

    @classmethod
    def setUpClass(cls):
        cls.extractor = FaceLandmarkExtractor()

    def test_blank_image_yields_zero_faces(self):
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.extractor.extract(blank)
        self.assertIsInstance(result, FaceDetectionResult)
        self.assertEqual(result.face_count, 0)
        self.assertIsNone(result.single_face)

    def test_noise_image_does_not_crash(self):
        rng = np.random.default_rng(seed=42)
        noise = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        result = self.extractor.extract(noise)
        self.assertIsInstance(result, FaceDetectionResult)
        self.assertEqual(result.image_width, 640)
        self.assertEqual(result.image_height, 480)

    def test_empty_image_raises(self):
        with self.assertRaises(ValueError):
            self.extractor.extract(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_none_image_raises(self):
        with self.assertRaises(ValueError):
            self.extractor.extract(None)

    def test_extract_from_missing_path_raises(self):
        with self.assertRaises(ValueError):
            self.extractor.extract_from_path("/nonexistent/path/does_not_exist.jpg")


class TestSchemas(unittest.TestCase):
    """Pure dataclass behavior — no model dependency."""

    def test_bounding_box_width_height(self):
        bbox = BoundingBox(x_min=0.2, y_min=0.1, x_max=0.8, y_max=0.9)
        self.assertAlmostEqual(bbox.width, 0.6)
        self.assertAlmostEqual(bbox.height, 0.8)

    def test_face_detection_result_single_face_helper(self):
        face = FaceResult(
            bounding_box=BoundingBox(0, 0, 1, 1),
            landmarks=(Landmark(0.5, 0.5, 0.0),),
            detection_confidence=None,
        )
        result_one = FaceDetectionResult(image_width=100, image_height=100, faces=(face,))
        self.assertIs(result_one.single_face, face)

        result_two = FaceDetectionResult(image_width=100, image_height=100, faces=(face, face))
        self.assertIsNone(result_two.single_face)
        self.assertEqual(result_two.face_count, 2)


if __name__ == "__main__":
    unittest.main()