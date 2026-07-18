"""
Not part of the product pipeline — a debug utility for visually confirming
that detection + landmarks look right on real photos (the Bucket 1
checkpoint: "landmarks reliably extracted across a range of test photos").

Usage:
    python -m vision.debug_draw path/to/photo.jpg [output_path.jpg]
"""

import sys

import cv2

from vision.face_landmarks import FaceLandmarkExtractor
from vision.landmark_groups import FACE_OVAL, CHEEKBONE_LEFT, CHEEKBONE_RIGHT, CHIN, FOREHEAD_CENTER


def draw_debug_image(image_path: str, output_path: str) -> None:
    extractor = FaceLandmarkExtractor()
    result = extractor.extract_from_path(image_path)

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    print(f"Image: {image_path} ({w}x{h})")
    print(f"Faces detected: {result.face_count}")

    for i, face in enumerate(result.faces):
        bbox = face.bounding_box
        cv2.rectangle(
            image,
            (int(bbox.x_min * w), int(bbox.y_min * h)),
            (int(bbox.x_max * w), int(bbox.y_max * h)),
            (0, 255, 0),
            2,
        )
        # all landmarks, small dots
        for lm in face.landmarks:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 1, (0, 200, 255), -1)

        # highlight key named points used by later buckets
        for idx, color, label in [
            (CHIN, (0, 0, 255), "chin"),
            (FOREHEAD_CENTER, (255, 0, 0), "forehead"),
            (CHEEKBONE_LEFT, (255, 255, 0), "cheek_L"),
            (CHEEKBONE_RIGHT, (255, 255, 0), "cheek_R"),
        ]:
            lm = face.landmarks[idx]
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, pt, 4, color, -1)

        print(f"  Face {i}: bbox width={bbox.width:.3f} height={bbox.height:.3f}")

    cv2.imwrite(output_path, image)
    print(f"Wrote annotated image to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m vision.debug_draw <input_image> [output_image]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "debug_output.jpg"
    draw_debug_image(in_path, out_path)