"""Extract HairSense's geometric features from a labeled face-shape dataset.

The output preserves the source label and the dataset's original train/test
split. It does not relabel ``oblong`` as ``rectangle`` or create labels for
classes that are absent from the source.

Example:
    python scripts/prepare_face_shape_features.py
    python scripts/prepare_face_shape_features.py --limit 25 --output data/processed/sample.csv --overwrite
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from vision.face_features import FEATURE_NAMES, extract_face_shape_features
from vision.schemas import Landmark

DEFAULT_DATASET = WORKSPACE_ROOT / "archive (7)" / "FaceShape Dataset"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "face_landmarker.task"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "data" / "processed" / "face_shape_features.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_NAMES = {"training_set": "train", "testing_set": "test"}
CSV_COLUMNS = [
    "image_path",
    "source_label",
    "source_split",
    "sha256",
    "image_width",
    "image_height",
    "face_count",
    "status",
    *FEATURE_NAMES,
    "error",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_images(dataset_dir: Path) -> list[tuple[Path, str, str]]:
    rows = []
    for source_split, split in SPLIT_NAMES.items():
        split_dir = dataset_dir / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected dataset split directory: {split_dir}")
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            source_label = class_dir.name.strip().lower()
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    rows.append((path, source_label, split))
    if not rows:
        raise ValueError(f"No supported images found under {dataset_dir}")
    return rows


def _extract_row(path: Path, source_label: str, split: str, landmarker) -> dict[str, object]:
    relative_path = path.resolve().relative_to(WORKSPACE_ROOT).as_posix()
    row: dict[str, object] = {
        "image_path": relative_path,
        "source_label": source_label,
        "source_split": split,
        "sha256": _sha256(path),
        "image_width": "",
        "image_height": "",
        "face_count": 0,
        "status": "",
        **{name: "" for name in FEATURE_NAMES},
        "error": "",
    }

    try:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None or image_bgr.size == 0:
            row["status"] = "decode_error"
            return row
        height, width = image_bgr.shape[:2]
        row["image_width"] = width
        row["image_height"] = height
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = landmarker.detect(mp_image)
        faces = result.face_landmarks
        row["face_count"] = len(faces)
        if len(faces) != 1:
            row["status"] = "no_face" if not faces else "multiple_faces"
            return row

        landmarks = tuple(Landmark(x=point.x, y=point.y, z=point.z) for point in faces[0])
        features = extract_face_shape_features(landmarks, image_width=width, image_height=height)
        row.update(features)
        row["status"] = "ok"
    except Exception as exc:  # Keep a row for each source image and make failures reviewable.
        row["status"] = "processing_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Process only the first N images, for a quick smoke run.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output CSV.")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    model_path = args.landmarker.resolve()
    output_path = args.output.resolve()
    if not model_path.is_file():
        parser.error(f"MediaPipe landmarker model not found: {model_path}")
    if output_path.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output_path}; pass --overwrite to replace it.")

    images = _discover_images(dataset_dir)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be a positive integer.")
        images = images[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    totals: dict[str, int] = {}
    records: list[dict[str, object]] = []
    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            for index, (path, label, split) in enumerate(images, start=1):
                row = _extract_row(path, label, split, landmarker)
                records.append(row)
                status = str(row["status"])
                totals[status] = totals.get(status, 0) + 1
                if index % 250 == 0 or index == len(images):
                    print(f"Processed {index}/{len(images)} images; statuses={totals}", flush=True)

        # Write only after native image libraries have finished. This keeps
        # their diagnostics on the console and out of the machine-readable CSV.
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
        with temporary_path.open("r", newline="", encoding="utf-8") as handle:
            written_rows = list(csv.reader(handle))
        if len(written_rows) != len(images) + 1 or any(
            len(row) != len(CSV_COLUMNS) for row in written_rows
        ):
            raise ValueError("Generated CSV failed its row-count or column-count integrity check.")
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print(f"Feature CSV written to {output_path}")
    print("Source labels and train/test splits were retained without remapping.")
    print("Treat these features as exploratory until source-label quality and subject overlap are verified.")


if __name__ == "__main__":
    main()
