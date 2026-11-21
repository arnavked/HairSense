"""Evaluate the ONNX face-shape model on adjudicated, real-photo labels.

The manifest must contain subject-disjoint train, validation, and test splits.
Only the test split is used for metrics. Multiple test photos for one subject
are averaged into one prediction so each person contributes one vote.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_MANIFEST = WORKSPACE_ROOT / "data" / "face_shape_eval_manifest.csv"
DEFAULT_MODEL = WORKSPACE_ROOT / "models" / "face_shape_classifier.onnx"
DEFAULT_META = WORKSPACE_ROOT / "models" / "face_shape_classifier_meta.json"
DEFAULT_LANDMARKER = WORKSPACE_ROOT / "models" / "face_landmarker.task"

CLASSES = ("oval", "round", "square", "oblong", "heart", "diamond")
REQUIRED_COLUMNS = {
    "image_path",
    "subject_id",
    "face_shape",
    "split",
    "reviewer_ids",
    "annotator_count",
    "blind_review_completed",
    "adjudicated",
    "consent_recorded",
}


def _is_true(value: str | None) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}


def load_manifest(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Read and validate labels, subject grouping, split isolation, and consent."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Copy the template in data/ and add adjudicated labels."
        )

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("The evaluation manifest is empty.")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    if not rows:
        raise ValueError(
            "The manifest has no labeled rows. Real-photo metrics remain unavailable "
            "until consented images receive adjudicated labels."
        )

    seen_images: set[str] = set()
    subject_split: dict[str, str] = {}
    subject_label: dict[str, str] = {}
    counts: dict[str, int] = defaultdict(int)
    split_classes: dict[str, set[str]] = defaultdict(set)

    for line_number, row in enumerate(rows, start=2):
        image_path = (row.get("image_path") or "").strip()
        subject_id = (row.get("subject_id") or "").strip()
        label = (row.get("face_shape") or "").strip().lower()
        split = (row.get("split") or "").strip().lower()

        if not image_path or not subject_id:
            raise ValueError(f"Manifest line {line_number}: image_path and subject_id are required.")
        if label not in CLASSES:
            raise ValueError(
                f"Manifest line {line_number}: unknown face_shape {label!r}; "
                f"expected one of {', '.join(CLASSES)}."
            )
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Manifest line {line_number}: split must be train, validation, or test."
            )
        if not _is_true(row.get("adjudicated")):
            raise ValueError(
                f"Manifest line {line_number}: face_shape must be an adjudicated label."
            )
        if not _is_true(row.get("blind_review_completed")):
            raise ValueError(
                f"Manifest line {line_number}: labels must be independently reviewed blind "
                "to source labels and model predictions."
            )
        if not _is_true(row.get("consent_recorded")):
            raise ValueError(
                f"Manifest line {line_number}: consent_recorded must be true."
            )
        reviewer_ids = {
            reviewer_id.strip()
            for reviewer_id in (row.get("reviewer_ids") or "").split(";")
            if reviewer_id.strip()
        }
        try:
            annotator_count = int(row.get("annotator_count") or "")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Manifest line {line_number}: annotator_count must be an integer."
            ) from exc
        if annotator_count < 2 or len(reviewer_ids) < 2:
            raise ValueError(
                f"Manifest line {line_number}: use at least two independent reviewers and "
                "record their distinct pseudonymous IDs separated by semicolons."
            )
        if annotator_count != len(reviewer_ids):
            raise ValueError(
                f"Manifest line {line_number}: annotator_count must match the number of "
                "distinct reviewer_ids."
            )

        resolved_image = (WORKSPACE_ROOT / image_path).resolve()
        if not resolved_image.is_file():
            raise FileNotFoundError(
                f"Manifest line {line_number}: image does not exist: {image_path}"
            )
        normalized_path = str(resolved_image).casefold()
        if normalized_path in seen_images:
            raise ValueError(f"Manifest line {line_number}: duplicate image_path {image_path!r}.")
        seen_images.add(normalized_path)

        previous_split = subject_split.setdefault(subject_id, split)
        if previous_split != split:
            raise ValueError(
                f"Subject {subject_id!r} appears in both {previous_split!r} and {split!r} splits."
            )
        previous_label = subject_label.setdefault(subject_id, label)
        if previous_label != label:
            raise ValueError(
                f"Subject {subject_id!r} has conflicting adjudicated face-shape labels."
            )

        row["_resolved_image_path"] = str(resolved_image)
        row["_subject_id"] = subject_id
        row["_face_shape"] = label
        row["_split"] = split
        counts[split] += 1
        split_classes[split].add(label)

    if not all(counts.get(split, 0) for split in ("train", "validation", "test")):
        raise ValueError(
            "Manifest must contain train, validation, and test examples. "
            "Keep every subject in exactly one split."
        )
    for split in ("train", "validation", "test"):
        missing_classes = set(CLASSES).difference(split_classes[split])
        if missing_classes:
            raise ValueError(
                f"Manifest {split} split is missing classes: "
                f"{', '.join(sorted(missing_classes))}. Include all six classes in every split."
            )
    return rows, dict(counts)


def evaluate(
    manifest_path: Path,
    model_path: Path,
    meta_path: Path,
    landmarker_path: Path,
) -> dict[str, Any]:
    """Run the frozen classifier on test subjects and return aggregate metrics."""
    from vision.face_classifier import FaceShapeClassifier
    from vision.face_landmarks import FaceLandmarkExtractor

    rows, split_image_counts = load_manifest(manifest_path)
    classifier = FaceShapeClassifier(str(model_path), str(meta_path))
    if tuple(classifier.classes) != CLASSES:
        raise ValueError(
            f"Model classes {classifier.classes!r} do not match expected order {CLASSES!r}."
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        model_metadata = json.load(handle)
    extractor = FaceLandmarkExtractor(model_path=str(landmarker_path), max_num_faces=5)

    subject_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    subject_labels: dict[str, str] = {}
    test_image_count = 0

    for row in rows:
        if row["_split"] != "test":
            continue

        image = cv2.imread(row["_resolved_image_path"])
        if image is None or image.size == 0:
            raise ValueError(f"Could not decode test image: {row['image_path']}")
        detection = extractor.extract(image)
        if detection.face_count != 1:
            raise ValueError(
                f"Expected exactly one face in test image {row['image_path']!r}; "
                f"detected {detection.face_count}. Fix the manifest image before evaluation."
            )

        prediction = classifier.classify(detection.single_face)
        subject = row["_subject_id"]
        subject_probabilities[subject].append(
            np.array([prediction.probabilities[name] for name in CLASSES], dtype=np.float64)
        )
        subject_labels[subject] = row["_face_shape"]
        test_image_count += 1

    subjects = sorted(subject_probabilities)
    y_true = [subject_labels[subject] for subject in subjects]
    y_prob = np.stack(
        [np.mean(subject_probabilities[subject], axis=0) for subject in subjects]
    )
    y_pred = [CLASSES[index] for index in np.argmax(y_prob, axis=1)]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASSES),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "evaluation_type": "real_photo_subject_disjoint_test",
        "model_training_data_source": model_metadata.get("training_data_source", "unspecified"),
        "metric_unit": "subject",
        "test_subject_count": len(subjects),
        "test_image_count": test_image_count,
        "manifest_image_counts_by_split": split_image_counts,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(CLASSES), average="macro", zero_division=0)
        ),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support_subjects": report[name]["support"],
            }
            for name in CLASSES
        },
        "confusion_matrix_labels": list(CLASSES),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(CLASSES)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_META)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report.")
    args = parser.parse_args()

    try:
        result = evaluate(args.manifest, args.model, args.metadata, args.landmarker)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
