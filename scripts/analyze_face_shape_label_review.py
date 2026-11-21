"""Summarize the blind five-class label-review packet.

By default this reads only the two blind reviewer sheets and reports
inter-reviewer agreement. Source-folder labels are read only when
``--compare-source-labels`` is passed, after both reviews and adjudication are
complete.

Example:
    python scripts/analyze_face_shape_label_review.py
    python scripts/analyze_face_shape_label_review.py --compare-source-labels
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKET = WORKSPACE_ROOT / "data" / "processed" / "face_shape_label_review"
DEFAULT_OUTPUT = DEFAULT_PACKET / "agreement_report.json"
LABELS = ("heart", "oblong", "oval", "round", "square", "diamond", "unclear")
SOURCE_LABELS = ("heart", "oblong", "oval", "round", "square")
VISIBILITY = ("clear", "partly_hidden", "poor")
POSES = ("frontal", "slight_turn", "oblique", "unclear")


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required review file not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no review rows: {path}")
    return rows


def _load_reviewer(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "review_id",
        "outline_visibility",
        "pose",
        "label",
        "confidence_1_to_3",
    }
    rows = _read_csv(path, required)
    indexed: dict[str, dict[str, str]] = {}
    for line_number, row in enumerate(rows, start=2):
        review_id = (row.get("review_id") or "").strip()
        if not review_id:
            raise ValueError(f"{path.name}:{line_number}: review_id is blank.")
        if review_id in indexed:
            raise ValueError(f"{path.name}:{line_number}: duplicate review_id {review_id}.")

        label = (row.get("label") or "").strip().lower()
        if label not in LABELS:
            raise ValueError(
                f"{path.name}:{line_number}: label must be one of {', '.join(LABELS)}; "
                "finish the sheet before analyzing it."
            )
        visibility = (row.get("outline_visibility") or "").strip().lower()
        if visibility not in VISIBILITY:
            raise ValueError(
                f"{path.name}:{line_number}: outline_visibility must be one of "
                f"{', '.join(VISIBILITY)}."
            )
        pose = (row.get("pose") or "").strip().lower()
        if pose not in POSES:
            raise ValueError(
                f"{path.name}:{line_number}: pose must be one of {', '.join(POSES)}."
            )
        try:
            confidence = int(row.get("confidence_1_to_3") or "")
        except ValueError as exc:
            raise ValueError(
                f"{path.name}:{line_number}: confidence_1_to_3 must be 1, 2, or 3."
            ) from exc
        if confidence not in {1, 2, 3}:
            raise ValueError(f"{path.name}:{line_number}: confidence must be 1, 2, or 3.")

        indexed[review_id] = {
            "label": label,
            "outline_visibility": visibility,
            "pose": pose,
            "confidence_1_to_3": str(confidence),
        }
    return indexed


def _cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a:
        return None
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[label] / len(labels_a)) * (counts_b[label] / len(labels_b))
        for label in LABELS
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def _distribution(values: list[str], categories: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(values)
    return {category: counts.get(category, 0) for category in categories}


def _compare_to_source(
    packet_dir: Path,
    reviewer_a: dict[str, dict[str, str]],
    reviewer_b: dict[str, dict[str, str]],
    ordered_ids: list[str],
) -> dict[str, Any]:
    key_path = packet_dir / "coordinator_only" / "answer_key.csv"
    adjudication_path = packet_dir / "coordinator_only" / "adjudication.csv"
    key_rows = _read_csv(key_path, {"review_id", "source_label"})
    adjudication_rows = _read_csv(
        adjudication_path,
        {"review_id", "annotator_a_label", "annotator_b_label", "adjudicated_label"},
    )

    source_labels: dict[str, str] = {}
    for line_number, row in enumerate(key_rows, start=2):
        review_id = (row.get("review_id") or "").strip()
        source_label = (row.get("source_label") or "").strip().lower()
        if review_id in source_labels:
            raise ValueError(f"answer_key.csv:{line_number}: duplicate review_id {review_id}.")
        if source_label not in SOURCE_LABELS:
            raise ValueError(f"answer_key.csv:{line_number}: invalid source label {source_label!r}.")
        source_labels[review_id] = source_label

    adjudicated: dict[str, str] = {}
    for line_number, row in enumerate(adjudication_rows, start=2):
        review_id = (row.get("review_id") or "").strip()
        if review_id in adjudicated:
            raise ValueError(f"adjudication.csv:{line_number}: duplicate review_id {review_id}.")
        if review_id not in reviewer_a or review_id not in reviewer_b:
            raise ValueError(f"adjudication.csv:{line_number}: unknown review_id {review_id!r}.")
        recorded_a = (row.get("annotator_a_label") or "").strip().lower()
        recorded_b = (row.get("annotator_b_label") or "").strip().lower()
        if recorded_a != reviewer_a[review_id]["label"] or recorded_b != reviewer_b[review_id]["label"]:
            raise ValueError(
                f"adjudication.csv:{line_number}: copied reviewer labels do not match the blind sheets."
            )
        label = (row.get("adjudicated_label") or "").strip().lower()
        if label not in LABELS:
            raise ValueError(
                f"adjudication.csv:{line_number}: enter an adjudicated label from "
                f"{', '.join(LABELS)} before comparing source labels."
            )
        adjudicated[review_id] = label

    if set(source_labels) != set(ordered_ids) or set(adjudicated) != set(ordered_ids):
        raise ValueError("Review sheets, answer key, and adjudication file must contain the same review IDs.")

    def agreement(labels: dict[str, str]) -> dict[str, Any]:
        matches = sum(labels[review_id] == source_labels[review_id] for review_id in ordered_ids)
        return {
            "matches_source_label": matches,
            "total": len(ordered_ids),
            "match_rate": matches / len(ordered_ids),
        }

    per_source_class: dict[str, dict[str, int | float]] = {}
    for source_label in SOURCE_LABELS:
        matching_ids = [review_id for review_id in ordered_ids if source_labels[review_id] == source_label]
        matches = sum(adjudicated[review_id] == source_label for review_id in matching_ids)
        per_source_class[source_label] = {
            "matches": matches,
            "total": len(matching_ids),
            "match_rate": matches / len(matching_ids) if matching_ids else 0.0,
        }

    return {
        "note": "Source-folder labels are a comparison target, not verified ground truth.",
        "reviewer_a": agreement({review_id: reviewer_a[review_id]["label"] for review_id in ordered_ids}),
        "reviewer_b": agreement({review_id: reviewer_b[review_id]["label"] for review_id in ordered_ids}),
        "adjudicated": agreement(adjudicated),
        "per_source_class_adjudicated_match": per_source_class,
        "source_vs_adjudicated_confusion_matrix_rows_source_columns_adjudicated": [
            [
                sum(
                    source_labels[review_id] == source_label
                    and adjudicated[review_id] == adjudicated_label
                    for review_id in ordered_ids
                )
                for adjudicated_label in LABELS
            ]
            for source_label in SOURCE_LABELS
        ],
        "source_confusion_matrix_row_labels": list(SOURCE_LABELS),
        "source_confusion_matrix_column_labels": list(LABELS),
        "adjudicated_label_counts": _distribution(list(adjudicated.values()), LABELS),
    }


def analyze(packet_dir: Path, compare_source_labels: bool) -> dict[str, Any]:
    reviewer_a = _load_reviewer(packet_dir / "annotator_a.csv")
    reviewer_b = _load_reviewer(packet_dir / "annotator_b.csv")
    if set(reviewer_a) != set(reviewer_b):
        only_a = sorted(set(reviewer_a).difference(reviewer_b))
        only_b = sorted(set(reviewer_b).difference(reviewer_a))
        raise ValueError(f"Reviewer sheets have different review IDs; only A: {only_a}; only B: {only_b}.")

    ordered_ids = sorted(reviewer_a)
    labels_a = [reviewer_a[review_id]["label"] for review_id in ordered_ids]
    labels_b = [reviewer_b[review_id]["label"] for review_id in ordered_ids]
    agreement_count = sum(a == b for a, b in zip(labels_a, labels_b))
    report: dict[str, Any] = {
        "packet_image_count": len(ordered_ids),
        "label_categories": list(LABELS),
        "inter_reviewer": {
            "exact_agreement_count": agreement_count,
            "total": len(ordered_ids),
            "exact_agreement_rate": agreement_count / len(ordered_ids),
            "cohen_kappa": _cohen_kappa(labels_a, labels_b),
            "reviewer_a_label_counts": _distribution(labels_a, LABELS),
            "reviewer_b_label_counts": _distribution(labels_b, LABELS),
            "confusion_matrix_rows_reviewer_a_columns_reviewer_b": [
                [
                    sum(a == row_label and b == column_label for a, b in zip(labels_a, labels_b))
                    for column_label in LABELS
                ]
                for row_label in LABELS
            ],
            "outline_visibility_counts": {
                "reviewer_a": _distribution(
                    [reviewer_a[review_id]["outline_visibility"] for review_id in ordered_ids], VISIBILITY
                ),
                "reviewer_b": _distribution(
                    [reviewer_b[review_id]["outline_visibility"] for review_id in ordered_ids], VISIBILITY
                ),
            },
            "pose_counts": {
                "reviewer_a": _distribution(
                    [reviewer_a[review_id]["pose"] for review_id in ordered_ids], POSES
                ),
                "reviewer_b": _distribution(
                    [reviewer_b[review_id]["pose"] for review_id in ordered_ids], POSES
                ),
            },
        },
    }
    if compare_source_labels:
        report["source_label_comparison"] = _compare_to_source(
            packet_dir, reviewer_a, reviewer_b, ordered_ids
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compare-source-labels",
        action="store_true",
        help="After adjudication is complete, compare the final review with source folder labels.",
    )
    args = parser.parse_args()
    try:
        report = analyze(args.packet_dir.resolve(), args.compare_source_labels)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
