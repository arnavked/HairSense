"""Train a five-class exploratory model from prepared face-shape features.

This pilot keeps the uploaded dataset's original labels, including ``oblong``.
It does not update or replace HairSense's six-class production classifier.
Metrics use the dataset's supplied image split and are not subject-disjoint.

Example:
    python scripts/train_face_shape_exploratory.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from vision.face_features import FEATURE_NAMES

DEFAULT_FEATURES = WORKSPACE_ROOT / "data" / "processed" / "face_shape_features.csv"
DEFAULT_REPORT = WORKSPACE_ROOT / "data" / "processed" / "face_shape_exploratory_report.json"
DEFAULT_MODEL = WORKSPACE_ROOT / "data" / "processed" / "face_shape_5class_exploratory.joblib"
CLASSES = ("heart", "oblong", "oval", "round", "square")
REQUIRED_COLUMNS = {
    "image_path", "source_label", "source_split", "sha256", "status", *FEATURE_NAMES
}


def _load_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature CSV not found: {path}. Run scripts/prepare_face_shape_features.py first."
        )
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Feature CSV has no header.")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Feature CSV is missing columns: {', '.join(sorted(missing))}.")
        rows = list(reader)
    if not rows:
        raise ValueError("Feature CSV is empty.")
    if any(None in row or not row.get("image_path") for row in rows):
        raise ValueError("Feature CSV contains malformed rows.")
    if any(row["source_label"] not in CLASSES for row in rows):
        raise ValueError("Feature CSV contains labels outside the expected source taxonomy.")
    if any(row["source_split"] not in {"train", "test"} for row in rows):
        raise ValueError("Feature CSV source_split must be train or test.")
    return rows, dict(Counter(row["status"] for row in rows))


def _usable_unique_rows(
    rows: list[dict[str, str]], split: str
) -> tuple[list[dict[str, str]], int]:
    usable = [
        row for row in rows
        if row["source_split"] == split and row["status"] == "ok"
    ]
    by_digest: dict[str, dict[str, str]] = {}
    for row in usable:
        digest = row["sha256"].strip()
        if not digest:
            raise ValueError(f"Missing SHA-256 digest for {row['image_path']}.")
        previous = by_digest.get(digest)
        if previous is not None:
            if previous["source_label"] != row["source_label"]:
                raise ValueError("An exact duplicate image has conflicting source labels.")
            continue
        by_digest[digest] = row
    return list(by_digest.values()), len(usable) - len(by_digest)


def _candidates() -> dict[str, Any]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ]),
        "rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=1.0, gamma="scale", class_weight="balanced")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.15,
                alpha=0.001,
                random_state=42,
            )),
        ]),
    }


def run_experiment(feature_path: Path, report_path: Path, model_path: Path) -> dict[str, Any]:
    rows, status_counts = _load_rows(feature_path)
    train_rows, dropped_train_duplicates = _usable_unique_rows(rows, "train")
    test_rows, dropped_test_duplicates = _usable_unique_rows(rows, "test")
    train_digests = {row["sha256"] for row in train_rows}
    test_digests = {row["sha256"] for row in test_rows}
    overlap = train_digests & test_digests
    if overlap:
        raise ValueError(f"Found {len(overlap)} exact image duplicates across train and test.")
    if set(row["source_label"] for row in train_rows) != set(CLASSES):
        raise ValueError("Training rows must contain all five source classes.")
    if set(row["source_label"] for row in test_rows) != set(CLASSES):
        raise ValueError("Testing rows must contain all five source classes.")

    def arrays(selected_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(
            [[float(row[name]) for name in FEATURE_NAMES] for row in selected_rows],
            dtype=np.float32,
        )
        y = np.asarray([row["source_label"] for row in selected_rows], dtype=str)
        if not np.isfinite(x).all():
            raise ValueError("Feature CSV contains a non-finite feature value.")
        return x, y

    X_train, y_train = arrays(train_rows)
    X_test, y_test = arrays(test_rows)
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    validation_results = {}
    candidates = _candidates()
    for name, candidate in candidates.items():
        candidate.fit(X_fit, y_fit)
        prediction = candidate.predict(X_validation)
        validation_results[name] = {
            "accuracy": float(accuracy_score(y_validation, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_validation, prediction)),
            "macro_f1": float(f1_score(
                y_validation, prediction, labels=list(CLASSES), average="macro", zero_division=0
            )),
        }

    selected_name = max(
        validation_results,
        key=lambda name: (
            validation_results[name]["macro_f1"],
            validation_results[name]["balanced_accuracy"],
        ),
    )
    final_model = clone(candidates[selected_name])
    final_model.fit(X_train, y_train)
    test_prediction = final_model.predict(X_test)
    class_report = classification_report(
        y_test,
        test_prediction,
        labels=list(CLASSES),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )

    def label_counts(selected_rows: list[dict[str, str]]) -> dict[str, int]:
        counts = Counter(row["source_label"] for row in selected_rows)
        return {label: int(counts.get(label, 0)) for label in CLASSES}

    multi_face_counts = {
        split: dict(Counter(
            row["source_label"] for row in rows
            if row["source_split"] == split and row["status"] == "multiple_faces"
        ))
        for split in ("train", "test")
    }
    report = {
        "experiment_type": "exploratory_five_class_face_shape_feature_model",
        "model_is_production": False,
        "source_labels": list(CLASSES),
        "feature_names": list(FEATURE_NAMES),
        "input_feature_csv": str(feature_path),
        "image_status_counts": status_counts,
        "usable_unique_image_counts": {
            "train": label_counts(train_rows),
            "test": label_counts(test_rows),
        },
        "excluded_multiple_face_counts": multi_face_counts,
        "exact_duplicate_records_removed": {
            "train": dropped_train_duplicates,
            "test": dropped_test_duplicates,
        },
        "exact_train_test_hash_overlap": 0,
        "validation": {
            "split_method": "stratified image-level 80/20 holdout from the supplied training split",
            "model_selection_metric": "macro_f1, with balanced_accuracy as tie-breaker",
            "candidate_scores": validation_results,
            "selected_model": selected_name,
        },
        "supplied_test_split": {
            "metric_unit": "image",
            "sample_count": len(y_test),
            "accuracy": float(accuracy_score(y_test, test_prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, test_prediction)),
            "macro_f1": float(f1_score(
                y_test, test_prediction, labels=list(CLASSES), average="macro", zero_division=0
            )),
            "per_class": {
                label: {
                    "precision": class_report[label]["precision"],
                    "recall": class_report[label]["recall"],
                    "f1": class_report[label]["f1-score"],
                    "support_images": class_report[label]["support"],
                }
                for label in CLASSES
            },
            "confusion_matrix_labels": list(CLASSES),
            "confusion_matrix": confusion_matrix(
                y_test, test_prediction, labels=list(CLASSES)
            ).tolist(),
        },
        "limitations": [
            "The source covers five of HairSense's six classes; diamond is absent.",
            "The source train/test split has no subject identifiers, so subject overlap cannot be ruled out.",
            "The archive provides no annotation rubric, annotator agreement, or consent/rights records.",
            "This is a feature-only experiment on MediaPipe outputs, not an end-to-end image model or a production classifier.",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    joblib.dump(final_model, model_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        report = run_experiment(args.features.resolve(), args.report.resolve(), args.model.resolve())
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    print(f"Exploratory report written to {args.report.resolve()}")
    print(f"Exploratory model written to {args.model.resolve()}")


if __name__ == "__main__":
    main()
