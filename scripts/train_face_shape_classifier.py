"""
Train and export face-shape classifier to ONNX.

Fulfills:
- CV-003: Train a synthetic-data baseline classifier (not real-photo validated)
- CV-004: Export classifier to ONNX

Real-photo evaluation is implemented separately in
scripts/evaluate_face_shape_classifier.py and requires adjudicated labels.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Add workspace root to sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from vision.face_features import FEATURE_NAMES

CLASSES = ["oval", "round", "square", "rectangle", "heart", "diamond"]

# Anthropometric distribution parameters per face shape: (mean, std) for each feature
# Order matches FEATURE_NAMES:
# [aspect_ratio, jaw_to_cheek_ratio, forehead_to_cheek_ratio, jaw_to_forehead_ratio, chin_angle_deg, cheek_prominence]
SHAPE_DISTRIBUTIONS = {
    "oval": {
        "aspect_ratio": (1.43, 0.05),
        "jaw_to_cheek_ratio": (0.83, 0.03),
        "forehead_to_cheek_ratio": (0.86, 0.03),
        "jaw_to_forehead_ratio": (0.97, 0.04),
        "chin_angle_deg": (92.0, 3.5),
        "cheek_prominence": (1.18, 0.03),
    },
    "round": {
        "aspect_ratio": (1.15, 0.04),
        "jaw_to_cheek_ratio": (0.87, 0.03),
        "forehead_to_cheek_ratio": (0.86, 0.03),
        "jaw_to_forehead_ratio": (1.01, 0.04),
        "chin_angle_deg": (108.0, 4.0),
        "cheek_prominence": (1.16, 0.03),
    },
    "square": {
        "aspect_ratio": (1.15, 0.04),
        "jaw_to_cheek_ratio": (0.98, 0.03),
        "forehead_to_cheek_ratio": (0.96, 0.03),
        "jaw_to_forehead_ratio": (1.02, 0.03),
        "chin_angle_deg": (101.0, 3.5),
        "cheek_prominence": (1.03, 0.02),
    },
    "rectangle": {
        "aspect_ratio": (1.56, 0.06),
        "jaw_to_cheek_ratio": (0.96, 0.03),
        "forehead_to_cheek_ratio": (0.95, 0.03),
        "jaw_to_forehead_ratio": (1.01, 0.03),
        "chin_angle_deg": (96.0, 3.5),
        "cheek_prominence": (1.04, 0.02),
    },
    "heart": {
        "aspect_ratio": (1.36, 0.05),
        "jaw_to_cheek_ratio": (0.72, 0.03),
        "forehead_to_cheek_ratio": (0.98, 0.03),
        "jaw_to_forehead_ratio": (0.73, 0.03),
        "chin_angle_deg": (76.0, 3.5),
        "cheek_prominence": (1.18, 0.03),
    },
    "diamond": {
        "aspect_ratio": (1.40, 0.05),
        "jaw_to_cheek_ratio": (0.71, 0.03),
        "forehead_to_cheek_ratio": (0.74, 0.03),
        "jaw_to_forehead_ratio": (0.96, 0.04),
        "chin_angle_deg": (78.0, 3.5),
        "cheek_prominence": (1.38, 0.04),
    },
}


def generate_dataset(samples_per_class: int = 1500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    x_list = []
    y_list = []

    for class_idx, class_name in enumerate(CLASSES):
        dist = SHAPE_DISTRIBUTIONS[class_name]
        class_samples = np.zeros((samples_per_class, len(FEATURE_NAMES)), dtype=np.float32)
        for f_idx, feat_name in enumerate(FEATURE_NAMES):
            mean, std = dist[feat_name]
            class_samples[:, f_idx] = rng.normal(loc=mean, scale=std, size=samples_per_class)
        
        # Add slight cross-feature correlations / sensor jitter
        jitter = rng.normal(0, 0.015, class_samples.shape)
        class_samples = np.clip(class_samples + jitter, a_min=0.1, a_max=180.0)

        x_list.append(class_samples)
        y_list.append(np.full(samples_per_class, fill_value=class_idx, dtype=np.int64))

    X = np.vstack(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    # Shuffle
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_and_export():
    print(f"Generating synthetic training dataset ({len(CLASSES)} classes, {len(FEATURE_NAMES)} features)...")
    X, y = generate_dataset(samples_per_class=1500, seed=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Pipeline: StandardScaler -> MLPClassifier. Synthetic data do not support
    # claims about real-photo accuracy or probability calibration.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            alpha=0.001,
            early_stopping=True,
            n_iter_no_change=15,
        )),
    ])

    print("Training MLP classifier...")
    pipeline.fit(X_train, y_train)

    # Synthetic-data diagnostic only; real-photo validation remains open.
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("\n--- Synthetic Feature Holdout Diagnostic (not real-photo validation) ---")
    print(f"Synthetic holdout accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASSES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # CV-004: Export to ONNX
    output_dir = os.path.join(WORKSPACE_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "face_shape_classifier.onnx")
    meta_path = os.path.join(output_dir, "face_shape_classifier_meta.json")

    print(f"\nExporting to ONNX at {onnx_path}...")
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        options={id(pipeline): {"zipmap": False}},
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Metadata for runtime
    metadata = {
        "classes": CLASSES,
        "feature_names": list(FEATURE_NAMES),
        "training_data_source": "synthetic_feature_distributions",
        "synthetic_validation_accuracy": float(acc),
        "real_photo_validation": {
            "status": "not_evaluated",
            "sample_count": 0,
        },
        "model_type": "StandardScaler + MLPClassifier(64, 32)",
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model exported successfully. Metadata written to {meta_path}")

    # Verify ONNX Runtime loads and runs correctly
    print("\nValidating ONNX Runtime inference...")
    session = ort.InferenceSession(onnx_path)
    sample_input = X_test[:5]
    input_name = session.get_inputs()[0].name
    ort_outputs = session.run(None, {input_name: sample_input})
    ort_preds = ort_outputs[0]
    ort_probs = ort_outputs[1]

    skl_preds = pipeline.predict(sample_input)
    skl_probs = pipeline.predict_proba(sample_input)

    np.testing.assert_array_equal(ort_preds, skl_preds)
    np.testing.assert_allclose(ort_probs, skl_probs, rtol=1e-4, atol=1e-4)
    print("Verification passed! ONNX Runtime output matches Scikit-learn outputs exactly.")


if __name__ == "__main__":
    train_and_export()
