"""Create a blind, balanced annotation packet from the training split.

The packet hides each image's source class from annotators and includes a
separate answer key. It samples distinct image files, not distinct people;
subject IDs must come from source metadata or participant records.

Example:
    python scripts/create_face_shape_label_review.py --per-class 50
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_FEATURES = WORKSPACE_ROOT / "data" / "processed" / "face_shape_features.csv"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "data" / "processed" / "face_shape_label_review"
CLASSES = ("heart", "oblong", "oval", "round", "square")
REQUIRED_COLUMNS = {"image_path", "source_label", "source_split", "sha256", "status"}
ANNOTATION_COLUMNS = [
    "review_id",
    "image_file",
    "subject_id",
    "subject_id_source",
    "outline_visibility",
    "pose",
    "label",
    "confidence_1_to_3",
    "review_notes",
]
KEY_COLUMNS = ["review_id", "source_label", "source_split", "original_image_path", "sha256"]


def _read_candidates(path: Path) -> dict[str, list[dict[str, str]]]:
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
    if any(None in row or not row.get("image_path") for row in rows):
        raise ValueError("Feature CSV contains malformed rows.")

    candidates = {label: [] for label in CLASSES}
    seen_hashes: dict[str, str] = {}
    for row in rows:
        if row["source_split"] != "train" or row["status"] != "ok":
            continue
        label = row["source_label"]
        if label not in candidates:
            continue
        digest = row["sha256"].strip()
        if not digest:
            raise ValueError(f"Missing SHA-256 digest for {row['image_path']}.")
        previous_label = seen_hashes.get(digest)
        if previous_label is not None:
            if previous_label != label:
                raise ValueError("An exact duplicate image has conflicting source labels.")
            # Drop additional copies of the same image within its source class.
            continue
        seen_hashes[digest] = label
        candidates[label].append(row)
    return candidates


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_packet(features_path: Path, output_dir: Path, per_class: int, seed: int) -> None:
    if per_class < 1:
        raise ValueError("per_class must be a positive integer.")
    if output_dir.exists():
        raise FileExistsError(f"Review packet already exists: {output_dir}")

    candidates = _read_candidates(features_path)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for label in CLASSES:
        if len(candidates[label]) < per_class:
            raise ValueError(
                f"Need {per_class} unique, single-face training images for {label}; "
                f"found {len(candidates[label])}."
            )
        selected.extend(rng.sample(candidates[label], per_class))
    rng.shuffle(selected)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    coordinator_dir = output_dir / "coordinator_only"
    coordinator_dir.mkdir()
    annotator_a: list[dict[str, str]] = []
    annotator_b: list[dict[str, str]] = []
    answer_key: list[dict[str, str]] = []
    adjudication: list[dict[str, str]] = []

    for index, source_row in enumerate(selected, start=1):
        review_id = f"R{index:04d}"
        relative_path = Path(source_row["image_path"])
        source_path = (WORKSPACE_ROOT / relative_path).resolve()
        if not source_path.is_relative_to(WORKSPACE_ROOT) or not source_path.is_file():
            raise FileNotFoundError(f"Invalid or missing image path: {source_row['image_path']}")
        image_file = f"images/{review_id}.jpg"
        destination = output_dir / image_file
        with Image.open(source_path) as source_image:
            review_image = ImageOps.exif_transpose(source_image).convert("RGB")
            # Re-save without EXIF so names, orientation tags, or source metadata
            # cannot reveal the folder label during the blind annotation pass.
            review_image.save(destination, format="JPEG", quality=95, optimize=True)

        blank_row = {
            "review_id": review_id,
            "image_file": image_file,
            "subject_id": "",
            "subject_id_source": "",
            "outline_visibility": "",
            "pose": "",
            "label": "",
            "confidence_1_to_3": "",
            "review_notes": "",
        }
        annotator_a.append(blank_row.copy())
        annotator_b.append(blank_row.copy())
        answer_key.append({
            "review_id": review_id,
            "source_label": source_row["source_label"],
            "source_split": source_row["source_split"],
            "original_image_path": source_row["image_path"],
            "sha256": source_row["sha256"],
        })
        adjudication.append({
            "review_id": review_id,
            "annotator_a_label": "",
            "annotator_a_confidence_1_to_3": "",
            "annotator_b_label": "",
            "annotator_b_confidence_1_to_3": "",
            "adjudicated_label": "",
            "adjudication_notes": "",
        })

    _write_csv(output_dir / "annotator_a.csv", ANNOTATION_COLUMNS, annotator_a)
    _write_csv(output_dir / "annotator_b.csv", ANNOTATION_COLUMNS, annotator_b)
    _write_csv(coordinator_dir / "adjudication.csv", list(adjudication[0]), adjudication)
    _write_csv(coordinator_dir / "answer_key.csv", KEY_COLUMNS, answer_key)

    readme = f"""# Face-Shape Label Review Packet

This packet contains {len(selected)} randomly selected, one-face images from the training split: {per_class} per source class. It is a starter quality audit, not a representative accuracy estimate. Sampling used seed {seed} and removed exact duplicate image files first.

## Blind review

1. Give `annotator_a.csv` and `annotator_b.csv` to two reviewers separately. Do not share the answer key, source folder labels, or model predictions before both are complete.
2. Open the images under `images/` using the `image_file` value. Labels may be `heart`, `oblong`, `oval`, `round`, `square`, `diamond`, or `unclear`.
3. For `outline_visibility`, use `clear`, `partly_hidden`, or `poor`. For `pose`, use `frontal`, `slight_turn`, `oblique`, or `unclear`. Confidence is 1 (low) to 3 (high).
4. Enter a `subject_id` only when source records or participant records support it. Use a pseudonymous ID, not a person's name. Do not infer identity from a face; use `unknown` when records do not establish it. The source dataset has no supplied subject IDs, so this field may remain unknown.
5. After both independent sheets are complete, the coordinator copies results into `coordinator_only/adjudication.csv`, resolves disagreements, then consults `coordinator_only/answer_key.csv` to compare audited labels with original folder labels. Do not give reviewers the `coordinator_only/` directory.

After both sheets are complete, run `python scripts/analyze_face_shape_label_review.py` to summarize reviewer agreement without reading the source-label answer key. After adjudication, run `python scripts/analyze_face_shape_label_review.py --compare-source-labels` for a separate comparison with the original folder labels. Source labels are not verified ground truth.

The renamed JPEGs are local review copies with EXIF metadata removed. Keep this packet and the original face images out of public Git commits.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"Created {len(selected)} blind review images at {output_dir}")
    print(f"Sampled {per_class} images per class; labels are in a separate answer_key.csv.")
    print("This samples distinct image hashes, not verified distinct people.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        create_packet(args.features.resolve(), args.output_dir.resolve(), args.per_class, args.seed)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
