# Five-Class Face-Shape Feature Pilot

## Purpose and scope

This is an exploratory experiment using the uploaded dataset and HairSense's existing MediaPipe geometry features. It keeps the source classes as `heart`, `oblong`, `oval`, `round`, and `square`. It does not alter or replace the six-class production model.

The pipeline selected a model using a stratified 20% image-level validation split from the supplied training folder, then evaluated the selected model once on the supplied testing folder. Exact duplicate images were removed from training; images with multiple detected faces were excluded. The selected model is logistic regression on standardized features.

## Result

| Supplied test metric | Result |
|---|---:|
| Test images | 999 |
| Accuracy | 56.6% |
| Balanced accuracy | 56.5% |
| Macro F1 | 55.7% |

Models were selected only on the internal validation split:

| Candidate | Validation balanced accuracy | Validation macro F1 |
|---|---:|---:|
| Majority baseline | 20.0% | 6.7% |
| Logistic regression | 55.7% | 55.2% |
| RBF SVM | 54.1% | 53.6% |
| MLP | 55.0% | 54.2% |
| Random forest | 52.8% | 52.6% |

Logistic regression was selected by validation macro F1. It scored 55.7% macro F1 on the supplied test split.

| Source label | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Heart | 45.6% | 49.2% | 47.3% | 199 |
| Oblong | 71.2% | 80.5% | 75.6% | 200 |
| Oval | 43.2% | 31.5% | 36.4% | 200 |
| Round | 54.5% | 51.0% | 52.7% | 200 |
| Square | 62.7% | 70.5% | 66.4% | 200 |

Confusion matrix (rows are true source labels; columns are predicted labels):

| True \\ Predicted | Heart | Oblong | Oval | Round | Square |
|---|---:|---:|---:|---:|---:|
| Heart | 98 | 29 | 37 | 20 | 15 |
| Oblong | 23 | 161 | 13 | 1 | 2 |
| Oval | 64 | 28 | 63 | 30 | 15 |
| Round | 16 | 4 | 26 | 102 | 52 |
| Square | 14 | 4 | 7 | 34 | 141 |

The pilot separates Oblong and Square more reliably than Oval; Oval recall is 31.5%. This points to substantial overlap or noise in these six geometric features and/or the source labels. It does not establish real-world accuracy.

## Oval error review

The complete confusion matrix shows 137 of 200 source-labeled Oval test images were misclassified: 64 as Heart, 30 as Round, 28 as Oblong, and 15 as Square. The earlier discussion focused on the first three groups; the Square group is included here as well.

I visually inspected 3 confidence-spanning examples from each predicted error class (12 error examples), along with 3 correctly predicted Oval examples. Several images have hair obscuring the forehead or jaw edge, close crops, or other presentation differences. Some shapes look plausible under more than one class; the sample is not enough to prove that any particular source label is wrong. Two reviewed images also show visible source-site watermarks. Keep the dataset's source and usage terms under review.

The measured features show why these classes are hard to separate. Correctly predicted Oval images and Oval images predicted as Heart have almost the same mean aspect ratio (1.198 vs. 1.193) and jaw-to-cheek ratio (0.876 vs. 0.877). The model separates those groups more through subtler features: mean forehead-to-cheek ratio is 0.846 for correctly predicted Oval and 0.878 for Oval predicted as Heart; mean chin angle is 101.1° vs. 96.3°. Oval images predicted as Round have a lower mean aspect ratio (1.149), while those predicted as Oblong have a higher one (1.254). The six-feature representation captures meaningful trends, but the source classes overlap in it.

No source labels were changed during this review. The full 137-row list is in the local, Git-ignored `data/processed/oval_error_review.csv`, with blank fields for human review. Because this test split has now been inspected for errors, treat it as a diagnostic set; use a new, untouched, subject-disjoint set for final accuracy reporting after changes.

## Blind training-label audit status

A separate packet contains 250 random, single-face images from the training folder (50 per source class), with the source labels hidden from reviewers. The two reviewer sheets and adjudication file are currently blank, so no agreement or label-correction estimate is available yet. This packet is a label-quality audit; it is not a new evaluation set and does not establish unique subjects.

After two reviewers complete the sheets independently, run `python scripts/analyze_face_shape_label_review.py` for agreement and outline/pose summaries. Once adjudication is complete, run it with `--compare-source-labels` to compare the human review with the original folder labels. Source labels are a comparison target, not verified ground truth. Keep the packet and generated report under ignored `data/processed/`.

## Data used

After excluding multi-face rows and deduplicating training images, the model used 3,991 training images and 999 test images. No exact image hash crossed the supplied train/test split. Subject IDs were not included, so identity overlap and image-level leakage cannot be ruled out.

The uploaded archive does not include an annotation rubric, annotator agreement, or consent/rights records. Its source license also needs confirmation from the original dataset page.

## Next work

1. Complete the independent review of the 250-image training packet and quantify label agreement and outline visibility.
2. If the sample shows meaningful disagreement or poor visibility, refine the rubric and review a larger training subset before retraining the five-class pilot.
3. Recover subject IDs only from source records, and use them to group every known image of a subject into one split. If reliable IDs cannot be recovered, treat image-level results as exploratory and collect a new, consented, subject-disjoint evaluation cohort.
4. Keep Diamond collection for the later six-class model; it is outside the current five-class pilot.

The reproducible experiment is in `scripts/train_face_shape_exploratory.py`. Its JSON report and joblib model are generated under `data/processed/`, which is ignored by Git along with the uploaded photos and feature CSV.
