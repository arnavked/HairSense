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

## Data used

After excluding multi-face rows and deduplicating training images, the model used 3,991 training images and 999 test images. No exact image hash crossed the supplied train/test split. Subject IDs were not included, so identity overlap and image-level leakage cannot be ruled out.

The uploaded archive does not include an annotation rubric, annotator agreement, or consent/rights records. Its source license also needs confirmation from the original dataset page.

## Next work

1. Review a sample of the Oval errors and resolve the source label rubric with independent reviewers.
2. Add Diamond examples before training a real-data six-class model.
3. Build a consented, adjudicated, subject-disjoint validation/test set before making a production accuracy claim.

The reproducible experiment is in `scripts/train_face_shape_exploratory.py`. Its JSON report and joblib model are generated under `data/processed/`, which is ignored by Git along with the uploaded photos and feature CSV.
