# Uploaded Face-Shape Dataset Audit

## Inventory

The uploaded `archive (7).zip` contains an extracted `FaceShape Dataset` directory with 5,000 JPEGs:

| Source split | Heart | Oblong | Oval | Round | Square | Total |
|---|---:|---:|---:|---:|---:|---:|
| Training | 800 | 800 | 800 | 800 | 800 | 4,000 |
| Testing | 200 | 200 | 200 | 200 | 200 | 1,000 |

All 5,000 images decoded during the inventory scan. Image dimensions vary. The zip contains no dataset card, annotation rubric, annotator details, subject identifiers, or consent/rights records. The folder layout resembles the five-class dataset previously discussed, but the archive alone does not establish its source or license; retain the source page and confirm those terms before redistribution or publication.

An exact SHA-256 scan found four duplicate image pairs, all within the training split and all within one class. No exact duplicate crosses from training into testing. Remove the extra copy in each pair before training so those images do not receive extra weight.

## Fit for HairSense

HairSense now uses six labels: `oval`, `round`, `square`, `oblong`, `heart`, and `diamond`. The upload directly covers five of them; it has no `diamond` examples. Keep the source label `oblong` in derived data. This dataset alone cannot train the six-class model.

The supplied split is image-level. There are no subject IDs, so the split cannot be checked for the same person appearing in both train and test. It is suitable for an exploratory five-class feature experiment after reviewing label mapping and duplicates, but the test folder is not a defensible subject-disjoint evaluation set. The project's real-photo evaluation protocol also calls for independently reviewed/adjudicated labels and documented permission; those facts are not provided in this archive.

## Data preparation

Run `python scripts/prepare_face_shape_features.py` to produce `data/processed/face_shape_features.csv`. The script reuses one MediaPipe landmarker for batch processing and records each image's relative path, original label, original split, SHA-256 digest, face count, extraction status, and HairSense geometric features. It preserves source labels and emits rows for images where exactly one face was not detected so they can be reviewed.

The full extraction pass produced 5,000 well-formed CSV records: 4,994 with one detected face and usable features, and 6 with multiple detected faces. There were no image decode failures or zero-face results. The multiple-face images are flagged for manual review rather than included as single-face training examples.

The uploaded photos and generated feature CSV are ignored by Git. Do not add the raw photos or a private evaluation manifest to a public commit.

## Recommended next step

1. Confirm the dataset's source/license and whether its labels were independently checked.
2. Review the Oval errors and refine the geometric features or labeling rubric.
3. Add a separately sourced and reviewed `diamond` set before attempting a real-data six-class training run.
4. Build a private, subject-disjoint validation/test set with adjudicated labels for final model assessment.
