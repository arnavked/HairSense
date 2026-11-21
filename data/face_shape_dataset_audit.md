# Uploaded Face-Shape Dataset Audit

## Inventory

The uploaded `archive (7).zip` contains an extracted `FaceShape Dataset` directory with 5,000 JPEGs:

| Source split | Heart | Oblong | Oval | Round | Square | Total |
|---|---:|---:|---:|---:|---:|---:|
| Training | 800 | 800 | 800 | 800 | 800 | 4,000 |
| Testing | 200 | 200 | 200 | 200 | 200 | 1,000 |

All 5,000 images decoded during the inventory scan. The folder structure and counts match the [Kaggle Face Shape Dataset listing](https://www.kaggle.com/datasets/niten19/face-shape-dataset), which describes the images as female celebrities and lists a CC0 license. The archive itself contains no source URLs, annotation rubric, annotator details, or subject identifiers. The Kaggle listing establishes the dataset-level license claim, but not how each image's label was assigned or whether a person appears in more than one file. The images also include visible watermarks in some cases, so verify image-level provenance before redistributing them.

An exact SHA-256 scan found four duplicate image pairs, all within the training split and all within one class. No exact duplicate crosses from training into testing. Remove the extra copy in each pair before training so those images do not receive extra weight.

## Fit for HairSense

HairSense now uses six labels: `oval`, `round`, `square`, `oblong`, `heart`, and `diamond`. The upload directly covers five of them; it has no `diamond` examples. Keep the source label `oblong` in derived data. This dataset alone cannot train the six-class model.

The supplied split is image-level. There are no subject IDs, so the split cannot be checked for the same person appearing in both train and test. It is suitable for an exploratory five-class feature experiment after reviewing labels and duplicates, but the test folder is not a defensible subject-disjoint evaluation set. This test split has now also been inspected for Oval errors, so keep it for diagnostics and reserve a new, untouched test set for final metrics. Because the Kaggle listing describes female celebrity images, this dataset alone may not represent HairSense's intended users if the product is meant for a broader population. The project's real-photo evaluation protocol expects independently adjudicated labels and recorded permission; the archive does not provide annotation records or participant-consent records. The Kaggle CC0 listing is a dataset license claim, not participant metadata.

## Data preparation

Run `python scripts/prepare_face_shape_features.py` to produce `data/processed/face_shape_features.csv`. The script reuses one MediaPipe landmarker for batch processing and records each image's relative path, original label, original split, SHA-256 digest, face count, extraction status, and HairSense geometric features. It preserves source labels and emits rows for images where exactly one face was not detected so they can be reviewed.

The full extraction pass produced 5,000 well-formed CSV records: 4,994 with one detected face and usable features, and 6 with multiple detected faces. There were no image decode failures or zero-face results. The multiple-face images are flagged for manual review rather than included as single-face training examples.

Run `python scripts/create_face_shape_label_review.py --per-class 50` to create a blind review packet with 250 random, single-face training images (50 per source class). The packet contains independently fillable sheets for two annotators, a separate source-label answer key, and an adjudication sheet. Exact duplicate image files are excluded. It samples unique image files, not unique people; enter a pseudonymous `subject_id` only when source or participant records establish it. Do not infer identity from a face. If no reliable subject IDs can be recovered, use newly collected data with subject IDs for subject-disjoint training and validation.

The uploaded photos and generated feature CSV are ignored by Git. Do not add the raw photos or a private evaluation manifest to a public commit.

## Recommended next step

1. Use the blind 250-image packet to measure label agreement and outline visibility; compare adjudicated labels with the answer key only afterward.
2. Review the Oval errors and refine the geometric features or labeling rubric.
3. Recover subject IDs from source records where possible; otherwise collect new consented images with pseudonymous IDs and broader user coverage.
4. Add a separately sourced and reviewed `diamond` set before attempting a real-data six-class training run.
5. Build a private, subject-disjoint validation/test set with adjudicated labels for final model assessment.
