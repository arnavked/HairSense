# Six-Class Real-Photo Collection and Evaluation Plan

## Current state

The uploaded Kaggle archive has five labels (`heart`, `oblong`, `oval`, `round`, `square`) and no `diamond` examples. Its image-level train/test folders have no documented subject IDs or annotation records. The old test folder has already been used for diagnostics, so it is not a final test set.

One public candidate, [Face Shape (Real) by Face Detection BeautySense](https://universe.roboflow.com/face-detection-beautysense/face-shape-real-lagcm), lists 800 images, eight labels including Diamond, and a CC BY 4.0 license. Its public page does not document subject IDs or the annotation process. Treat it as a **candidate for audited training data only**: before use, confirm the downloaded version's label counts, image provenance, subject overlap, class mapping, and license terms, then independently review every proposed Diamond image. Do not use it as the final evaluation set.

No new photos have been imported by this plan. A six-class real-photo model and its accuracy remain unavailable until reviewed Diamond examples and a clean, subject-disjoint evaluation cohort exist.

## Collection requirements

1. Use the six canonical labels: `oval`, `round`, `square`, `oblong`, `heart`, `diamond`. Do not use `rectangle` as a separate class.
2. Record one pseudonymous `subject_id` per person from participant or source records. Never infer identity from a face. Keep the identity key, if any, outside the repository.
3. Obtain and record permission for every image. For participant photos, record informed consent before collection. For externally sourced images, keep the source URL and license record and review whether the terms cover this use.
4. Collect varied, clear, front-facing images with the hairline, jaw, and chin visible. Exclude images with multiple faces, strong occlusion, extreme pose, heavy perspective distortion, or insufficient resolution; record an exclusion reason rather than guessing a label.
5. Ask two reviewers to label each image independently. Randomize opaque image IDs and hide source labels, subject identity, split, and model predictions. Record their individual labels in `face_shape_annotation_log.template.csv`.
6. Adjudicate disagreements using the written face-shape rubric. Keep the original votes and adjudication record; do not overwrite a source label to force agreement. Mark ambiguous or unreviewable images as excluded.
7. Assign subjects—not images—to train, validation, or test. Keep every image and near-duplicate of a subject in that same split. The final test cohort must not be used to choose features, thresholds, labels, or model settings.
8. Prefer a newly consented cohort for the final test split. Do not claim disjointness from the uploaded celebrity archive unless source records establish identities and cross-source overlap has been checked.

### Reviewer guide

Reviewers should judge the visible facial outline, not hairstyle, makeup, pose, or a source-provided label. Use these cues consistently and mark low-visibility or genuinely ambiguous cases for exclusion/adjudication:

- **Oval:** face is longer than wide; cheek area is widest; forehead and jaw taper gently, with a rounded chin.
- **Round:** length and width are close; cheeks are full; jaw and chin have a soft curved outline.
- **Square:** length and width are close; forehead, cheek, and jaw widths are similar; jaw corners look broad or angular.
- **Oblong:** face is notably longer than wide; forehead, cheeks, and jaw stay comparatively similar in width; jaw can be soft or angular. Use this project label for the long/rectangular category.
- **Heart:** forehead or temples are widest; cheeks remain broad; the jaw narrows to a pointed chin.
- **Diamond:** cheekbones are widest; both forehead and jaw are narrower; chin is often tapered. To distinguish it from heart, look for the narrower forehead rather than a broad forehead/temples.

These are annotation cues, not objective medical categories. In the vote log, use `outline_visibility` values `clear`, `partial`, or `poor`; use confidence from 1 (low) to 5 (high). A reviewer should not force a label for a poor or ambiguous outline.

## Staged size targets

These are collection targets for a practical pilot, not guarantees of statistical precision:

| Split | Target distinct subjects per class | Six-class total |
|---|---:|---:|
| Train | 50 | 300 |
| Validation | 20 | 120 |
| Test | 30 minimum; 50 preferred | 180 minimum; 300 preferred |

Aim for one image per person in the first pass. If multiple images are kept, keep them together by subject and average predictions at the subject level as the evaluation script does. Report per-class support and uncertainty; do not present a small pilot score as a validated product accuracy.

## Files and next actions

- `face_shape_eval_manifest.template.csv` records the final adjudicated label, subject split, independent reviewer IDs, consent, and review status. Copy it to the ignored `data/face_shape_eval_manifest.csv` and keep the populated file private.
- `face_shape_annotation_log.template.csv` is a separate blind-vote log. Store reviewer IDs as pseudonyms; in the evaluation manifest, put the distinct IDs separated by semicolons and set `annotator_count` to the matching count. Keep the completed log private; do not commit face images, identity keys, or populated manifests.
- First collect and review Diamond examples, then fill all six classes in train and validation. Reserve new test subjects before model iteration. Only then train the six-class real-data model and run `scripts/evaluate_face_shape_classifier.py` once the final manifest is complete.

