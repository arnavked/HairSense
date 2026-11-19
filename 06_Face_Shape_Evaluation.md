# Face-Shape Evaluation Protocol

## Current status

The current classifier is trained on synthetic feature distributions. Its synthetic holdout score is a diagnostic of that generated data only; it is not an estimate of accuracy on real photos. No adjudicated real-photo labels are currently included in this repository.

## Building the evaluation set

1. Use images collected with permission for this purpose. Keep raw photos in an access-controlled location and do not commit them unless their owners have explicitly agreed to public distribution.
2. Use the rubric in 07_Face_Shape_Labeling_Guide.md. Have at least two annotators label each image independently, then adjudicate disagreements before writing the final face_shape.
3. Assign subjects, not images, to train, validation, or test splits. A subject must appear in only one split, including when multiple photos of that person are used.
4. Keep the test split frozen while changing features, training data, or model settings. Use it once for final reporting after those choices are fixed.
5. Fill data/face_shape_eval_manifest.template.csv into a private data/face_shape_eval_manifest.csv. Paths are relative to the repository root. The manifest records final labels and consent status, not annotator names.

The evaluation script validates consent, adjudication, at least two annotators, unique image paths, and subject-level split isolation. It averages probabilities across each test subject's images so subjects with more photos do not receive extra weight. Every test image must decode and contain exactly one detected face.

## Running the report

First create the ONNX classifier with scripts/train_face_shape_classifier.py, then run:

    python scripts/evaluate_face_shape_classifier.py --manifest data/face_shape_eval_manifest.csv --output reports/face_shape_real_photo_metrics.json

The report includes subject-level accuracy, balanced accuracy, macro F1, per-class precision/recall/F1, support, and a confusion matrix. Publish the test subject count with the metrics. Do not treat the MLP's returned probabilities as calibrated confidence until calibration is checked on a separate real-photo validation split.

## Feature geometry

MediaPipe x/y coordinates are normalized against image width and height separately. Geometry must convert them back to pixel coordinates before calculating Euclidean distances or angles. FaceResult therefore carries the source image dimensions; callers passing raw landmark sequences must provide those dimensions explicitly.
