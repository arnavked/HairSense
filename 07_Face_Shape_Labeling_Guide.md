# Face-Shape Labeling Guide

Use this rubric to label visible facial geometry in a clear, front-facing image. It is an operational guide for consistent project annotations, not a medical classification standard. Do not infer a label from hair, hairstyle, gender, or the classifier's prediction.

## Before labeling

- Confirm the face is unobstructed enough to compare the forehead, cheekbones, jaw, and chin.
- Ignore hair volume and facial hair. Do not guess the hairline when hair covers the forehead.
- Label from the overall proportions and outline, not one measurement alone.
- Record a label only when the annotator can choose one class with reasonable confidence. Mark unclear cases for adjudication; exclude unresolved cases from the scored set.

## Class cues

| Label | Main cues |
|---|---|
| Oval | Face is longer than it is wide; cheekbones are often the widest area; forehead and jaw narrow gently; chin is rounded. |
| Round | Face length and width are relatively similar; cheeks are full; jaw and chin have soft, rounded contours. |
| Square | Forehead, cheek, and jaw widths are relatively similar; jaw corners look broad or angular; face is not strongly elongated. |
| Oblong | Face is clearly longer than it is wide; forehead, cheek, and jaw widths are relatively similar; outline is more straight-sided than oval. |
| Heart | Forehead or temples are wider than the jaw; the lower face tapers toward a narrow or pointed chin. |
| Diamond | Cheekbones are the widest area; forehead and jaw are narrower; chin is often narrow or pointed. |

## Independent labels and adjudication

1. At least two annotators label each image independently, without seeing each other's label or the model output.
2. If labels agree, record the agreed class and set adjudicated to true.
3. If labels disagree or either annotator marks the image unclear, a third reviewer adjudicates. If no class is defensible, leave the image out of the evaluation set.
4. Keep annotator names and discussion notes outside the repository. The CSV records only the annotator count and final decision.

Face-shape boundaries overlap. Report the held-out subject count and per-class confusion matrix so this uncertainty remains visible.
