"""
Named subsets of the MediaPipe Face Mesh landmark topology (468 base points,
478 with refine_landmarks=True for iris).

These indices are fixed by the MediaPipe face mesh model itself (not
something we choose) — documented at:
https://developers.google.com/mediapipe/solutions/vision/face_landmarker

Downstream consumers:
- Bucket 3 (quality checks): EYE_LEFT_OUTER / EYE_RIGHT_OUTER / NOSE_TIP for
  yaw angle estimation.
- Bucket 2/4 (face-shape classification + feature extraction): FACE_OVAL,
  JAWLINE, CHIN, FOREHEAD_CENTER, CHEEKBONE_LEFT/RIGHT for geometric ratios
  (jaw width vs cheekbone width vs forehead width vs face length, etc).

NOTE: verify these visually against real photos (see debug_draw.py) before
relying on them for classifier feature engineering in Bucket 2 — indices
are correct per the MediaPipe spec, but "which index is *the* cheekbone
point" is a modeling choice worth eyeballing once we have real data.
"""

# Full face silhouette / oval, ordered roughly clockwise from top.
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
)

# Lower-jaw subset of the oval, chin to chin, useful for jaw-width ratio.
JAWLINE = (152, 148, 176, 149, 150, 136, 172, 58, 132, 377, 400, 378, 379, 365, 397)

CHIN = 152
FOREHEAD_CENTER = 10
NOSE_TIP = 1

# Outer eye corners — good stable landmarks for yaw/roll estimation.
EYE_RIGHT_OUTER = 33
EYE_LEFT_OUTER = 263

# Approximate cheekbone widest points (zygomatic region).
CHEEKBONE_LEFT = 234
CHEEKBONE_RIGHT = 454