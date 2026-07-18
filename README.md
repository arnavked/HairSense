# models/

This directory holds the MediaPipe FaceLandmarker `.task` model file used by
`vision/face_landmarks.py`. It's gitignored (model binaries don't belong in
version control) — everyone who runs this repo needs to fetch it once:

```
./scripts/download_models.sh
```

Source: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
(float16 variant — good default speed/accuracy tradeoff; a `float32` variant
exists at the same path with `float16` swapped for `float32` if higher
precision is ever needed).

If you're in a network-restricted environment, download the file elsewhere
and copy it in manually as `face_landmarker.task`.
