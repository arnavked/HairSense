#!/usr/bin/env bash
# Fetches the MediaPipe FaceLandmarker model used by vision/face_landmarks.py.
# Requires network access — run this once, locally or in CI, wherever
# network isn't restricted (this repo's dev sandbox has none).
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH="${MODEL_DIR}/face_landmarker.task"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ]; then
  echo "Model already present at ${MODEL_PATH}"
  exit 0
fi

echo "Downloading FaceLandmarker model to ${MODEL_PATH} ..."
curl -L -o "${MODEL_PATH}" "${MODEL_URL}"
echo "Done."
