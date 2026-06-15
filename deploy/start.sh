#!/usr/bin/env bash
# Container entrypoint: pull checkpoints, then serve.
set -euo pipefail
cd "$(dirname "$0")/.."

# Restore the ~6 GB production checkpoints from the HF Hub (idempotent).
python deploy/download_checkpoints.py

# One worker: models load lazily onto a single GPU; extra workers would
# duplicate VRAM. Threads handle concurrent requests; long timeout covers
# slow diffusion sampling.
exec gunicorn app:app \
    --bind "0.0.0.0:${PORT:-7860}" \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --graceful-timeout 30 \
    --access-logfile -
