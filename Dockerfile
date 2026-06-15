# SynthMed — Hugging Face Spaces (Docker SDK).
# torch + torchvision + CUDA come prebuilt in the base image.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    PORT=7860

# System libs for medical-image I/O (SimpleITK / nibabel / skimage).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching.
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r deploy/requirements.txt

# App code only — heavyweight model/data dirs are excluded by .dockerignore and
# are fetched at startup from the HF Hub (see deploy/download_checkpoints.py).
COPY . /app

# HF Spaces run as a non-root user (uid 1000); make the tree + HF cache writable
# so startup can download checkpoints into the model dirs.
RUN chmod +x deploy/start.sh && \
    mkdir -p /tmp/hf && \
    chmod -R 777 /app /tmp/hf

EXPOSE 7860
CMD ["bash", "deploy/start.sh"]
