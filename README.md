---
title: SynthMed
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SynthMed — Medical Image Synthesis & Downstream-Utility Evaluation

A unified platform for medical image synthesis and **downstream-utility** evaluation
across three paired datasets (placental MRI · cerebrovascular CT/MRA · prostate mpMRI).
It trains five generator families (pix2pix, SPADE, CycleGAN, latent diffusion, and a
mask-conditioned latent diffusion) under one pipeline and evaluates them on both image
quality and Train-on-Synthetic-Test-on-Real (TSTR) segmentation utility.

**Headline finding:** *realism ≠ relevance* — the generator with the best FID is not
the most useful for training real downstream models.

## How this Space boots

- **Code** lives in this repo; **checkpoints do not** (they total ~6 GB).
- On startup, `deploy/download_checkpoints.py` pulls the production checkpoints from
  the HF Model Hub repo named in the `HF_MODEL_REPO` Space variable, restoring them
  into the app tree, then `gunicorn` serves `app:app` on port 7860.

### Required Space configuration

| Variable / secret | Purpose |
|---|---|
| `HF_MODEL_REPO` (variable) | model repo holding the checkpoints, e.g. `user/synthmed-checkpoints` |
| `HF_TOKEN` (secret) | only if that checkpoint repo is **private** |

### Hardware

GPU is strongly recommended — the diffusion models are slow on CPU (the GAN
families remain usable). Pick a GPU tier in **Settings → Hardware**.

See `deploy/DEPLOY.md` for the full, step-by-step deployment guide.

> ⚠️ Research prototype. Synthetic images are for research only — not for diagnostic use.
# SynthMed
