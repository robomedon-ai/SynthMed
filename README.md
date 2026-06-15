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

A unified web platform for **medical image synthesis** and **downstream-utility
evaluation** across three paired datasets (placental MRI · cerebrovascular CT/MRA ·
prostate mpMRI). It trains five generator families — pix2pix, SPADE, CycleGAN, latent
diffusion (LDM), and a mask-conditioned *combined* LDM — under one common pipeline and
evaluates them on both image quality and **Train-on-Synthetic-Test-on-Real (TSTR)**
segmentation utility.

This repository is the research artifact accompanying the paper
**"Realism Is Not Relevance: A Cross-Dataset Study of Generative Models and Downstream
Utility for Medical Image Synthesis."**

> **Headline finding — *realism ≠ relevance*.** The generator with the best image-quality
> score (FID) is **not** the most useful for training real downstream models. A
> mask-conditioned latent diffusion model with a *worse* FID consistently produces the
> most useful synthetic training data, nearly matching real data.

![SynthMed home dashboard](figs/image1.png)

---

## Motivation

Progress in medical machine learning is constrained by data: annotated medical images
are scarce, expensive, and limited by privacy and inter-site variability. Generative
models are increasingly proposed to fill this gap, but the choice among them is almost
always made using **image-quality metrics** (FID, SSIM, PSNR, LPIPS) on a single dataset.
This rests on an unexamined assumption — that more realistic-looking images are also more
useful for a downstream task.

SynthMed tests that assumption empirically. With model implementations, training schedule,
and evaluation protocol held **fixed**, only the dataset varies. Each generator is ranked
twice — by image quality and by downstream utility — and the two rankings are compared.
Because every generator is trained and evaluated identically, any disagreement is
attributable to the model and the data, not to implementation differences.

---

## The Platform

SynthMed is a single web application providing three capabilities, plus a results
dashboard. The unifying design principle is **uniformity**: every dataset is reduced to a
common 256×256 axial-slice representation, and every generator implements one of two
interfaces (image→image or mask→image). The same U-Net/PatchGAN, SPADE, SD-VAE, and
diffusion U-Net code is reused across datasets; only thin data adapters differ.

### 1. Visualization — multi-planar 2D viewers + WebGL volume renderer

Per-dataset 2D slice browsing with mask overlays, windowing, and a 3D WebGL volume
renderer.

| Placental MRI (PASD) | Cerebrovascular CT/MRA (TopBrain) |
|---|---|
| ![PASD viewer](figs/image2.png) | ![TopBrain CTA viewer](figs/image3.png) |

![TopBrain TOF-MRA viewer with 3D Circle-of-Willis rendering](figs/image4.png)

### 2. Synthesis — per-dataset, per-model inference behind a common API

A processing-modules hub exposes every trained generator with interactive controls
(sampler steps, classifier-free guidance, ensemble averaging, seed).

![Processing modules hub](figs/image5.png)

Representative synthesis modules:

| Prostate T2→ADC translation | Prostate anatomy-mask→T2 (SPADE) |
|---|---|
| ![Prostate T2 to ADC](figs/image9.png) | ![Prostate anatomy mask to T2](figs/image10.png) |

| TopBrain MRA→CTA (pix2pix vs LDM) | TopBrain vessel-mask→CTA (SPADE) |
|---|---|
| ![TopBrain MRA to CTA](figs/image11.png) | ![TopBrain vessel mask to CTA](figs/image12.png) |

| PASD mask→MRI generation | PASD cross-modality (CycleGAN, BTFE↔TSE) |
|---|---|
| ![PASD mask to MRI](figs/image13.png) | ![PASD cross-modality translation](figs/image14.png) |

### 3. Evaluation — downstream-utility (TSTR) and image-quality protocols

For each generator a synthetic training set (matched in size to the real one) trains a
U-Net segmenter from scratch; Dice is reported on the **real** held-out test set. A
segmenter trained on real data is the upper-bound reference. The viewer overlays
predictions from each synthetic-trained segmenter (red) against the real-trained one
(green).

| Prostate gland (TSTR) | TopBrain vessels (TSTR) | PASD placenta (TSTR) |
|---|---|---|
| ![Prostate TSTR](figs/image6.png) | ![TopBrain TSTR](figs/image7.png) | ![PASD TSTR](figs/image8.png) |

### 4. Results dashboard

Image-quality metrics (FID, KID, SSIM, PSNR, LPIPS) and TSTR Dice tables alongside
qualitative sample grids.

![Results dashboard](figs/image15.png)

---

## Datasets

Three paired datasets that differ in modality relationship and mask density.

| Dataset | Task | Modalities | Mask coverage |
|---|---|---|---|
| **PASD** (placental MRI) | mask→MRI | BTFE / TSE | ~2% |
| **TopBrain** (cerebrovascular) | MRA→CTA | CT / MRA | ~1–2% |
| **Prostate158** | T2→ADC | T2 / ADC / DWI | ~30% |

They span a same-imaging-type pair (PASD: two T2-weighted MRI sequences), two
physically-distinct pairs (TopBrain CT↔MRA; prostate T2↔ADC), and a wide range of mask
coverage (~2% to ~30%).

---

## Generative Model Families

All five share one implementation reused across datasets:

- **pix2pix** — canonical conditional GAN for paired translation; U-Net generator +
  70×70 PatchGAN, with an L1 + adversarial objective.
- **CycleGAN** — unpaired translation via cycle consistency; two ResNet generators + two
  PatchGAN discriminators with an image-replay buffer.
- **SPADE-GAN** — mask→image synthesis; the segmentation mask is re-injected at every
  normalization layer for tight mask adherence.
- **Latent diffusion (LDM)** — diffusion in a frozen Stable-Diffusion VAE latent,
  conditioned on the source-image latent; *v*-prediction + classifier-free guidance,
  DPM-Solver++ sampling.
- **Combined LDM** — the LDM additionally conditioned on the one-hot segmentation mask at
  latent resolution. **The headline model of the study.**

---

## Key Results

### Image-quality rankings do not predict downstream utility

Downstream utility (TSTR Dice on **real** test data, mean ± std over three seeds) on the
two image→image datasets. *"Real"* is the upper-bound reference; **best synthetic in bold.**

| Trained on | Prostate (gland) | TopBrain (vessel) |
|---|---|---|
| Real data | 0.723 ± .003 | 0.671 ± .006 |
| **Combined LDM** | **0.731 ± .004** | **0.626 ± .008** |
| LDM *(best FID)* | 0.714 ± .008 | 0.608 ± .010 |
| Pix2pix | 0.710 ± .007 | 0.499 ± .013 |
| CycleGAN | 0.599 ± .019 | 0.373 ± .011 |

The utility ranking **Combined > LDM > pix2pix > CycleGAN** is identical across both
datasets — yet it is **not** the FID ranking. The FID winner is plain LDM, whose FID is
*better* than the combined LDM's (TopBrain 95.7 vs 105.1; prostate 72.7 vs 75.6), but it
loses on downstream Dice. The combined model matches or slightly exceeds real data on
prostate (0.731 vs 0.723) and reaches ~93% of real-data utility on TopBrain. With
three-seed error bars, every adjacent gap exceeds the per-condition standard deviation, so
the effect is not seed noise.

### Other findings

- **Synthetic data approaches real-data utility.** On PASD (mask→MRI), a segmenter trained
  *purely* on LDM-synthesized MRI reaches Dice **0.856 ± .002** vs **0.900 ± .002** for
  real data; LDM ≈ pix2pix (0.845) ≫ sparse-mask SPADE (0.633).
- **Cross-modality CycleGAN helps only for same-type modalities.** Worst generator on the
  physically-distinct pairs (TopBrain, prostate), but the best augmenter on PASD
  (BTFE↔TSE). In a low-data sweep, cross-modality pairs win at every training-set size —
  **+7.3 Dice points at N=50** (0.879 vs 0.806), roughly a 10× data-efficiency multiplier.
- **Mask conditioning helps only when masks are dense.** SPADE collapses on ~1–2% vessel
  masks (TopBrain) and yields the worst utility on ~2% placenta masks (PASD), but the
  combined LDM is the best generator overall on ~30% prostate zonal masks.
- **Augmentation (real + synthetic) never materially harms Dice** and helps most where real
  data is scarce; it even rehabilitates CycleGAN, the worst synthetic-only source, into the
  best prostate augmenter (0.599 → 0.737).
- **The ranking is robust to the segmenter.** Re-running TSTR with an 8.3M-parameter
  residual U-Net preserves the ranking on every dataset.

**Takeaway:** measure *relevance*, not just *realism*. Use TSTR-style utility evaluation as
a standard complement to FID/SSIM when the goal is augmentation or modality substitution.

---

## Repository Structure

The code is organized by dataset prefix (`cv_` = cerebrovascular/TopBrain, `pasd_` =
placental, `prostate_` = prostate) and stage:

- `*_data.py` — dataset adapters and preprocessing
- `train_*_<model>.py` — training entry points per dataset/generator
- `*_<model>.py` — model definitions (pix2pix, spade, cyclegan, ldm, vae, …)
- `eval_*.py`, `*_eval/` — image-quality and qualitative evaluation
- `tstr*.py`, `aggregate_tstr.py` — downstream-utility (TSTR) protocol
- `app.py`, `templates/`, `static/` — the Flask web application
- `deploy/` — checkpoint download/upload and deployment helpers

> Datasets and trained checkpoints are **not** included in this repository (the
> checkpoints total ~6 GB). Training scripts and the download helper let you reproduce or
> restore them.

---

## How this Space boots

- **Code** lives in this repo; **checkpoints do not** (they total ~6 GB).
- On startup, `deploy/download_checkpoints.py` pulls the production checkpoints from the
  HF Model Hub repo named in the `HF_MODEL_REPO` Space variable, restoring them into the
  app tree, then `gunicorn` serves `app:app` on port 7860.

### Required Space configuration

| Variable / secret | Purpose |
|---|---|
| `HF_MODEL_REPO` (variable) | model repo holding the checkpoints, e.g. `user/synthmed-checkpoints` |
| `HF_TOKEN` (secret) | only if that checkpoint repo is **private** |

### Hardware

GPU is strongly recommended — the diffusion models are slow on CPU (the GAN families
remain usable). Pick a GPU tier in **Settings → Hardware**.

---

## Acknowledgments

This research has been partially supported by the Croatian Science Foundation under
project numbers UIP-2025-02-2755 and IP-2024-05-9492.

> ⚠️ **Research prototype.** Synthetic images are for research only — **not for diagnostic
> use.**
