"""Stage 1 of the CV LDM: SD-VAE decoder fine-tune on TopBrain slices.

Architecturally identical to pasd_vae.py — reuses load_vae(), vae_forward(),
VAEReconLoss(). The only thing that differs is the training data, handled
in train_cv_vae.py.

Why a separate file:
  - Keeps the CV pipeline grep-able as its own module set.
  - Lets us later swap in CV-specific VAE adaptations without touching PASD.
"""

# Re-export the PASD VAE helpers so the rest of the CV pipeline imports cv_vae
# rather than pasd_vae. If we ever need to diverge (e.g. larger latent for
# higher-res CTA), edit this file only.
from pasd_vae import (   # noqa: F401
    VAE_HF_ID,
    load_vae,
    trainable_parameters,
    vae_forward,
    VAEReconLoss,
)
