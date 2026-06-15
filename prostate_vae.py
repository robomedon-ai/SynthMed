"""Stage 1 of the prostate LDM: SD-VAE decoder fine-tune on prostate158 slices.

Thin re-export of pasd_vae helpers (same pattern as cv_vae.py). Lets us
later swap in prostate-specific VAE adaptations without touching PASD/CV code.
"""

from pasd_vae import (   # noqa: F401
    VAE_HF_ID,
    load_vae,
    trainable_parameters,
    vae_forward,
    VAEReconLoss,
)
