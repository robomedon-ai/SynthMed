"""Stage 2 of the prostate LDM: T2-conditional latent diffusion for T2 → ADC.

Thin re-export of cv_ldm helpers. Same channel-concat conditioning design
(4 noisy ADC latent + 4 T2 latent = 8 in_channels), v-prediction,
classifier-free guidance.
"""

from cv_ldm import (   # noqa: F401
    LATENT_CHANNELS,
    LATENT_SIZE,
    COND_CHANNELS,
    IN_CHANNELS,
    VAE_SCALING_FACTOR,
    build_unet,
    build_train_scheduler,
    build_inference_scheduler,
    training_step,
    sample,
    cfg_dropout,
    EMAModel,
)
