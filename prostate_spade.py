"""SPADE-GAN for anatomy_mask → T2 on prostate158.

Reuses the cv_spade generator + helpers but with mask_nc = 3 (background +
peripheral zone + central gland). Critical hypothesis: prostate masks cover
~30% of slice area, so SPADE should have far more conditioning signal per
pixel than on TopBrain (where the 41 vessel labels cover ~1-2%) — see
[[topbrain-spade-sparsity]] for why TopBrain SPADE underperformed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import cv_spade as cv_sp
from pasd_spade import (   # noqa: F401
    SPADEResBlock,
    HingeGANLoss,
    FeatureMatchingLoss,
    VGGPerceptualLoss,
    FeaturePatchDiscriminator,
)
from pasd_pix2pix import init_weights   # noqa: F401


# Prostate158 t2_anatomy_reader1 mask uses 3 label values:
#   0 = background, 1 = peripheral zone, 2 = central gland.
PROSTATE_LABEL_NC = 3


class ProstateSPADEGenerator(cv_sp.CvSPADEGenerator):
    """Same architecture as CvSPADEGenerator but defaults to PROSTATE_LABEL_NC."""

    def __init__(self, mask_nc: int = PROSTATE_LABEL_NC, image_ch: int = 1,
                 ngf: int = 32, z_dim: int = 256):
        super().__init__(mask_nc=mask_nc, image_ch=image_ch, ngf=ngf, z_dim=z_dim)


def one_hot_mask(mask_int: torch.Tensor,
                 n_classes: int = PROSTATE_LABEL_NC) -> torch.Tensor:
    return cv_sp.one_hot_mask(mask_int, n_classes=n_classes)
