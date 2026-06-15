"""SPADE-GAN for vessel-mask → CTA on the TopBrain dataset.

This is the second cerebrovascular model: it uses the 40+ multi-class vessel
labels (Circle of Willis named segments etc.) — the unique feature of
TopBrain that nothing else in this codebase exploits.

Reuses the SPADE blocks from pasd_spade.py. Differences vs the PASD-SPADE:
  - label_nc = 41 (CT labels 0-40, one-hot encoded)
  - image_ch = 1 (grayscale CT slice)
  - No L1 loss (lesson from PASD: L1 with SPADE hurts mask faithfulness)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pasd_spade import (
    SPADEResBlock,
    HingeGANLoss,
    FeatureMatchingLoss,
    VGGPerceptualLoss,
    FeaturePatchDiscriminator,
)
from pasd_pix2pix import init_weights  # noqa: F401  re-export


# TopBrain CT mask uses label values 0-40 (background + 40 named segments).
CV_LABEL_NC = 41


class CvSPADEGenerator(nn.Module):
    """SPADE generator for 256×256 single-channel angiography outputs.

    Random style z (256-dim) → MLP → reshape (16*ngf, 4, 4) → 6 SPADE ResBlocks
    with bilinear upsample-×2 → grayscale image.
    """

    def __init__(self, mask_nc: int = CV_LABEL_NC, image_ch: int = 1,
                 ngf: int = 32, z_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        self.start_h = self.start_w = 4
        self.fc = nn.Linear(z_dim, 16 * ngf * self.start_h * self.start_w)

        # Six SPADE residual blocks with upsampling between them
        self.head_0  = SPADEResBlock(16 * ngf, 16 * ngf, mask_nc)
        self.G_mid_0 = SPADEResBlock(16 * ngf, 16 * ngf, mask_nc)
        self.G_mid_1 = SPADEResBlock(16 * ngf,  8 * ngf, mask_nc)
        self.up_0    = SPADEResBlock( 8 * ngf,  4 * ngf, mask_nc)
        self.up_1    = SPADEResBlock( 4 * ngf,  2 * ngf, mask_nc)
        self.up_2    = SPADEResBlock( 2 * ngf,      ngf, mask_nc)

        self.conv_img = nn.Conv2d(ngf, image_ch, 3, padding=1)

    def _up(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear",
                             align_corners=False)

    def forward(self, mask_onehot: torch.Tensor,
                z: torch.Tensor | None = None) -> torch.Tensor:
        """mask_onehot: (B, CV_LABEL_NC, 256, 256)  →  out: (B, 1, 256, 256)."""
        B = mask_onehot.size(0)
        if z is None:
            z = torch.randn(B, self.z_dim, device=mask_onehot.device,
                            dtype=mask_onehot.dtype)
        x = self.fc(z).view(B, -1, self.start_h, self.start_w)
        x = self.head_0(x, mask_onehot);  x = self._up(x)
        x = self.G_mid_0(x, mask_onehot); x = self._up(x)
        x = self.G_mid_1(x, mask_onehot); x = self._up(x)
        x = self.up_0(x, mask_onehot);    x = self._up(x)
        x = self.up_1(x, mask_onehot);    x = self._up(x)
        x = self.up_2(x, mask_onehot);    x = self._up(x)
        x = F.leaky_relu(x, 0.2)
        return torch.tanh(self.conv_img(x))


def one_hot_mask(mask_int: torch.Tensor, n_classes: int = CV_LABEL_NC
                 ) -> torch.Tensor:
    """(B, H, W) long → (B, n_classes, H, W) float32 one-hot."""
    mask_int = mask_int.clamp(0, n_classes - 1).long()
    onehot = F.one_hot(mask_int, num_classes=n_classes)
    return onehot.permute(0, 3, 1, 2).float()
