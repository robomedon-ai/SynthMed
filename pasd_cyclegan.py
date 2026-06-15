"""CycleGAN for cross-modality PASD MRI translation (BTFE ↔ TSE).

Reference: Zhu et al., "Unpaired Image-to-Image Translation using
Cycle-Consistent Adversarial Networks", ICCV 2017.

Why CycleGAN for this dataset:
- PASD has BOTH BTFE and TSE for many patients — a paired-modality structure
  that mask-conditioned generators (pix2pix / SPADE / LDM) can't exploit.
- CycleGAN learns the *style* mapping between modalities without needing
  per-pixel pairing: G_AB(real_A) → fake_B, G_BA(fake_B) → reconstructed_A.
- This produces a fundamentally new kind of synthetic augmentation: given
  any real BTFE, you get a plausible TSE version (and vice versa), with
  the anatomy preserved by the cycle-consistency loss.

Architecture:
  G: 9-block ResNet generator (256x256, Johnson et al. style — the standard
     CycleGAN backbone).
  D: 70x70 PatchGAN (reused from pasd_pix2pix.py).
  Losses: LSGAN adversarial + λ_cycle * L1 cycle + λ_identity * L1 identity.

Inputs are (B, 3, 256, 256) tensors in [-1, 1] from pasd_data.get_pasd_dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pasd_pix2pix import PatchDiscriminator, init_weights


# ── ResNet block ──

class _ResBlock(nn.Module):
    """Standard CycleGAN ResBlock: reflection pad + conv + InstanceNorm + ReLU."""

    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


# ── 9-block ResNet generator (Johnson et al.; CycleGAN default for 256x256) ──

class ResnetGenerator(nn.Module):
    """In-shape (B, in_ch, 256, 256) → out-shape (B, out_ch, 256, 256) in [-1, 1]."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, ngf: int = 64,
                 n_blocks: int = 9):
        super().__init__()
        layers: list[nn.Module] = []
        # Initial conv (reflection pad to avoid edge artifacts)
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
        ]
        # Downsample x2 twice (256 → 128 → 64)
        m = ngf
        for _ in range(2):
            layers += [
                nn.Conv2d(m, m * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(m * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            m *= 2
        # n_blocks ResBlocks at the bottleneck
        for _ in range(n_blocks):
            layers += [_ResBlock(m)]
        # Upsample x2 twice (64 → 128 → 256)
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(m, m // 2, 3, stride=2, padding=1,
                                   output_padding=1, bias=False),
                nn.InstanceNorm2d(m // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            m //= 2
        # Final conv to image channels + tanh
        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(m, out_ch, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ── Loss ──

class LSGANLoss(nn.Module):
    """Least-Squares GAN loss (Mao et al.) — CycleGAN's standard choice.

    More stable than BCE for unpaired translation. Targets are scalars
    (real=1.0, fake=0.0) broadcast over the PatchGAN's output map.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def __call__(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
        return self.mse(prediction, target)


# ── Image buffer (Shrivastava et al.; standard CycleGAN trick) ──

class ImageBuffer:
    """Replay buffer of previously generated images, used to update Ds.

    Each iteration the buffer is updated with the new batch of fakes;
    with prob 0.5 the D sees a *historical* fake from the buffer instead.
    This stabilizes training by preventing the D from chasing the latest G.
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.images: list[torch.Tensor] = []

    def query(self, batch: torch.Tensor) -> torch.Tensor:
        if self.capacity == 0:
            return batch
        out = []
        for img in batch:
            img = img.unsqueeze(0)
            if len(self.images) < self.capacity:
                self.images.append(img.detach().clone())
                out.append(img)
            else:
                if torch.rand(1).item() < 0.5:
                    # Pop a random historical image, push current
                    idx = torch.randint(0, self.capacity, (1,)).item()
                    out.append(self.images[idx].clone())
                    self.images[idx] = img.detach().clone()
                else:
                    out.append(img)
        return torch.cat(out, dim=0)
