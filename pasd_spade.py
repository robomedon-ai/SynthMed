"""SPADE-GAN for mask-conditioned PASD MRI synthesis.

Reference: Park et al., "Semantic Image Synthesis with Spatially-Adaptive
Normalization", CVPR 2019.

Why SPADE for this dataset:
- Channel-concat (pix2pix / our LDM) injects mask info only at the input;
  it gets diluted through the network. SPADE re-injects the mask at EVERY
  normalization layer, giving much tighter mask adherence.
- The PDF explicitly cites SPADE as the SOTA recipe for mask-controlled
  medical synthesis when data is scarce — exactly our PASD situation.

Architecture (matches the canonical SPADE setup, scaled down for our data):
  Generator: random z (256-dim) -> MLP -> reshape (16*ngf, 4, 4) -> 6 SPADE
             ResBlocks with bilinear upsampling -> Tanh => (3, 256, 256).
  Discriminator: PatchGAN reused from pasd_pix2pix.py.

Inputs in [-1, 1]: mask is (B, 1, 256, 256), image is (B, 3, 256, 256).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the existing PatchGAN discriminator
from pasd_pix2pix import PatchDiscriminator, init_weights


# ── SPADE normalization ──

class SPADE(nn.Module):
    """Spatially-adaptive normalization.

    BN-normalize x, then modulate per-pixel by (1+γ, β) where γ,β are
    convolved from the (resized) segmentation mask.
    """

    def __init__(self, norm_nc: int, label_nc: int = 1,
                 hidden_nc: int = 128, kernel: int = 3):
        super().__init__()
        # Use a *parameter-free* normalization so all "style" comes from mask.
        self.norm = nn.BatchNorm2d(norm_nc, affine=False)
        pad = kernel // 2
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel, padding=pad),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden_nc, norm_nc, kernel, padding=pad)
        self.beta  = nn.Conv2d(hidden_nc, norm_nc, kernel, padding=pad)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = self.norm(x)
        # Resize mask to x's spatial size; keep continuous values so grads flow
        m = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
        h = self.shared(m)
        g = self.gamma(h)
        b = self.beta(h)
        return n * (1 + g) + b


class SPADEResBlock(nn.Module):
    """Residual block with SPADE normalization + ReLU + spectral-norm convs."""

    def __init__(self, fin: int, fout: int, label_nc: int = 1):
        super().__init__()
        fmid = min(fin, fout)
        self.learned_shortcut = (fin != fout)
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmid, label_nc)
        self.conv_0 = nn.utils.spectral_norm(nn.Conv2d(fin,  fmid, 3, padding=1))
        self.conv_1 = nn.utils.spectral_norm(nn.Conv2d(fmid, fout, 3, padding=1))
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)
            self.conv_s = nn.utils.spectral_norm(
                nn.Conv2d(fin, fout, 1, bias=False))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv_0(F.leaky_relu(self.norm_0(x, mask), 0.2))
        h = self.conv_1(F.leaky_relu(self.norm_1(h, mask), 0.2))
        if self.learned_shortcut:
            s = self.conv_s(self.norm_s(x, mask))
        else:
            s = x
        return s + h


# ── Generator: random z + mask -> image ──

class SPADEGenerator(nn.Module):
    """SPADE generator for 256x256 outputs.

    Random style z (256-dim) is up-projected to (16*ngf, 4, 4) then passed
    through 6 SPADE ResBlocks with bilinear upsample-x2 between them.
    """

    def __init__(self, mask_ch: int = 1, image_ch: int = 3,
                 ngf: int = 32, z_dim: int = 256):
        super().__init__()
        self.z_dim = z_dim
        # 4x4 start, 6 upsamples -> 256x256
        self.start_h = self.start_w = 4
        self.fc = nn.Linear(z_dim, 16 * ngf * self.start_h * self.start_w)

        # SPADE residual stack — channel halves at the larger spatial sizes
        self.head_0 = SPADEResBlock(16 * ngf, 16 * ngf, mask_ch)  # 4 -> upsample 8
        self.G_mid_0 = SPADEResBlock(16 * ngf, 16 * ngf, mask_ch) # 8 -> upsample 16
        self.G_mid_1 = SPADEResBlock(16 * ngf,  8 * ngf, mask_ch) # 16 -> upsample 32
        self.up_0 = SPADEResBlock( 8 * ngf,  4 * ngf, mask_ch)    # 32 -> upsample 64
        self.up_1 = SPADEResBlock( 4 * ngf,  2 * ngf, mask_ch)    # 64 -> upsample 128
        self.up_2 = SPADEResBlock( 2 * ngf,      ngf, mask_ch)    # 128 -> upsample 256

        self.conv_img = nn.Conv2d(ngf, image_ch, 3, padding=1)

    def _up(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear",
                             align_corners=False)

    def forward(self, mask: torch.Tensor,
                z: torch.Tensor | None = None) -> torch.Tensor:
        B = mask.size(0)
        if z is None:
            z = torch.randn(B, self.z_dim, device=mask.device,
                            dtype=mask.dtype)
        x = self.fc(z).view(B, -1, self.start_h, self.start_w)

        x = self.head_0(x, mask); x = self._up(x)   # 4 -> 8
        x = self.G_mid_0(x, mask); x = self._up(x)  # 8 -> 16
        x = self.G_mid_1(x, mask); x = self._up(x)  # 16 -> 32
        x = self.up_0(x, mask); x = self._up(x)     # 32 -> 64
        x = self.up_1(x, mask); x = self._up(x)     # 64 -> 128
        x = self.up_2(x, mask); x = self._up(x)     # 128 -> 256

        x = F.leaky_relu(x, 0.2)
        x = self.conv_img(x)
        return torch.tanh(x)


# ── Losses ──

class HingeGANLoss(nn.Module):
    """Hinge GAN loss — standard for SPADE/SAGAN-style training. More stable
    than BCE for spectrally-normalized GANs."""

    def __call__(self, prediction: torch.Tensor, is_real: bool,
                 for_d: bool = True) -> torch.Tensor:
        if for_d:
            if is_real:
                return torch.relu(1.0 - prediction).mean()
            return torch.relu(1.0 + prediction).mean()
        # Generator step: just maximize prediction
        return (-prediction).mean()


class FeatureMatchingLoss(nn.Module):
    """L1 over discriminator intermediate features (Wang et al. pix2pixHD)."""

    def __init__(self, weight: float = 10.0):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def __call__(self, fake_feats: list, real_feats: list) -> torch.Tensor:
        n = min(len(fake_feats), len(real_feats))
        if n == 0:
            return torch.zeros((), device=fake_feats[0].device if fake_feats else "cpu")
        total = 0.0
        for ff, rf in zip(fake_feats[:n], real_feats[:n]):
            total = total + self.l1(ff, rf.detach())
        return self.weight * (total / n)


class VGGPerceptualLoss(nn.Module):
    """Standard VGG19 perceptual loss (Johnson et al.)."""

    def __init__(self, weight: float = 10.0):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        # Slices at common cut-points: relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        slices, layer_idx = [], [1, 6, 11, 20, 29]
        prev = 0
        for idx in layer_idx:
            slices.append(nn.Sequential(*list(vgg[prev:idx + 1])))
            prev = idx + 1
        self.slices = nn.ModuleList(slices)
        for p in self.parameters():
            p.requires_grad = False
        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weight = weight
        self.l1 = nn.L1Loss()
        # Layer weights (deeper = lower weight, common practice)
        self.layer_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1, 1] -> [0, 1] then ImageNet-normalize
        return ((x + 1) / 2 - self.mean) / self.std

    def __call__(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        f, r = self._norm(fake), self._norm(real)
        total = 0.0
        for sl, w in zip(self.slices, self.layer_weights):
            f = sl(f); r = sl(r)
            total = total + w * self.l1(f, r.detach())
        return self.weight * total


# ── Discriminator wrapper that also exposes intermediate features ──

class FeaturePatchDiscriminator(nn.Module):
    """PatchGAN that returns per-layer features for the FM loss."""

    def __init__(self, in_ch: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        D = PatchDiscriminator(in_ch=in_ch, ndf=ndf, n_layers=n_layers)
        # Re-expose D.model as a sequence of layer blocks for hooks
        self.layers = nn.ModuleList(list(D.model))

    def forward(self, x: torch.Tensor):
        feats = []
        for layer in self.layers:
            x = layer(x)
            # Only keep activations after each "block" (conv+norm+relu), i.e.
            # immediately after a LeakyReLU — same as pix2pixHD's FM scheme.
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        return x, feats     # x is the final logit map
