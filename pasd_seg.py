"""Small segmentation U-Net for TSTR on PASD placental masks.

Binary segmentation: input MRI slice (3 channels, [-1, 1]) -> placenta mask
(1 channel, sigmoid).

~7M params, trains in ~5-10 min per condition on the RTX 5080.

Reference: standard 4-level U-Net, BN+ReLU, transposed-conv upsampling.
Combined Dice + BCE loss, the standard recipe for binary medical segmentation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ── U-Net architecture ──

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """4-level U-Net, base=32 channels (~7M params)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block(in_ch, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.bottom = _conv_block(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottom(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)  # raw logits


# ── Second architecture: Residual U-Net (for robustness checks) ──

class _ResBlock(nn.Module):
    """Two 3x3 convs with a projected identity skip (pre-act residual)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                     if in_ch != out_ch else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.proj(x))


class ResUNet(nn.Module):
    """Residual U-Net; genuinely different from SmallUNet (residual blocks,
    bilinear upsampling instead of transposed conv). For #2 robustness check."""
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32):
        super().__init__()
        self.e1 = _ResBlock(in_ch, base)
        self.e2 = _ResBlock(base, base * 2)
        self.e3 = _ResBlock(base * 2, base * 4)
        self.e4 = _ResBlock(base * 4, base * 8)
        self.bottom = _ResBlock(base * 8, base * 16)
        self.d4 = _ResBlock(base * 16 + base * 8, base * 8)
        self.d3 = _ResBlock(base * 8 + base * 4, base * 4)
        self.d2 = _ResBlock(base * 4 + base * 2, base * 2)
        self.d1 = _ResBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, out_ch, 1)
        self.pool = nn.MaxPool2d(2)

    def _up(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b  = self.bottom(self.pool(e4))
        d4 = self.d4(torch.cat([self._up(b),  e4], dim=1))
        d3 = self.d3(torch.cat([self._up(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self._up(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self._up(d2), e1], dim=1))
        return self.out(d1)


# ── Losses & metrics ──

def dice_score(pred_logits: torch.Tensor, target: torch.Tensor,
               eps: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    """Dice over the batch (per-image then mean), binarized."""
    p = (torch.sigmoid(pred_logits) > threshold).float()
    t = (target > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(pred_logits: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6, threshold: float = 0.5) -> torch.Tensor:
    p = (torch.sigmoid(pred_logits) > threshold).float()
    t = (target > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = ((p + t) > 0).float().sum(dim=(1, 2, 3))
    iou = (inter + eps) / (union + eps)
    return iou.mean()


class DiceBCELoss(nn.Module):
    """Standard combined loss: BCE for pixel-level, Dice for region overlap."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dw, self.bw = dice_weight, bce_weight

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor):
        # target in {-1, +1} from the dataset; rescale to {0, 1}
        if target.min() < 0:
            target = (target + 1) / 2
        bce = F.binary_cross_entropy_with_logits(pred_logits, target)
        # soft dice
        p = torch.sigmoid(pred_logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
        return self.bw * bce + self.dw * dice.mean()


# ── Synthetic dataset (loaded from a pre-generated folder) ──

class SynthSegDataset(Dataset):
    """Loads (synthetic_image, mask) pairs that were pre-generated to disk.

    Layout expected:
        root/
            images/idx00000.jpg ... (RGB synthetic MRI)
            masks/idx00000.png  ... (binary PNG, 0=bg / 255=fg)
    """

    def __init__(self, root: str | Path, image_size: int = 256):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.msk_dir = self.root / "masks"
        self.items = sorted([p.stem for p in self.img_dir.glob("*.jpg")])
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        stem = self.items[i]
        img = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        msk = Image.open(self.msk_dir / f"{stem}.png").convert("L")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            msk = msk.resize((self.image_size, self.image_size), Image.NEAREST)
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_t = img_t * 2.0 - 1.0   # match real Dataset convention [-1, 1]
        msk_t = torch.from_numpy((np.array(msk) > 127).astype(np.float32))
        msk_t = msk_t.unsqueeze(0) * 2.0 - 1.0
        # Keys match the real dataset so ConcatDataset + default_collate works.
        return {"image": img_t, "mask": msk_t,
                "subject": f"synth_{stem}", "slice_idx": i}


class MixedDataset(Dataset):
    """Concatenation of two datasets with deterministic interleave."""
    def __init__(self, a: Dataset, b: Dataset):
        self.a, self.b = a, b
    def __len__(self):
        return len(self.a) + len(self.b)
    def __getitem__(self, i):
        if i < len(self.a):
            return self.a[i]
        return self.b[i - len(self.a)]
