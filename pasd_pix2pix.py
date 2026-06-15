"""Pix2pix baseline for mask-conditioned PASD MRI synthesis.

Reference: Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks", CVPR 2017.

  Generator     : 8-down + 8-up U-Net, skip connections, instance norm.
  Discriminator : 70x70 PatchGAN.
  Loss          : BCE adversarial + lambda_L1 * L1(fake, real).

Input  = binary placenta mask (1 channel, [-1, 1]).
Output = synthetic MRI slice    (3 channels, [-1, 1], grayscale-replicated).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Generator: 256x256 U-Net (8 down + 8 up) ──

class _UnetSkipBlock(nn.Module):
    """One down/up pair of the symmetric U-Net (Isola et al. style)."""

    def __init__(self, outer_ch: int, inner_ch: int,
                 input_ch: int | None = None,
                 submodule: nn.Module | None = None,
                 outermost: bool = False, innermost: bool = False,
                 use_dropout: bool = False):
        super().__init__()
        self.outermost = outermost
        if input_ch is None:
            input_ch = outer_ch

        downconv = nn.Conv2d(input_ch, inner_ch, 4, stride=2, padding=1,
                             bias=False)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = nn.InstanceNorm2d(inner_ch, affine=True)
        uprelu = nn.ReLU(inplace=True)
        upnorm = nn.InstanceNorm2d(outer_ch, affine=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_ch * 2, outer_ch, 4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_ch, outer_ch, 4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_ch * 2, outer_ch, 4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UnetGenerator(nn.Module):
    """U-Net generator for 256x256 inputs (8 down/up stages)."""

    def __init__(self, in_ch: int = 1, out_ch: int = 3, ngf: int = 64):
        super().__init__()
        # Build innermost-first
        block = _UnetSkipBlock(ngf * 8, ngf * 8, innermost=True)
        for _ in range(3):  # 3 intermediate layers with dropout
            block = _UnetSkipBlock(ngf * 8, ngf * 8, submodule=block,
                                   use_dropout=True)
        block = _UnetSkipBlock(ngf * 4, ngf * 8, submodule=block)
        block = _UnetSkipBlock(ngf * 2, ngf * 4, submodule=block)
        block = _UnetSkipBlock(ngf,     ngf * 2, submodule=block)
        self.model = _UnetSkipBlock(out_ch, ngf, input_ch=in_ch,
                                    submodule=block, outermost=True)

    def forward(self, x):
        return self.model(x)


# ── Discriminator: 70x70 PatchGAN ──

class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN: receives (mask, image) concatenated channels."""

    def __init__(self, in_ch: int = 4, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = ndf
        for n in range(1, n_layers):
            prev_nf, nf = nf, min(ndf * 2 ** n, 512)
            layers += [
                nn.Conv2d(prev_nf, nf, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(nf, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        prev_nf, nf = nf, min(ndf * 2 ** n_layers, 512)
        layers += [
            nn.Conv2d(prev_nf, nf, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(nf, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, mask_and_image):
        return self.model(mask_and_image)


# ── Weight init (DCGAN style) ──

def init_weights(net: nn.Module, init_gain: float = 0.02):
    def _init(m):
        cls = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cls or "Linear" in cls):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "InstanceNorm" in cls and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(_init)


# ── Loss helpers ──

class GANLoss(nn.Module):
    """Standard BCE-with-logits GAN loss for PatchGAN outputs."""

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)
