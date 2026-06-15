"""Stage 1 of the LDM: domain-adapt the pretrained SD-VAE decoder for PASD MRI.

Starting from `stabilityai/sd-vae-ft-mse` and fine-tuning ONLY the decoder
(encoder frozen). This is the standard recipe for adapting Stable Diffusion's
VAE to a new domain at a fraction of the compute of training from scratch.

Latent shape: (4, 32, 32) for a 256x256 input. Scaling factor = 0.18215.

Loss = L1(recon, target) + lpips_weight * LPIPS(recon, target)
KL is NOT included because the encoder is frozen — the latent distribution
stays put, only the decoder shifts to render MRI-like outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import AutoencoderKL


VAE_HF_ID = "stabilityai/sd-vae-ft-mse"


def load_vae(freeze_encoder: bool = True,
             from_pretrained: str = VAE_HF_ID) -> AutoencoderKL:
    """Load the SD-VAE and (optionally) freeze the encoder side."""
    vae = AutoencoderKL.from_pretrained(from_pretrained)
    if freeze_encoder:
        for p in vae.encoder.parameters():
            p.requires_grad = False
        for p in vae.quant_conv.parameters():
            p.requires_grad = False
    return vae


def trainable_parameters(vae: AutoencoderKL):
    """Yield only the parameters that require_grad — pass to the optimizer."""
    for p in vae.parameters():
        if p.requires_grad:
            yield p


def vae_forward(vae: AutoencoderKL, x: torch.Tensor,
                sample_posterior: bool = True
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode -> sample -> decode.

    Args:
        x: (B, 3, H, W) in [-1, 1].
        sample_posterior: if True, sample from q(z|x); if False, use the mean
                          (more deterministic for reconstruction eval).

    Returns:
        recon: (B, 3, H, W) in [-1, 1], decoder output.
        latent: (B, 4, H/8, W/8), the (scaled) latent that was decoded.
    """
    # Encoder runs in no_grad if frozen, but we don't gate here — torch
    # tracks gradients only for trainable params automatically.
    posterior = vae.encode(x).latent_dist
    z = posterior.sample() if sample_posterior else posterior.mean
    z_scaled = z * vae.config.scaling_factor
    recon = vae.decode(z_scaled / vae.config.scaling_factor).sample
    return recon, z_scaled


# ── Reconstruction loss ──

class VAEReconLoss(nn.Module):
    """L1 + LPIPS reconstruction loss for decoder fine-tuning."""

    def __init__(self, lpips_weight: float = 1.0,
                 lpips_net: str = "alex"):
        super().__init__()
        import lpips
        # AlexNet LPIPS is faster than VGG; both are fine for this stage.
        self.lpips_fn = lpips.LPIPS(net=lpips_net, verbose=False)
        for p in self.lpips_fn.parameters():
            p.requires_grad = False
        self.lpips_weight = lpips_weight

    def forward(self, recon: torch.Tensor, target: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(recon, target)
        # LPIPS expects [-1,1]; both inputs are already there
        lp = self.lpips_fn(recon, target).mean()
        total = l1 + self.lpips_weight * lp
        return total, {"l1": float(l1.detach().cpu()),
                       "lpips": float(lp.detach().cpu()),
                       "total": float(total.detach().cpu())}
