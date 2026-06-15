"""Stage 2 of the CV LDM: MRA-conditional latent diffusion for MRA → CTA.

Architecture: diffusers' UNet2DModel operating on SD-VAE's (4, 32, 32) latents.
Conditioning: channel-concat the MR latent (encoded from real MRA slice) onto
the noisy CT latent, giving 8 input channels. v-prediction, scaled-linear beta
schedule, classifier-free guidance.

Why this design (instead of mask-conditional like PASD-LDM):
  - PDF specifically cites paired diffusion (Koch et al. on TopCoW) as the
    SOTA for MRA→CTA — direct image-to-image, no masks.
  - Sidesteps the TopBrain mask-sparsity issue ([[topbrain-spade-sparsity]]).
  - Provides a stochastic alternative to deterministic pix2pix_mr2ct.

CFG dropout: with prob `cfg_drop_p` the MR conditioning is zeroed during
training. At sample time the UNet is run twice (uncond / cond) and mixed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)


# ── Constants ──

LATENT_CHANNELS = 4
LATENT_SIZE = 32                # SD-VAE 8× downsample on 256 → 32
COND_CHANNELS = LATENT_CHANNELS  # MR latent has the same 4 channels
IN_CHANNELS = LATENT_CHANNELS + COND_CHANNELS  # 8
VAE_SCALING_FACTOR = 0.18215


# ── Model ──

def build_unet(block_out_channels=(128, 256, 512, 512),
               layers_per_block: int = 2,
               attention_head_dim: int = 8) -> UNet2DModel:
    """Conditional UNet, ~120 M params at defaults — matched to data size."""
    return UNet2DModel(
        sample_size=LATENT_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=LATENT_CHANNELS,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=attention_head_dim,
    )


# ── Helpers ──

def cat_cond(noisy_latent: torch.Tensor, mr_latent: torch.Tensor) -> torch.Tensor:
    """Concatenate (B, 4, 32, 32) noisy CT latent + (B, 4, 32, 32) MR latent."""
    return torch.cat([noisy_latent, mr_latent], dim=1)


def cfg_dropout(mr_latent: torch.Tensor, drop_p: float = 0.1) -> torch.Tensor:
    """Per-sample CFG dropout — zero the MR conditioning with prob `drop_p`."""
    B = mr_latent.size(0)
    drop = torch.rand(B, device=mr_latent.device) < drop_p
    if drop.any():
        mr_latent = mr_latent.clone()
        mr_latent[drop] = 0.0
    return mr_latent


# ── Schedulers ──

def build_train_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        timestep_spacing="leading",
    )


def build_inference_scheduler() -> DPMSolverMultistepScheduler:
    return DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )


# ── Training step ──

def training_step(unet: UNet2DModel, vae,
                  mr_3ch: torch.Tensor, ct_3ch: torch.Tensor,
                  scheduler: DDPMScheduler,
                  cfg_drop_p: float = 0.1) -> tuple[torch.Tensor, dict]:
    """One training step.

    mr_3ch / ct_3ch: (B, 3, 256, 256) in [-1, 1] (grayscale-replicated to 3ch).
    """
    device = mr_3ch.device

    with torch.no_grad():
        mr_lat = vae.encode(mr_3ch).latent_dist.sample() * VAE_SCALING_FACTOR
        ct_lat = vae.encode(ct_3ch).latent_dist.sample() * VAE_SCALING_FACTOR

    mr_lat = cfg_dropout(mr_lat, drop_p=cfg_drop_p)

    noise = torch.randn_like(ct_lat)
    bs = ct_lat.size(0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (bs,),
                      device=device, dtype=torch.long)
    noisy_ct = scheduler.add_noise(ct_lat, noise, t)
    target = scheduler.get_velocity(ct_lat, noise, t)

    unet_in = cat_cond(noisy_ct, mr_lat)
    pred = unet(unet_in, t).sample
    loss = F.mse_loss(pred, target)
    return loss, {"mse": float(loss.detach().cpu())}


# ── Sampling ──

@torch.no_grad()
def sample(unet: UNet2DModel, vae,
           mr_3ch: torch.Tensor,
           num_inference_steps: int = 25,
           guidance_scale: float = 1.5,
           generator: torch.Generator | None = None,
           device: str = "cuda") -> torch.Tensor:
    """Sample a synthetic CT slice from a real MR slice.

    mr_3ch: (B, 3, 256, 256) in [-1, 1].
    Returns: (B, 3, 256, 256) decoded CT in [-1, 1].
    """
    bs = mr_3ch.size(0)
    scheduler = build_inference_scheduler()
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Encode the MR conditioning (no grad, deterministic via posterior.mean)
    mr_lat = vae.encode(mr_3ch).latent_dist.mean * VAE_SCALING_FACTOR
    null_mr = torch.zeros_like(mr_lat)

    z = torch.randn((bs, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                    device=device, generator=generator,
                    dtype=next(unet.parameters()).dtype)
    z = z * scheduler.init_noise_sigma

    use_cfg = guidance_scale > 1.0
    for t in scheduler.timesteps:
        scaled = scheduler.scale_model_input(z, t)
        if use_cfg:
            cat_lat  = torch.cat([scaled, scaled], dim=0)
            cat_cond_lat = torch.cat([null_mr, mr_lat], dim=0)
            pred = unet(cat_cond(cat_lat, cat_cond_lat), t).sample
            uncond, cond = pred.chunk(2, dim=0)
            pred = uncond + guidance_scale * (cond - uncond)
        else:
            pred = unet(cat_cond(scaled, mr_lat), t).sample
        z = scheduler.step(pred, t, z).prev_sample

    images = vae.decode(z / VAE_SCALING_FACTOR).sample
    return images.clamp(-1, 1)


# ── EMA (mirror pasd_ldm.EMAModel) ──

class EMAModel:
    def __init__(self, model, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])
    @torch.no_grad()
    def restore(self, model):
        if self._backup is None: return
        for n, p in model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None
    def state_dict(self):
        return {"decay": self.decay,
                "shadow": {k: v.cpu() for k, v in self.shadow.items()}}
    def load_state_dict(self, sd, device="cpu"):
        self.decay = sd["decay"]
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}
