"""CV LDM with COMBINED conditioning: MR image latent + vessel-mask one-hot.

Hypothesis: the vessel mask adds vessel-placement constraints on top of the MR's
dense structural prior. If so, FID should drop below cv_ldm.py's 95.7.
If not, masks are redundant given MR conditioning.

Architecture changes vs cv_ldm.py:
  - in_channels = 4 (noisy CT latent) + 4 (MR latent) + 41 (mask one-hot @ 32×32)
                = 49.  Out channels = 4 (CT noise prediction). Everything else same.
  - CFG: JOINT dropout of MR + mask conditioning (both zeroed together with prob p).

The mask is the CT vessel mask (labels 0–40). Nearest-neighbor downsample to
32×32 latent res, then one-hot to 41 channels.
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
LATENT_SIZE = 32                      # SD-VAE 8× downsample
MR_COND_CHANNELS = LATENT_CHANNELS     # MR latent has 4 channels
MASK_NC = 41                          # CT vessel label classes 0..40
IN_CHANNELS = LATENT_CHANNELS + MR_COND_CHANNELS + MASK_NC   # 4 + 4 + 41 = 49
VAE_SCALING_FACTOR = 0.18215


# ── Model ──

def build_unet(block_out_channels=(128, 256, 512, 512),
               layers_per_block: int = 2,
               attention_head_dim: int = 8) -> UNet2DModel:
    """Same UNet shape as cv_ldm.build_unet but in_channels=49 instead of 8."""
    return UNet2DModel(
        sample_size=LATENT_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=LATENT_CHANNELS,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
        ),
        attention_head_dim=attention_head_dim,
    )


# ── Mask preprocessing ──

def mask_to_latent_onehot(mask_long: torch.Tensor) -> torch.Tensor:
    """(B, H, W) long labels in [0, MASK_NC) → (B, MASK_NC, 32, 32) float one-hot.

    Nearest downsample to latent resolution preserves class identity.
    """
    if mask_long.ndim == 4 and mask_long.size(1) == 1:
        mask_long = mask_long.squeeze(1)
    # Downsample to latent size first to keep one-hot tiny
    m = mask_long.float().unsqueeze(1)   # (B, 1, H, W)
    m_lat = F.interpolate(m, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")
    m_lat = m_lat.squeeze(1).long().clamp(0, MASK_NC - 1)
    onehot = F.one_hot(m_lat, num_classes=MASK_NC)
    return onehot.permute(0, 3, 1, 2).float()   # (B, 41, 32, 32)


def cat_cond(noisy_latent, mr_latent, mask_onehot):
    return torch.cat([noisy_latent, mr_latent, mask_onehot], dim=1)


def cfg_dropout(mr_latent: torch.Tensor, mask_onehot: torch.Tensor,
                drop_p: float = 0.1):
    """Joint dropout — zero BOTH conditioning sources together for a fraction
    of samples each step. Keeps the model usable in fully-unconditional mode."""
    B = mr_latent.size(0)
    drop = torch.rand(B, device=mr_latent.device) < drop_p
    if drop.any():
        mr_latent = mr_latent.clone(); mask_onehot = mask_onehot.clone()
        mr_latent[drop] = 0.0
        mask_onehot[drop] = 0.0
    return mr_latent, mask_onehot


# ── Schedulers (same as cv_ldm.py) ──

def build_train_scheduler():
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        timestep_spacing="leading",
    )


def build_inference_scheduler():
    return DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )


# ── Training step ──

def training_step(unet, vae,
                  mr_3ch: torch.Tensor, ct_3ch: torch.Tensor,
                  mask_long: torch.Tensor,
                  scheduler: DDPMScheduler,
                  cfg_drop_p: float = 0.1):
    """One training step.

    mr_3ch, ct_3ch: (B, 3, 256, 256) in [-1, 1]
    mask_long:      (B, 256, 256) long labels 0..40
    """
    device = mr_3ch.device

    with torch.no_grad():
        mr_lat = vae.encode(mr_3ch).latent_dist.sample() * VAE_SCALING_FACTOR
        ct_lat = vae.encode(ct_3ch).latent_dist.sample() * VAE_SCALING_FACTOR

    mask_onehot = mask_to_latent_onehot(mask_long)
    mr_lat, mask_onehot = cfg_dropout(mr_lat, mask_onehot, drop_p=cfg_drop_p)

    noise = torch.randn_like(ct_lat)
    bs = ct_lat.size(0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (bs,),
                      device=device, dtype=torch.long)
    noisy_ct = scheduler.add_noise(ct_lat, noise, t)
    target = scheduler.get_velocity(ct_lat, noise, t)

    unet_in = cat_cond(noisy_ct, mr_lat, mask_onehot)
    pred = unet(unet_in, t).sample
    loss = F.mse_loss(pred, target)
    return loss, {"mse": float(loss.detach().cpu())}


# ── Sampling ──

@torch.no_grad()
def sample(unet, vae,
           mr_3ch: torch.Tensor,
           mask_long: torch.Tensor,
           num_inference_steps: int = 25,
           guidance_scale: float = 1.5,
           generator: torch.Generator | None = None,
           device: str = "cuda") -> torch.Tensor:
    """Sample synthetic CT from MR + mask.

    mr_3ch:    (B, 3, 256, 256) in [-1, 1]
    mask_long: (B, 256, 256) long labels 0..40
    """
    bs = mr_3ch.size(0)
    scheduler = build_inference_scheduler()
    scheduler.set_timesteps(num_inference_steps, device=device)

    mr_lat = vae.encode(mr_3ch).latent_dist.mean * VAE_SCALING_FACTOR
    mask_onehot = mask_to_latent_onehot(mask_long.to(device))
    null_mr   = torch.zeros_like(mr_lat)
    null_mask = torch.zeros_like(mask_onehot)

    z = torch.randn((bs, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                    device=device, generator=generator,
                    dtype=next(unet.parameters()).dtype)
    z = z * scheduler.init_noise_sigma

    use_cfg = guidance_scale > 1.0
    for t in scheduler.timesteps:
        scaled = scheduler.scale_model_input(z, t)
        if use_cfg:
            cat_lat = torch.cat([scaled, scaled], dim=0)
            cat_mr  = torch.cat([null_mr, mr_lat], dim=0)
            cat_mk  = torch.cat([null_mask, mask_onehot], dim=0)
            pred = unet(cat_cond(cat_lat, cat_mr, cat_mk), t).sample
            uncond, cond = pred.chunk(2, dim=0)
            pred = uncond + guidance_scale * (cond - uncond)
        else:
            pred = unet(cat_cond(scaled, mr_lat, mask_onehot), t).sample
        z = scheduler.step(pred, t, z).prev_sample

    images = vae.decode(z / VAE_SCALING_FACTOR).sample
    return images.clamp(-1, 1)


# ── EMA (mirror cv_ldm.EMAModel) ──

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
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}
    def load_state_dict(self, sd, device="cpu"):
        self.decay = sd["decay"]
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}
