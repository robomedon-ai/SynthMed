"""Prostate LDM with COMBINED conditioning: T2 image latent + anatomy-mask one-hot.

Mirrors cv_ldm_combined.py (TopBrain) but for prostate158:
  - conditioning image = T2 latent (4 ch)
  - mask = anatomy mask, 3 classes (0=bg, 1=peripheral zone, 2=central gland)
  - in_channels = 4 (noisy ADC latent) + 4 (T2 latent) + 3 (mask one-hot @ 32×32) = 11

Hypothesis (from [[topbrain-combined-ldm]]): adding the mask trades distributional
fidelity (FID) for paired pixel-match (SSIM/PSNR). On TopBrain that trade was
unfavorable because vessel masks are sparse (~1-2%). Prostate masks are DENSE
(~30% of slice), so the SSIM/PSNR gain may be larger and worthwhile here —
directly targeting our open SSIM gap.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DPMSolverMultistepScheduler


# ── Constants ──

LATENT_CHANNELS = 4
LATENT_SIZE = 32
T2_COND_CHANNELS = LATENT_CHANNELS         # T2 latent has 4 channels
MASK_NC = 3                                # anatomy: bg / PZ / CG
IN_CHANNELS = LATENT_CHANNELS + T2_COND_CHANNELS + MASK_NC   # 4 + 4 + 3 = 11
VAE_SCALING_FACTOR = 0.18215


def build_unet(block_out_channels=(128, 256, 512, 512),
               layers_per_block: int = 2,
               attention_head_dim: int = 8) -> UNet2DModel:
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


def mask_to_latent_onehot(mask_long: torch.Tensor) -> torch.Tensor:
    """(B, H, W) long labels in [0, MASK_NC) → (B, MASK_NC, 32, 32) float one-hot."""
    if mask_long.ndim == 4 and mask_long.size(1) == 1:
        mask_long = mask_long.squeeze(1)
    m = mask_long.float().unsqueeze(1)
    m_lat = F.interpolate(m, size=(LATENT_SIZE, LATENT_SIZE), mode="nearest")
    m_lat = m_lat.squeeze(1).long().clamp(0, MASK_NC - 1)
    onehot = F.one_hot(m_lat, num_classes=MASK_NC)
    return onehot.permute(0, 3, 1, 2).float()


def cat_cond(noisy_latent, img_latent, mask_onehot):
    return torch.cat([noisy_latent, img_latent, mask_onehot], dim=1)


def cfg_dropout(img_latent, mask_onehot, drop_p: float = 0.1):
    """Joint dropout — zero BOTH conditioning sources together for a fraction."""
    B = img_latent.size(0)
    drop = torch.rand(B, device=img_latent.device) < drop_p
    if drop.any():
        img_latent = img_latent.clone(); mask_onehot = mask_onehot.clone()
        img_latent[drop] = 0.0
        mask_onehot[drop] = 0.0
    return img_latent, mask_onehot


def build_train_scheduler():
    return DDPMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="v_prediction",
        timestep_spacing="leading",
    )


def build_inference_scheduler():
    return DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", prediction_type="v_prediction",
        algorithm_type="dpmsolver++", use_karras_sigmas=True,
    )


def training_step(unet, vae,
                  cond_3ch: torch.Tensor, target_3ch: torch.Tensor,
                  mask_long: torch.Tensor,
                  scheduler: DDPMScheduler, cfg_drop_p: float = 0.1):
    """cond_3ch = T2, target_3ch = ADC, both (B,3,256,256) in [-1,1];
    mask_long = (B,256,256) long anatomy labels 0..2."""
    device = cond_3ch.device
    with torch.no_grad():
        img_lat = vae.encode(cond_3ch).latent_dist.sample() * VAE_SCALING_FACTOR
        tgt_lat = vae.encode(target_3ch).latent_dist.sample() * VAE_SCALING_FACTOR
    mask_onehot = mask_to_latent_onehot(mask_long)
    img_lat, mask_onehot = cfg_dropout(img_lat, mask_onehot, drop_p=cfg_drop_p)
    noise = torch.randn_like(tgt_lat)
    bs = tgt_lat.size(0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (bs,),
                      device=device, dtype=torch.long)
    noisy = scheduler.add_noise(tgt_lat, noise, t)
    target = scheduler.get_velocity(tgt_lat, noise, t)
    pred = unet(cat_cond(noisy, img_lat, mask_onehot), t).sample
    loss = F.mse_loss(pred, target)
    return loss, {"mse": float(loss.detach().cpu())}


@torch.no_grad()
def sample(unet, vae, cond_3ch: torch.Tensor, mask_long: torch.Tensor,
           num_inference_steps: int = 25, guidance_scale: float = 1.5,
           generator: torch.Generator | None = None, device: str = "cuda"):
    bs = cond_3ch.size(0)
    scheduler = build_inference_scheduler()
    scheduler.set_timesteps(num_inference_steps, device=device)
    img_lat = vae.encode(cond_3ch).latent_dist.mean * VAE_SCALING_FACTOR
    mask_onehot = mask_to_latent_onehot(mask_long.to(device))
    null_img  = torch.zeros_like(img_lat)
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
            cat_im  = torch.cat([null_img, img_lat], dim=0)
            cat_mk  = torch.cat([null_mask, mask_onehot], dim=0)
            pred = unet(cat_cond(cat_lat, cat_im, cat_mk), t).sample
            uncond, cond = pred.chunk(2, dim=0)
            pred = uncond + guidance_scale * (cond - uncond)
        else:
            pred = unet(cat_cond(scaled, img_lat, mask_onehot), t).sample
        z = scheduler.step(pred, t, z).prev_sample
    return vae.decode(z / VAE_SCALING_FACTOR).sample.clamp(-1, 1)


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
            if n in self.shadow: p.copy_(self.shadow[n])
    @torch.no_grad()
    def restore(self, model):
        if self._backup is None: return
        for n, p in model.named_parameters():
            if n in self._backup: p.copy_(self._backup[n])
        self._backup = None
    def state_dict(self):
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}
    def load_state_dict(self, sd, device="cpu"):
        self.decay = sd["decay"]
        self.shadow = {k: v.to(device) for k, v in sd["shadow"].items()}
