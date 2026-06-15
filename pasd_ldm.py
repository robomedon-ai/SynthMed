"""Stage 2 of the LDM: conditional latent diffusion U-Net for PASD MRI.

Operates on SD-VAE's (4, 32, 32) latents for 256x256 images.

Conditioning:
  - Spatial: binary placenta mask, downsampled to 32x32, concatenated to the
    noisy latent along channels (so UNet sees 5 channels in).
  - Modality: class embedding {0: BTFE, 1: TSE, 2: NULL (for CFG)}.

Classifier-free guidance:
  - During training, with prob `cfg_drop_p` the mask is replaced with zeros
    AND the modality label is replaced with NULL_CLASS, jointly. This trains
    one network for both conditional and unconditional prediction.
  - At sample time, two forward passes (uncond, cond) are mixed:
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

Prediction parameterization: v-prediction (more stable for smaller models).

SPADE-style modulation of the UNet's GroupNorm blocks is intentionally left
for v2 — channel-concat alone is a strong, well-tested baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    UNet2DModel,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)


# ── Constants ──

LATENT_CHANNELS = 4
LATENT_SIZE = 32            # 256 / 8 (SD-VAE downsample factor)
MASK_CHANNELS = 1
IN_CHANNELS = LATENT_CHANNELS + MASK_CHANNELS  # 5

# Class labels: 0=BTFE, 1=TSE, 2=NULL (used during CFG dropout)
MODALITY_LABELS = {"BTFE": 0, "TSE": 1}
NULL_CLASS = 2
NUM_CLASSES = 3

# Scaling factor used by SD-VAE; latents are stored as z * scaling_factor.
VAE_SCALING_FACTOR = 0.18215


# ── Model ──

def build_unet(block_out_channels=(128, 256, 512, 512),
               layers_per_block: int = 2,
               attention_head_dim: int = 8) -> UNet2DModel:
    """Conditional UNet sized for PASD (~120M params at the defaults).

    Self-attention is enabled at the deeper blocks (mid + one decoder layer).
    """
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
        num_class_embeds=NUM_CLASSES,
    )


# ── Mask preprocessing ──

def mask_to_latent_size(mask_neg1_1: torch.Tensor) -> torch.Tensor:
    """Resize a (B, 1, H, W) mask in [-1, 1] to (B, 1, LATENT_SIZE, LATENT_SIZE).

    Nearest-neighbor to preserve binary edges; result stays in [-1, 1].
    """
    return F.interpolate(mask_neg1_1, size=(LATENT_SIZE, LATENT_SIZE),
                         mode="nearest")


def cat_cond(noisy_latent: torch.Tensor, mask_lat: torch.Tensor
             ) -> torch.Tensor:
    """Concatenate (B, 4, 32, 32) noisy latent + (B, 1, 32, 32) mask channel."""
    return torch.cat([noisy_latent, mask_lat], dim=1)


def cfg_dropout(mask_lat: torch.Tensor, class_labels: torch.Tensor,
                drop_p: float = 0.1
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample joint CFG dropout: with prob `drop_p`, replace BOTH the mask
    (with zeros) and the class label (with NULL_CLASS).

    The mask/class are kept together so the model learns one unconditional
    distribution instead of marginalizing over modality separately.
    """
    B = mask_lat.size(0)
    drop = torch.rand(B, device=mask_lat.device) < drop_p   # (B,)
    if drop.any():
        mask_lat = mask_lat.clone()
        class_labels = class_labels.clone()
        mask_lat[drop] = 0.0
        class_labels[drop] = NULL_CLASS
    return mask_lat, class_labels


# ── Schedulers ──

def build_train_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        timestep_spacing="leading",
    )


def build_inference_scheduler() -> DPMSolverMultistepScheduler:
    return DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )


# ── Training step (one minibatch) ──

def training_step(unet: UNet2DModel, vae,
                  x: torch.Tensor, mask: torch.Tensor,
                  modality_label: torch.Tensor,
                  scheduler: DDPMScheduler,
                  cfg_drop_p: float = 0.1) -> tuple[torch.Tensor, dict]:
    """One training step (no optimizer step / no AMP wrapping here).

    Args:
        unet : conditional UNet
        vae  : frozen (or fine-tuned-decoder) AutoencoderKL — only its encoder
               is called here, with no grad
        x    : (B, 3, 256, 256) in [-1, 1] — real images
        mask : (B, 1, 256, 256) in [-1, 1] — binary masks
        modality_label : (B,) long tensor, values in {BTFE=0, TSE=1}
    """
    device = x.device

    # Encode to latent space (no grad; VAE encoder is frozen).
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        z = posterior.sample() * VAE_SCALING_FACTOR

    # Mask to latent resolution
    mask_lat = mask_to_latent_size(mask)
    # CFG dropout
    mask_lat, class_labels = cfg_dropout(mask_lat, modality_label, cfg_drop_p)

    # Sample noise + timestep
    noise = torch.randn_like(z)
    bs = z.size(0)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                              (bs,), device=device, dtype=torch.long)

    noisy_z = scheduler.add_noise(z, noise, timesteps)
    # v-prediction target
    target = scheduler.get_velocity(z, noise, timesteps)

    # Forward
    unet_in = cat_cond(noisy_z, mask_lat)
    pred = unet(unet_in, timesteps, class_labels=class_labels).sample
    loss = F.mse_loss(pred, target)

    return loss, {"mse": float(loss.detach().cpu())}


# ── CFG sampling ──

@torch.no_grad()
def sample(unet: UNet2DModel, vae,
           mask: torch.Tensor, modality: str = "BTFE",
           num_inference_steps: int = 25,
           guidance_scale: float = 3.0,
           generator: torch.Generator | None = None,
           device: str = "cuda") -> torch.Tensor:
    """Sample synthetic images from a batch of masks.

    Args:
        mask : (B, 1, 256, 256) in [-1, 1].
    Returns:
        (B, 3, 256, 256) tensor in [-1, 1] — VAE-decoded synthetic images.
    """
    if modality not in MODALITY_LABELS:
        raise ValueError(f"modality must be one of {list(MODALITY_LABELS)}")
    modality_idx = MODALITY_LABELS[modality]
    bs = mask.size(0)

    scheduler = build_inference_scheduler()
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Initial random latent
    z = torch.randn((bs, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE),
                    device=device, generator=generator,
                    dtype=next(unet.parameters()).dtype)
    z = z * scheduler.init_noise_sigma

    mask_lat = mask_to_latent_size(mask.to(device=device,
                                           dtype=z.dtype))
    null_mask = torch.zeros_like(mask_lat)

    cond_labels = torch.full((bs,), modality_idx, device=device,
                             dtype=torch.long)
    uncond_labels = torch.full((bs,), NULL_CLASS, device=device,
                               dtype=torch.long)

    use_cfg = guidance_scale > 1.0

    for t in scheduler.timesteps:
        scaled = scheduler.scale_model_input(z, t)
        if use_cfg:
            # Run both passes in one forward via batch concat
            cat_lat = torch.cat([scaled, scaled], dim=0)
            cat_mask = torch.cat([null_mask, mask_lat], dim=0)
            cat_labels = torch.cat([uncond_labels, cond_labels], dim=0)
            pred = unet(cat_cond(cat_lat, cat_mask), t,
                        class_labels=cat_labels).sample
            uncond, cond = pred.chunk(2, dim=0)
            pred = uncond + guidance_scale * (cond - uncond)
        else:
            pred = unet(cat_cond(scaled, mask_lat), t,
                        class_labels=cond_labels).sample

        z = scheduler.step(pred, t, z).prev_sample

    # Decode through VAE (no grad, full precision is fine here)
    images = vae.decode(z / VAE_SCALING_FACTOR).sample
    return images.clamp(-1, 1)


# ── EMA (exponential moving average of UNet weights) ──

class EMAModel:
    """Lightweight EMA shadow of a model's parameters.

    Standard for diffusion training: keep an EMA copy that's used at inference;
    significantly stabilizes sample quality vs the raw training weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """Swap the model's weights with the EMA weights. Call restore() later."""
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is None:
            return
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
