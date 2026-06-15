"""LDM++ — adversarial latent diffusion for T2 → ADC (prostate158).

Upgrade over the vanilla conditional LDM (prostate_ldm / cv_ldm): on top of the
standard v-prediction MSE objective in latent space, we additionally

  1. recover the predicted clean latent x̂₀ from the v-prediction,
  2. decode it through the (frozen) VAE to image space,
  3. apply L1 + LPIPS perceptual + conditional PatchGAN adversarial losses
     against the real target image — but ONLY for low-noise timesteps, where
     x̂₀ is a meaningful estimate.

Motivation (from the SOTA medical-translation literature, esp. SynDiff,
Özbey et al. IEEE TMI 2023): pure latent MSE gives strong FID but blurry
per-pixel structure (low SSIM). Adding pixel-space adversarial + perceptual
supervision sharpens output and lifts SSIM while keeping the diffusion
backbone's distributional strength.

The diffusion MSE stays dominant so the model remains a proper sampler; the
pixel losses are a modest-weight refinement signal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Reuse the proven LDM building blocks
from cv_ldm import (   # noqa: F401
    LATENT_CHANNELS, LATENT_SIZE, COND_CHANNELS, IN_CHANNELS,
    VAE_SCALING_FACTOR,
    build_unet, build_train_scheduler, build_inference_scheduler,
    cat_cond, cfg_dropout, sample, EMAModel,
)


def recover_x0_from_v(scheduler, noisy_lat: torch.Tensor,
                      v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Recover the predicted clean latent x̂₀ from a v-prediction.

    With a = sqrt(ᾱ_t), b = sqrt(1-ᾱ_t):
        noisy = a·x₀ + b·ε
        v     = a·ε  − b·x₀
        ⇒ x̂₀ = a·noisy − b·v
    """
    acp = scheduler.alphas_cumprod.to(noisy_lat.device, noisy_lat.dtype)
    a = acp[t].sqrt().view(-1, 1, 1, 1)
    b = (1.0 - acp[t]).sqrt().view(-1, 1, 1, 1)
    return a * noisy_lat - b * v


def training_step_plus(unet, vae, disc,
                       cond_3ch: torch.Tensor, target_3ch: torch.Tensor,
                       scheduler,
                       gan_loss, lpips_fn,
                       cfg_drop_p: float = 0.1,
                       pix_t_threshold: int = 250,
                       lambda_pix: float = 1.0,
                       lambda_perc: float = 1.0,
                       lambda_adv: float = 0.5,
                       max_decode: int = 4):
    """One generator-side training step. Returns (loss_G, parts, fake_for_D).

    cond_3ch / target_3ch: (B, 3, 256, 256) in [-1, 1].
    fake_for_D: decoded x̂₀ images (detached) + their cond, for the D step,
                or None if no sample fell below the timestep threshold.
    """
    device = cond_3ch.device

    with torch.no_grad():
        cond_lat = vae.encode(cond_3ch).latent_dist.sample() * VAE_SCALING_FACTOR
        tgt_lat  = vae.encode(target_3ch).latent_dist.sample() * VAE_SCALING_FACTOR

    cond_lat_d = cfg_dropout(cond_lat, drop_p=cfg_drop_p)

    noise = torch.randn_like(tgt_lat)
    bs = tgt_lat.size(0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (bs,),
                      device=device, dtype=torch.long)
    noisy = scheduler.add_noise(tgt_lat, noise, t)
    v_target = scheduler.get_velocity(tgt_lat, noise, t)

    pred_v = unet(cat_cond(noisy, cond_lat_d), t).sample

    # ── Backbone diffusion loss (dominant) ──
    loss_mse = F.mse_loss(pred_v, v_target)

    parts = {"mse": float(loss_mse.detach().cpu())}
    loss_G = loss_mse
    fake_for_D = None

    # ── Pixel-space refinement at low noise only ──
    low = (t < pix_t_threshold)
    if low.any():
        x0_lat = recover_x0_from_v(scheduler, noisy, pred_v, t)
        idx = low.nonzero(as_tuple=True)[0]
        # Cap decoded subset so VAE-decode-with-grad memory is bounded
        if idx.numel() > max_decode:
            idx = idx[:max_decode]
        sub_lat = x0_lat[idx]
        sub_cond_img = cond_3ch[idx]
        sub_tgt_img  = target_3ch[idx]
        # Decode predicted clean latent → image (grad flows through frozen VAE)
        dec = vae.decode(sub_lat / VAE_SCALING_FACTOR).sample.clamp(-1, 1)

        l_pix  = F.l1_loss(dec, sub_tgt_img)
        l_perc = lpips_fn(dec, sub_tgt_img).mean()

        # Conditional PatchGAN: D sees (cond_img, adc_img) → 6 ch
        d_fake = disc(torch.cat([sub_cond_img, dec], dim=1))
        l_adv = gan_loss(d_fake, True)

        loss_G = (loss_mse
                  + lambda_pix  * l_pix
                  + lambda_perc * l_perc
                  + lambda_adv  * l_adv)
        parts.update({
            "pix":  float(l_pix.detach().cpu()),
            "perc": float(l_perc.detach().cpu()),
            "adv":  float(l_adv.detach().cpu()),
            "n_low": int(idx.numel()),
        })
        fake_for_D = (sub_cond_img.detach(), dec.detach(), sub_tgt_img.detach())

    return loss_G, parts, fake_for_D


def discriminator_step(disc, gan_loss, fake_for_D):
    """One discriminator step on a (cond, fake, real) triple. Returns (loss_D, parts)."""
    cond_img, fake_img, real_img = fake_for_D
    d_real = disc(torch.cat([cond_img, real_img], dim=1))
    d_fake = disc(torch.cat([cond_img, fake_img], dim=1))
    l_real = gan_loss(d_real, True)
    l_fake = gan_loss(d_fake, False)
    loss_D = 0.5 * (l_real + l_fake)
    return loss_D, {"d_real": float(l_real.detach().cpu()),
                    "d_fake": float(l_fake.detach().cpu())}
