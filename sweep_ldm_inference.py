"""No-retrain inference sweep on the baseline prostate LDM.

Loads the model once, sweeps (guidance, n_ensemble, steps), reports SSIM/PSNR/
FID per config. Ensemble = sample K times with different seeds and average →
reduces stochastic variance, which typically lifts SSIM/PSNR (paired metrics)
at some cost to FID/sharpness.

Usage:
    python sweep_ldm_inference.py --num 100
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import prostate_data
import prostate_vae as vae_mod
import prostate_ldm as ldm
import generation as gen
from eval_prostate import _gray_to_rgb_tensor, fidelity, paired


def load_baseline(device):
    vae = vae_mod.load_vae(freeze_encoder=True)
    sd = torch.load(str(gen.PROSTATE_VAE_CKPT), map_location="cpu", weights_only=False)
    vae.decoder.load_state_dict(sd["decoder"])
    vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    unet = ldm.build_unet()
    sd = torch.load(str(gen.PROSTATE_LDM_T2_ADC_CKPT), map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["unet"])
    if "ema" in sd:
        ema = ldm.EMAModel(unet); ema.load_state_dict(sd["ema"], device="cpu")
        ema.apply_to(unet)
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False
    return vae.to(device), unet.to(device)


@torch.no_grad()
def gen_config(vae, unet, ds, n, steps, guidance, n_ens, device, seed=0):
    reals, fakes = [], []
    for i in range(n):
        item = ds[i]
        adc = item["target"].repeat(3, 1, 1)
        t2_in = item["input"].repeat(3, 1, 1).unsqueeze(0).to(device)
        acc = None
        for k in range(n_ens):
            g = torch.Generator(device=device).manual_seed(seed + i * 100 + k)
            img = ldm.sample(unet, vae, t2_in, num_inference_steps=steps,
                             guidance_scale=guidance, generator=g, device=device)
            gray = img[0].clamp(-1, 1).add(1).div(2).mean(0)  # [0,1]
            acc = gray if acc is None else acc + gray
        mean_gray = (acc / n_ens).cpu().numpy()
        fakes.append(_gray_to_rgb_tensor(mean_gray))
        reals.append(adc)
    return torch.stack(reals), torch.stack(fakes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    vae, unet = load_baseline(device)
    ds = prostate_data.get_paired_2d_dataset(
        args.split, input_modality="t2", target_modality="adc",
        augment=False, image_size=256)
    n = min(args.num, len(ds))
    print(f"[sweep] N={n} split={args.split}\n")

    # (steps, guidance, n_ensemble)
    configs = [
        (25, 1.5, 1),   # current default
        (25, 1.0, 1),   # lower CFG → more faithful
        (25, 1.0, 4),   # + ensemble
        (25, 1.5, 4),   # default CFG + ensemble
        (50, 1.0, 4),   # more steps + ensemble
        (50, 1.0, 8),   # heavy ensemble
    ]
    print(f"{'steps':>5} {'cfg':>4} {'ens':>4} | {'FID':>7} {'KID':>7} "
          f"{'SSIM':>6} {'PSNR':>6} {'LPIPS':>6} | {'time':>6}")
    print("-" * 70)
    rows = []
    for steps, cfg, ens in configs:
        t0 = time.time()
        reals, fakes = gen_config(vae, unet, ds, n, steps, cfg, ens, device)
        fd = fidelity(reals, fakes, device)
        pd = paired(reals, fakes, device)
        dt = time.time() - t0
        rows.append((steps, cfg, ens, fd, pd, dt))
        print(f"{steps:>5} {cfg:>4} {ens:>4} | {fd['fid']:>7.2f} "
              f"{fd['kid_mean']:>7.4f} {pd['ssim']:>6.3f} {pd['psnr']:>6.2f} "
              f"{pd['lpips']:>6.3f} | {dt:>5.0f}s", flush=True)

    print("\nbaseline-of-record (N=400): FID 72.7  SSIM 0.542  PSNR 20.53  LPIPS 0.239")


if __name__ == "__main__":
    main()
