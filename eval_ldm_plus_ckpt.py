"""Evaluate an arbitrary LDM++ checkpoint on prostate158 T2→ADC.

Loads a given ldm_plus checkpoint (EMA-applied), samples ADC for N test slices,
and reports FID / KID / SSIM / PSNR / LPIPS. Lets us compare the pre-collapse
checkpoints (step 4000/6000) against the baseline LDM (FID 72.7 / SSIM 0.542)
without touching generation.py's hardcoded paths.

Usage:
    python eval_ldm_plus_ckpt.py --ckpt prostate_models/ldm_plus_t2_adc/ldm_step006000.pt --num 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import prostate_data
import prostate_vae as vae_mod
import prostate_ldm_plus as ldmp
import generation as gen
from eval_prostate import _gray_to_rgb_tensor, fidelity, paired, save_grid


def load_vae(device):
    vae = vae_mod.load_vae(freeze_encoder=True)
    sd = torch.load(str(gen.PROSTATE_VAE_CKPT), map_location="cpu", weights_only=False)
    vae.decoder.load_state_dict(sd["decoder"])
    vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    return vae.to(device)


def load_unet(ckpt_path, device):
    unet = ldmp.build_unet()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["unet"])
    # Apply EMA shadow weights (these are what we sample from)
    if "ema" in sd:
        ema = ldmp.EMAModel(unet)
        ema.load_state_dict(sd["ema"], device="cpu")
        ema.apply_to(unet)
        print(f"  [ema] applied (step {sd.get('step','?')})")
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False
    return unet.to(device)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", default="")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[eval] {args.ckpt}  split={args.split} N={args.num} "
          f"steps={args.steps} cfg={args.guidance}")

    vae = load_vae(device)
    unet = load_unet(args.ckpt, device)

    ds = prostate_data.get_paired_2d_dataset(
        args.split, input_modality="t2", target_modality="adc",
        augment=False, image_size=256)
    n = min(args.num, len(ds))

    reals, fakes, inputs = [], [], []
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        t2  = item["input"].repeat(3, 1, 1)
        adc = item["target"].repeat(3, 1, 1)
        t2_in = t2.unsqueeze(0).to(device)
        g = torch.Generator(device=device).manual_seed(args.seed + i)
        img = ldmp.sample(unet, vae, t2_in,
                          num_inference_steps=args.steps,
                          guidance_scale=args.guidance,
                          generator=g, device=device)
        fake_gray = img[0].clamp(-1, 1).add(1).div(2).mean(0).cpu().numpy()
        fakes.append(_gray_to_rgb_tensor(fake_gray))
        reals.append(adc); inputs.append(t2)
    gen_s = time.time() - t0

    reals = torch.stack(reals); fakes = torch.stack(fakes); inputs = torch.stack(inputs)
    fd = fidelity(reals, fakes, device)
    pd = paired(reals, fakes, device)
    print(f"\n[RESULT] {Path(args.ckpt).name}")
    print(f"  FID={fd['fid']:.2f}  KID={fd['kid_mean']:.4f}±{fd['kid_std']:.4f}  "
          f"SSIM={pd['ssim']:.3f}  PSNR={pd['psnr']:.2f}dB  LPIPS={pd['lpips']:.3f}  "
          f"(gen {gen_s:.0f}s for {n})")
    print(f"\n  baseline LDM:  FID 72.70  SSIM 0.542  PSNR 20.53  LPIPS 0.239")
    print(f"  baseline p2p:  FID 81.56  SSIM 0.545  PSNR 20.65  LPIPS 0.264")

    if args.grid:
        save_grid(inputs, reals, fakes, Path(args.grid), n=8)
        print(f"  grid → {args.grid}")


if __name__ == "__main__":
    main()
