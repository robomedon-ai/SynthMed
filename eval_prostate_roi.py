"""ROI-cropped evaluation for fair comparison with prostate-ROI SOTA (e.g. AI-ADC).

Most published prostate T2→ADC papers report SSIM on a cropped prostate ROI,
not the full slice. Background (air/skin/pelvis) inflates full-slice numbers in
some papers and deflates them in others — either way it's not comparable.

This evaluates on the anatomy-mask bounding box (the actual prostate zones,
PZ+CG), padded and resized to 128×128, so our SSIM/PSNR are measured on the
same kind of region the SOTA papers use. Only slices with a non-empty mask are
scored (i.e. slices that actually contain prostate).

Usage:
    python eval_prostate_roi.py --num 400 --ensemble 4 --guidance 1.0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

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
        ema = ldm.EMAModel(unet); ema.load_state_dict(sd["ema"], device="cpu"); ema.apply_to(unet)
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False
    return vae.to(device), unet.to(device)


def mask_bbox(mask2d: np.ndarray, pad: int = 12):
    """Bounding box of nonzero mask, padded, clamped to [0,256]. None if empty."""
    ys, xs = np.where(mask2d > 0)
    if ys.size == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(256, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(256, xs.max() + pad + 1)
    return y0, y1, x0, x1


def crop_resize(img_chw: torch.Tensor, box, size=128):
    """Crop (C,H,W) to box and resize to size×size."""
    y0, y1, x0, x1 = box
    c = img_chw[:, y0:y1, x0:x1].unsqueeze(0)
    c = F.interpolate(c, size=(size, size), mode="bilinear", align_corners=False)
    return c.squeeze(0)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=400)
    p.add_argument("--split", default="test")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--ensemble", type=int, default=1)
    p.add_argument("--roi_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    vae, unet = load_baseline(device)
    ds = prostate_data.get_paired_2d_dataset(
        args.split, input_modality="t2", target_modality="adc",
        augment=False, include_mask=True, image_size=256)
    n = min(args.num, len(ds))
    print(f"[roi-eval] N={n} split={args.split} steps={args.steps} "
          f"cfg={args.guidance} ensemble={args.ensemble} roi={args.roi_size}\n")

    roi_reals, roi_fakes = [], []
    full_reals, full_fakes = [], []
    n_scored = 0
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        mask2d = item["mask"].cpu().numpy()
        box = mask_bbox(mask2d)
        adc = item["target"].repeat(3, 1, 1)
        t2_in = item["input"].repeat(3, 1, 1).unsqueeze(0).to(device)

        acc = None
        for k in range(args.ensemble):
            g = torch.Generator(device=device).manual_seed(i * 100 + k)
            img = ldm.sample(unet, vae, t2_in, num_inference_steps=args.steps,
                             guidance_scale=args.guidance, generator=g, device=device)
            gray = img[0].clamp(-1, 1).add(1).div(2).mean(0)
            acc = gray if acc is None else acc + gray
        fake_gray = (acc / args.ensemble)
        fake = (fake_gray.unsqueeze(0).repeat(3, 1, 1) * 2 - 1).cpu()

        full_reals.append(adc); full_fakes.append(fake)
        if box is not None:
            roi_reals.append(crop_resize(adc, box, args.roi_size))
            roi_fakes.append(crop_resize(fake, box, args.roi_size))
            n_scored += 1
    dt = time.time() - t0

    print(f"slices with prostate ROI: {n_scored}/{n}  (gen {dt:.0f}s)\n")

    fr = torch.stack(full_reals); ff = torch.stack(full_fakes)
    full_pd = paired(fr, ff, device)
    full_fd = fidelity(fr, ff, device)
    print(f"FULL-SLICE:  FID {full_fd['fid']:.2f}  SSIM {full_pd['ssim']:.3f}  "
          f"PSNR {full_pd['psnr']:.2f}  LPIPS {full_pd['lpips']:.3f}")

    if roi_reals:
        rr = torch.stack(roi_reals); rf = torch.stack(roi_fakes)
        roi_pd = paired(rr, rf, device)
        print(f"PROSTATE ROI: SSIM {roi_pd['ssim']:.3f}  PSNR {roi_pd['psnr']:.2f}  "
              f"LPIPS {roi_pd['lpips']:.3f}   ← comparable to AI-ADC's ROI SSIM 0.86")

    print(f"\n(AI-ADC reports SSIM 0.863 on prostate-ROI, in-house data, GAN+contrastive)")


if __name__ == "__main__":
    main()
