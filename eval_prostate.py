"""Head-to-head evaluation of prostate158 generators.

Mirrors eval_cv.py. Compares pix2pix (and later LDM, CycleGAN) on the
held-out test patients from prostate_data.

Metrics:
  - FID, KID (distribution match vs real ADCs)
  - Paired SSIM, PSNR, LPIPS (does the synthetic ADC match the real ADC?)

Usage:
    python eval_prostate.py --split test --num 400 --out prostate_eval/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import prostate_data
import generation as gen


def _to_uint8(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-1, 1).add(1).div(2).mul(255).round().byte()


def _gray_to_rgb_tensor(arr01: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr01.astype(np.float32))
    return (t.unsqueeze(0).repeat(3, 1, 1) * 2 - 1).clamp(-1, 1)


@torch.no_grad()
def generate_pairs(model_name: str, split: str, num: int,
                   seed: int = 0, device: str = "cuda"):
    """For `num` slices, return (reals, fakes, inputs) as (B,3,H,W) in [-1,1]."""
    ds = prostate_data.get_paired_2d_dataset(
        split, input_modality="t2", target_modality="adc",
        augment=False, image_size=256)
    n = min(num, len(ds))
    print(f"[gen] {model_name}: {n} slices from {split}")
    reals, fakes, inputs = [], [], []
    for i in range(n):
        item = ds[i]
        t2  = item["input"].repeat(3, 1, 1)     # (3, 256, 256) in [-1, 1]
        adc = item["target"].repeat(3, 1, 1)
        # Convert T2 to PIL for the inference API
        t2_uint = ((item["input"][0] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
        t2_pil = Image.fromarray(t2_uint, "L")
        if model_name == "pix2pix":
            fake_pil = gen.prostate_translate_t2_to_adc(
                t2_pil, device=device, watermark=False)
        elif model_name == "ldm":
            fake_pil = gen.prostate_translate_t2_to_adc_ldm(
                t2_pil, num_inference_steps=25, guidance_scale=1.5,
                seed=seed + i, device=device, watermark=False)
        elif model_name == "cyclegan":
            fake_pil = gen.prostate_translate_cyclegan(
                t2_pil, direction="T2_to_ADC",
                device=device, watermark=False)
        else:
            raise ValueError(f"unknown model: {model_name}")
        fake_arr = np.asarray(fake_pil.convert("L"), dtype=np.float32) / 255.0
        fake_t = _gray_to_rgb_tensor(fake_arr)
        reals.append(adc); fakes.append(fake_t); inputs.append(t2)
    return torch.stack(reals), torch.stack(fakes), torch.stack(inputs)


def fidelity(reals, fakes, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(_to_uint8(reals).to(device), real=True)
    fid.update(_to_uint8(fakes).to(device), real=False)
    kid = KernelInceptionDistance(subset_size=min(50, len(reals)),
                                  normalize=False).to(device)
    kid.update(_to_uint8(reals).to(device), real=True)
    kid.update(_to_uint8(fakes).to(device), real=False)
    km, ks = kid.compute()
    return {"fid": float(fid.compute().cpu()),
            "kid_mean": float(km.cpu()), "kid_std": float(ks.cpu())}


def paired(reals, fakes, device):
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    from torchmetrics.image.psnr import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    r01 = ((reals + 1) / 2).to(device)
    f01 = ((fakes + 1) / 2).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex",
                                                  normalize=False).to(device)
    return {"ssim":  float(ssim(f01, r01).cpu()),
            "psnr":  float(psnr(f01, r01).cpu()),
            "lpips": float(lpips(fakes.to(device), reals.to(device)).cpu())}


def save_grid(inputs, reals, fakes, path: Path, n: int = 6, stride: int = 1):
    """4-col grid (T2 | real ADC | fake ADC | abs diff x4)."""
    # Pick `n` evenly-spaced indices so the grid covers multiple patients/depths
    idxs = list(range(0, len(inputs), max(1, len(inputs) // n)))[:n]
    def _np(x): return _to_uint8(x[idxs]).permute(0, 2, 3, 1).cpu().numpy()
    t2, adc, fk = _np(inputs), _np(reals), _np(fakes)
    diff = np.abs(adc.astype(np.int16) - fk.astype(np.int16)).clip(0, 255).astype(np.uint8) * 4
    diff = diff.clip(0, 255).astype(np.uint8)
    H, W = t2.shape[1], t2.shape[2]
    grid = np.zeros((n * H, 4 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,      0:W ] = t2[i]
        grid[i*H:(i+1)*H,      W:2*W] = adc[i]
        grid[i*H:(i+1)*H,    2*W:3*W] = fk[i]
        grid[i*H:(i+1)*H,    3*W:4*W] = diff[i]
    Image.fromarray(grid).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--num", type=int, default=400,
                   help="Test slices per model (~24 per patient × 19 patients ≈ 460 max for test).")
    p.add_argument("--out", default="prostate_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[eval] device={device}, split={args.split}, N={args.num}")

    report = {"args": vars(args), "models": {}}

    ck_map = {
        "pix2pix":  gen.PROSTATE_PIX2PIX_T2_ADC_CKPT,
        "ldm":      gen.PROSTATE_LDM_T2_ADC_CKPT,
        "cyclegan": gen.PROSTATE_CYCLEGAN_CKPT,
    }
    for name in ("pix2pix", "ldm", "cyclegan"):
        ck = ck_map[name]
        if not ck.exists():
            print(f"[eval] SKIP {name}: no checkpoint")
            continue
        print(f"\n=== {name.upper()} ===")
        t0 = time.time()
        reals, fakes, inputs = generate_pairs(name, args.split, args.num,
                                              args.seed, device)
        gen_s = time.time() - t0
        t0 = time.time()
        fd = fidelity(reals, fakes, device)
        pd = paired(reals, fakes, device)
        metr_s = time.time() - t0
        m = {**fd, **pd, "gen_s": round(gen_s, 1), "metr_s": round(metr_s, 1)}
        report["models"][name] = m
        print(f"[{name}] FID={m['fid']:.2f}  KID={m['kid_mean']:.4f}±{m['kid_std']:.4f}  "
              f"SSIM={m['ssim']:.3f}  PSNR={m['psnr']:.2f}dB  LPIPS={m['lpips']:.3f}  "
              f"(gen {gen_s:.1f}s + metrics {metr_s:.1f}s)")
        save_grid(inputs, reals, fakes, out / f"samples_{name}.png", n=8)

    with open(out / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] → {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
