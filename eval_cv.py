"""Head-to-head evaluation of CV MRA→CTA generators.

Compares pix2pix (deterministic) and LDM (stochastic) on the held-out test
patients from cv_data.subject_split.

Metrics:
  - FID, KID (distribution match vs real CTAs)
  - Paired SSIM, PSNR, LPIPS (does the synthetic CT match the real CT?)

Usage:
    python eval_cv.py --num 200 --out cv_eval/

Note: SPADE mask→CT is intentionally excluded — its task is different
(mask in, CT out; no MR conditioning). See REPORT.md for SPADE's evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import cv_data
import generation as gen


# ── helpers ──

def _to_uint8(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] (B, 3, H, W) → uint8 (B, 3, H, W)."""
    return t.clamp(-1, 1).add(1).div(2).mul(255).round().byte()


def _gray_to_rgb_tensor(arr01: np.ndarray) -> torch.Tensor:
    """(H, W) in [0, 1] → (3, H, W) in [-1, 1]."""
    t = torch.from_numpy(arr01.astype(np.float32))
    return (t.unsqueeze(0).repeat(3, 1, 1) * 2 - 1).clamp(-1, 1)


@torch.no_grad()
def generate_pairs(model_name: str, num: int, seed: int = 0,
                   device: str = "cuda"):
    """For `num` test slices, return (reals, fakes, mrs) as (B,3,H,W) in [-1,1]."""
    ds = cv_data.get_paired_2d_dataset("test", augment=False, image_size=256)
    n = min(num, len(ds))
    print(f"[gen] {model_name}: {n} slices")
    reals, fakes, mrs = [], [], []
    for i in range(n):
        item = ds[i]
        mr = item["mr"].repeat(3, 1, 1)         # (3, 256, 256) in [-1, 1]
        ct = item["ct"].repeat(3, 1, 1)
        # Convert mr to PIL for the inference API
        mr_uint = ((item["mr"][0] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
        mr_pil = Image.fromarray(mr_uint, "L")
        if model_name == "pix2pix":
            fake_pil = gen.cv_translate_mr2ct(mr_pil, device=device, watermark=False)
        elif model_name == "ldm":
            fakes_list = gen.cv_translate_mr2ct_ldm(
                mr_pil, num_samples=1, num_inference_steps=25,
                guidance_scale=1.5, seed=seed + i,
                device=device, watermark=False)
            fake_pil = fakes_list[0]
        else:
            raise ValueError(model_name)
        fake_arr = np.asarray(fake_pil.convert("L"), dtype=np.float32) / 255.0
        fake_t = _gray_to_rgb_tensor(fake_arr)
        reals.append(ct); fakes.append(fake_t); mrs.append(mr)
    return torch.stack(reals), torch.stack(fakes), torch.stack(mrs)


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


def save_grid(mrs, reals, fakes, path: Path, n: int = 6):
    """4-col grid (MR | real CT | fake CT | abs diff x4)."""
    n = min(n, len(mrs))
    def _np(x): return _to_uint8(x[:n]).permute(0, 2, 3, 1).cpu().numpy()
    mr, ct, fk = _np(mrs), _np(reals), _np(fakes)
    diff = np.abs(ct.astype(np.int16) - fk.astype(np.int16)).clip(0, 255).astype(np.uint8) * 4
    diff = diff.clip(0, 255).astype(np.uint8)
    H, W = mr.shape[1], mr.shape[2]
    grid = np.zeros((n * H, 4 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,        0:W ] = mr[i]
        grid[i*H:(i+1)*H,        W:2*W] = ct[i]
        grid[i*H:(i+1)*H,      2*W:3*W] = fk[i]
        grid[i*H:(i+1)*H,      3*W:4*W] = diff[i]
    Image.fromarray(grid).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=200,
                   help="Test slices per model (test set has 3 patients × ~240 = ~720 max)")
    p.add_argument("--out", default="cv_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[eval] device={device}, N={args.num}")

    report = {"args": vars(args), "models": {}}

    for name in ("pix2pix", "ldm"):
        ck_avail = (gen.CV_PIX2PIX_MR2CT_CKPT if name == "pix2pix"
                    else gen.CV_LDM_MR2CT_CKPT).exists()
        if not ck_avail:
            print(f"[eval] SKIP {name}: no checkpoint")
            continue
        print(f"\n=== {name.upper()} ===")
        t0 = time.time()
        reals, fakes, mrs = generate_pairs(name, args.num, args.seed, device)
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
        save_grid(mrs, reals, fakes, out / f"samples_{name}.png", n=6)

    # Comparison table
    if len(report["models"]) >= 2:
        print("\n========================================")
        print(f"{'metric':<14} {'pix2pix':>10} {'ldm':>10}")
        for k in ("fid", "kid_mean", "ssim", "psnr", "lpips"):
            print(f"{k:<14} {report['models']['pix2pix'][k]:>10.4f} "
                  f"{report['models']['ldm'][k]:>10.4f}")

    with open(out / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {out}/metrics.json")


if __name__ == "__main__":
    main()
