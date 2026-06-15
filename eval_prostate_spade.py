"""Eval SPADE anatomy_mask → T2 on prostate158.

Different task than pix2pix/LDM/CycleGAN (mask→T2 vs T2→ADC), so we score it
against the real T2 distribution and against the corresponding real T2 slice.

Usage:
    python eval_prostate_spade.py --split test --num 400 --out prostate_eval/
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
from eval_prostate import _to_uint8, _gray_to_rgb_tensor, fidelity, paired, save_grid


@torch.no_grad()
def generate_pairs_spade(split: str, num: int, seed: int = 0,
                          device: str = "cuda"):
    ds = prostate_data.get_paired_2d_dataset(
        split, input_modality="t2", target_modality="t2",
        augment=False, include_mask=True, image_size=256)
    n = min(num, len(ds))
    print(f"[gen] SPADE: {n} slices from {split}")
    reals, fakes, masks_vis = [], [], []
    for i in range(n):
        item = ds[i]
        t2  = item["target"].repeat(3, 1, 1)
        mask_long = item["mask"]
        fake_pil = gen.prostate_sample_mask_to_t2(
            mask_long, seed=seed + i, device=device, watermark=False)
        fake_arr = np.asarray(fake_pil.convert("L"), dtype=np.float32) / 255.0
        fake_t = _gray_to_rgb_tensor(fake_arr)
        # Mask visualisation as 3-ch for the grid
        m = mask_long.float().clamp(0, 2) / 2.0
        mask_t = (m * 2 - 1).unsqueeze(0).repeat(3, 1, 1)
        reals.append(t2); fakes.append(fake_t); masks_vis.append(mask_t)
    return torch.stack(reals), torch.stack(fakes), torch.stack(masks_vis)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--num", type=int, default=400)
    p.add_argument("--out", default="prostate_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[eval] device={device}, split={args.split}, N={args.num}")

    ck = gen.PROSTATE_SPADE_CKPT
    if not ck.exists():
        print(f"[eval] no SPADE checkpoint at {ck}")
        return

    t0 = time.time()
    reals, fakes, masks_vis = generate_pairs_spade(args.split, args.num,
                                                    args.seed, device)
    gen_s = time.time() - t0
    t0 = time.time()
    fd = fidelity(reals, fakes, device)
    pd = paired(reals, fakes, device)
    metr_s = time.time() - t0
    m = {**fd, **pd, "gen_s": round(gen_s, 1), "metr_s": round(metr_s, 1)}

    # Merge into existing metrics.json
    metrics_path = out / "metrics.json"
    report = {"models": {}}
    if metrics_path.exists():
        report = json.load(open(metrics_path))
    report.setdefault("models", {})["spade"] = m
    json.dump(report, open(metrics_path, "w"), indent=2)

    print(f"[spade] FID={m['fid']:.2f}  KID={m['kid_mean']:.4f}±{m['kid_std']:.4f}  "
          f"SSIM={m['ssim']:.3f}  PSNR={m['psnr']:.2f}dB  LPIPS={m['lpips']:.3f}  "
          f"(gen {gen_s:.1f}s + metrics {metr_s:.1f}s)")
    save_grid(masks_vis, reals, fakes, out / "samples_spade.png", n=8)
    print(f"[done] → {metrics_path}")


if __name__ == "__main__":
    main()
