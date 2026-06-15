"""Sweep LDM eval over (inference_steps, guidance_scale) and print a table.

Usage:
    python sweep_ldm_eval.py --modality BTFE --num 200 \
        --out_root pasd_eval/ldm_btfe_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import eval_pasd as ev


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", default="BTFE", choices=("BTFE", "TSE"))
    p.add_argument("--num", type=int, default=200,
                   help="Test samples per combo (fewer = faster; 344 is the full BTFE test)")
    p.add_argument("--steps", nargs="+", type=int, default=[25, 50])
    p.add_argument("--guidances", nargs="+", type=float, default=[1.5, 3.0, 5.0, 7.0])
    p.add_argument("--out_root", default="pasd_eval/ldm_btfe_sweep")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    rows = []
    grand_t0 = time.time()
    for steps in args.steps:
        for g in args.guidances:
            tag = f"s{steps}_g{g}"
            print(f"\n=== sweep {tag} ({len(rows)+1}/{len(args.steps)*len(args.guidances)}) ===",
                  flush=True)
            t0 = time.time()
            reals, fakes, masks = ev.generate_paired_set(
                args.modality, "ldm", "test", args.num, device, seed=0,
                num_inference_steps=steps, guidance_scale=g)
            gen_s = time.time() - t0

            fidkid = ev.fidelity_metrics(reals, fakes, device)
            paired = ev.paired_metrics(reals, fakes, device)
            metr_s = time.time() - t0 - gen_s

            row = {
                "steps": steps, "guidance": g,
                "fid": fidkid["fid"], "kid_mean": fidkid["kid_mean"],
                "ssim": paired["ssim"], "psnr": paired["psnr"],
                "lpips": paired["lpips"],
                "gen_s": round(gen_s, 1), "metr_s": round(metr_s, 1),
            }
            rows.append(row)
            print(f"[{tag}] FID={row['fid']:.2f}  KID={row['kid_mean']:.4f}  "
                  f"SSIM={row['ssim']:.3f}  PSNR={row['psnr']:.2f}dB  "
                  f"LPIPS={row['lpips']:.3f}  ({gen_s:.1f}s)", flush=True)

            ev.save_grid(reals, fakes, masks,
                         out_root / f"samples_{tag}.png", n=8)

    print(f"\n[done] sweep in {time.time()-grand_t0:.1f}s. Combos:")
    # ── Comparison table ──
    print(f"{'steps':>5} {'guid':>5} | {'FID':>7} {'KID':>8} {'SSIM':>6} "
          f"{'PSNR':>6} {'LPIPS':>6}")
    print("-" * 60)
    for r in rows:
        print(f"{r['steps']:>5} {r['guidance']:>5.1f} | "
              f"{r['fid']:>7.2f} {r['kid_mean']:>8.4f} "
              f"{r['ssim']:>6.3f} {r['psnr']:>6.2f} {r['lpips']:>6.3f}")

    # Identify best per metric
    print("\nBest:")
    best_fid = min(rows, key=lambda r: r["fid"])
    best_ssim = max(rows, key=lambda r: r["ssim"])
    best_lpips = min(rows, key=lambda r: r["lpips"])
    print(f"  FID   : steps={best_fid['steps']} g={best_fid['guidance']} -> {best_fid['fid']:.2f}")
    print(f"  SSIM  : steps={best_ssim['steps']} g={best_ssim['guidance']} -> {best_ssim['ssim']:.3f}")
    print(f"  LPIPS : steps={best_lpips['steps']} g={best_lpips['guidance']} -> {best_lpips['lpips']:.3f}")

    with open(out_root / "sweep.json", "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    print(f"\nReport: {out_root}/sweep.json")


if __name__ == "__main__":
    main()
