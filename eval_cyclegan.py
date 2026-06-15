"""Standard CycleGAN evaluation: per-direction FID + cycle + identity errors.

Usage:
    python eval_cyclegan.py --num 200 --out pasd_eval/cyclegan_btfe_tse

Metrics:
  1. Fidelity (per direction)
     FID(real_TSE,  G_AB(real_BTFE))    — does BTFE→TSE produce TSE-like distribution?
     FID(real_BTFE, G_BA(real_TSE))     — and the reverse?

  2. Cycle reconstruction
     L1(real_BTFE, G_BA(G_AB(real_BTFE)))   — should be small
     L1(real_TSE,  G_AB(G_BA(real_TSE)))    — should be small

  3. Identity
     L1(real_BTFE, G_BA(real_BTFE))   — applying the "to-BTFE" generator to BTFE should be ~identity
     L1(real_TSE,  G_AB(real_TSE))
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
from torch.utils.data import DataLoader

import pasd_data as pd
import generation as gen


def _to_uint8(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(-1, 1).add(1).div(2).mul(255).round().byte()


@torch.no_grad()
def collect(modality: str, split: str, num: int, device: str):
    """Pull `num` images from a split as a (N,3,256,256) tensor in [-1,1]."""
    ds = pd.get_pasd_dataset(modality, split=split, image_size=256, augment=False)
    n = min(num, len(ds))
    imgs = torch.stack([ds[i]["image"] for i in range(n)])
    return imgs.to(device)


def fid_between(reals: torch.Tensor, fakes: torch.Tensor,
                device: str) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(_to_uint8(reals).to(device), real=True)
    fid.update(_to_uint8(fakes).to(device), real=False)
    return float(fid.compute().cpu())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=200,
                   help="Per-modality samples to evaluate")
    p.add_argument("--out", default="pasd_eval/cyclegan_btfe_tse")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] CycleGAN — device={device}, N={args.num} per modality")

    G_AB, G_BA = gen._get_cyclegan(device=device)

    real_A = collect("BTFE", "test", args.num, device)
    real_B = collect("TSE",  "test", args.num, device)

    t0 = time.time()
    with torch.no_grad():
        fake_B = G_AB(real_A)                # BTFE → TSE
        rec_A  = G_BA(fake_B)                # cycle
        idt_B  = G_AB(real_B)                # identity: TSE → fake-TSE-ish

        fake_A = G_BA(real_B)                # TSE → BTFE
        rec_B  = G_AB(fake_A)                # cycle
        idt_A  = G_BA(real_A)                # identity: BTFE → fake-BTFE-ish
    gen_t = time.time() - t0

    # ── Cycle + identity errors (L1, normalized to [0,1] for interpretability) ──
    def l1(a, b):
        return float(F.l1_loss((a + 1) / 2, (b + 1) / 2).cpu())

    paired = {
        "cycle_AtoBtoA_l1":  l1(rec_A, real_A),
        "cycle_BtoAtoB_l1":  l1(rec_B, real_B),
        "identity_BA_on_A_l1": l1(idt_A, real_A),
        "identity_AB_on_B_l1": l1(idt_B, real_B),
    }
    print(f"[eval] L1 cycle  A→B→A = {paired['cycle_AtoBtoA_l1']:.4f}")
    print(f"[eval] L1 cycle  B→A→B = {paired['cycle_BtoAtoB_l1']:.4f}")
    print(f"[eval] L1 identity G_BA(real_A) vs A = {paired['identity_BA_on_A_l1']:.4f}")
    print(f"[eval] L1 identity G_AB(real_B) vs B = {paired['identity_AB_on_B_l1']:.4f}")

    # ── FID per direction ──
    t0 = time.time()
    fid_AB = fid_between(real_B, fake_B, device)   # is fake_B ~ real_B distribution?
    fid_BA = fid_between(real_A, fake_A, device)   # is fake_A ~ real_A distribution?
    print(f"[eval] FID BTFE→TSE (fake_TSE vs real_TSE): {fid_AB:.2f}  ({time.time()-t0:.1f}s)")
    print(f"[eval] FID TSE→BTFE (fake_BTFE vs real_BTFE): {fid_BA:.2f}")

    # ── Save sample grid (4 rows × 6 cols) for the report ──
    n_grid = 4
    def _np(x): return _to_uint8(x[:n_grid]).permute(0, 2, 3, 1).cpu().numpy()
    rA, rB, fA, fB, rcA, rcB = map(_np, [real_A, real_B, fake_A, fake_B, rec_A, rec_B])
    H, W = 256, 256
    grid = np.zeros((n_grid * H, 6 * W, 3), dtype=np.uint8)
    for i in range(n_grid):
        for j, col in enumerate([rA[i], fB[i], rcA[i], rB[i], fA[i], rcB[i]]):
            grid[i*H:(i+1)*H, j*W:(j+1)*W] = col
    Image.fromarray(grid).save(out_dir / "samples.png")

    report = {
        "args": vars(args),
        "n_per_modality": args.num,
        "generation_time_s": round(gen_t, 1),
        "fid_BTFE_to_TSE": fid_AB,
        "fid_TSE_to_BTFE": fid_BA,
        **paired,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] report: {out_dir}/metrics.json   grid: {out_dir}/samples.png")


if __name__ == "__main__":
    main()
