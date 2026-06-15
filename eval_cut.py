"""Evaluate a trained CUT generator on prostate158 T2→ADC.

Full-slice + prostate-ROI metrics, vs the baseline LDM.

Usage:
    python eval_cut.py --ckpt prostate_models/cut_t2_adc/best_G.pt --num 300
"""
from __future__ import annotations
import argparse, time
import numpy as np, torch, torch.nn.functional as F

import prostate_data, prostate_cut as cut
from eval_prostate import _gray_to_rgb_tensor, fidelity, paired
from eval_prostate_roi import mask_bbox, crop_resize


def load_G(ckpt, device):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    ngf = int(sd.get("args", {}).get("ngf", 64))
    G = cut.ResnetGenerator(1, 1, ngf=ngf, n_blocks=9)
    G.load_state_dict(sd["G"]); G.eval()
    for p in G.parameters(): p.requires_grad = False
    return G.to(device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=300)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--roi_size", type=int, default=128)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    G = load_G(args.ckpt, device)
    ds = prostate_data.get_paired_2d_dataset(
        args.split, input_modality="t2", target_modality="adc",
        augment=False, include_mask=True, image_size=256)
    n = min(args.num, len(ds))
    print(f"[eval-cut] {args.ckpt}  N={n}")

    reals, fakes = [], []
    roi_r, roi_f = [], []
    t0 = time.time()
    for i in range(n):
        item = ds[i]
        A = item["input"].unsqueeze(0).to(device)
        adc = item["target"].repeat(3, 1, 1)
        fake1 = G(A)[0].clamp(-1, 1)                       # (1,H,W)
        fake = fake1.repeat(3, 1, 1).cpu()
        reals.append(adc); fakes.append(fake)
        box = mask_bbox(item["mask"].cpu().numpy())
        if box is not None:
            roi_r.append(crop_resize(adc, box, args.roi_size))
            roi_f.append(crop_resize(fake, box, args.roi_size))
    dt = time.time() - t0

    fr, ff = torch.stack(reals), torch.stack(fakes)
    pd = paired(fr, ff, device); fd = fidelity(fr, ff, device)
    print(f"\nFULL-SLICE:  FID {fd['fid']:.2f}  KID {fd['kid_mean']:.4f}  "
          f"SSIM {pd['ssim']:.3f}  PSNR {pd['psnr']:.2f}  LPIPS {pd['lpips']:.3f}  ({dt:.0f}s)")
    if roi_r:
        rp = paired(torch.stack(roi_r), torch.stack(roi_f), device)
        print(f"PROSTATE ROI: SSIM {rp['ssim']:.3f}  PSNR {rp['psnr']:.2f}  LPIPS {rp['lpips']:.3f}  ({len(roi_r)} slices)")
    print(f"\nbaseline LDM (N=300 same harness): FID ~82  SSIM 0.546  PSNR 20.62  | ROI-SSIM 0.541")


if __name__ == "__main__":
    main()
