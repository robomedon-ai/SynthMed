"""Evaluate a PASD MRI generator (pix2pix today; LDM later).

Metrics, in three groups per the project plan:

  1. Fidelity (unpaired distribution match)
     - FID  : Frechet Inception Distance
     - KID  : Kernel Inception Distance (more stable on small N)

  2. Spatial faithfulness (paired: same mask -> compare with real slice)
     - SSIM : Structural Similarity
     - PSNR : Peak Signal-to-Noise Ratio
     - LPIPS: Perceptual distance (AlexNet)

  3. Privacy / memorization
     - NN-LPIPS: each synthetic sample's nearest neighbor (LPIPS) among
       the training set. Flags samples that may have memorized training data.

Downstream-utility (TSTR Dice) is intentionally NOT here — it needs a
trained segmenter on real-only / real+synth / synth-only, which is its
own training script. This module's outputs are the prerequisite numbers.

Usage:
    python eval_pasd.py --modality BTFE --model pix2pix --split test \
        --num 64 --out pasd_eval/pix2pix_btfe

Outputs:
    <out>/metrics.json     - all scalar metrics
    <out>/samples.png      - 3-column grid (mask | real | synthetic) for sanity
    <out>/nn_top.png       - the 4 closest (synthetic, nearest-train) pairs
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import pasd_data as pd
import generation as gen


# ── helpers ──

def _to_uint8(t: torch.Tensor) -> torch.Tensor:
    """[-1,1] (B,3,H,W) -> uint8 (B,3,H,W) for FID/KID."""
    return t.clamp(-1, 1).add(1).div(2).mul(255).round().byte()


def _to_pil(t: torch.Tensor) -> Image.Image:
    """[-1,1] (3,H,W) -> PIL RGB."""
    arr = t.clamp(-1, 1).add(1).div(2).mul(255).round().byte()
    return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy(), "RGB")


def _strip_watermark(img: Image.Image) -> Image.Image:
    """Drop the bottom watermark bar so metrics see pure pixels."""
    w, h = img.size
    return img.crop((0, 0, w, h - 22))   # ~22px tall bar


def _img_to_neg1_1_tensor(img: Image.Image, size: int = 256) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1


@torch.no_grad()
def generate_paired_set(modality: str, model_name: str,
                        split: str, num: int, device: str,
                        seed: int = 0,
                        num_inference_steps: int = 25,
                        guidance_scale: float = 3.0):
    """For each of `num` test items: load (real_image, real_mask), then run
    the generator on the mask. Returns three torch tensors stacked along B:
    reals, fakes, masks  (all in [-1,1], (B,3,H,W) for images / (B,1,H,W) for masks).

    num_inference_steps and guidance_scale only matter for stochastic models (LDM).
    """
    ds = pd.get_pasd_dataset(modality, split=split, image_size=256, augment=False)
    indices = list(range(min(num, len(ds))))

    reals, fakes, masks = [], [], []
    for i in indices:
        item = ds[i]
        real_t = item["image"]
        mask_t = item["mask"]
        mask_pil = Image.fromarray(
            ((mask_t + 1) / 2 * 255).byte().squeeze(0).cpu().numpy(), "L")

        fakes_pil = gen.sample(mask_pil, modality=modality,
                               model=model_name, num_samples=1,
                               seed=seed + i, device=device,
                               watermark=False,
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale)
        fake_t = _img_to_neg1_1_tensor(fakes_pil[0])

        reals.append(real_t)
        fakes.append(fake_t)
        masks.append(mask_t)

    return torch.stack(reals), torch.stack(fakes), torch.stack(masks)


# ── metric computations ──

def fidelity_metrics(reals: torch.Tensor, fakes: torch.Tensor,
                     device: str) -> dict:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    out = {}
    # FID
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(_to_uint8(reals).to(device), real=True)
    fid.update(_to_uint8(fakes).to(device), real=False)
    out["fid"] = float(fid.compute().cpu())

    # KID — subset_size must be <= min(real, fake)
    n = min(len(reals), len(fakes))
    subset = min(50, n)
    kid = KernelInceptionDistance(subset_size=subset, normalize=False).to(device)
    kid.update(_to_uint8(reals).to(device), real=True)
    kid.update(_to_uint8(fakes).to(device), real=False)
    kid_mean, kid_std = kid.compute()
    out["kid_mean"] = float(kid_mean.cpu())
    out["kid_std"] = float(kid_std.cpu())
    out["n"] = n
    return out


@torch.no_grad()
def mask_dice_metric(fakes: torch.Tensor, masks: torch.Tensor,
                     modality: str, device: str) -> dict:
    """Mean Dice between segmenter(synthetic) and the input mask.

    Uses the real-only TSTR segmenter as the reference. Measures whether
    the generator placed the placenta where the mask said it should be —
    the metric SPADE-style models are explicitly designed for.
    """
    import generation as gen
    segmenter = gen._get_segmenter(modality, device=device)
    if segmenter is None:
        return {"mean_mask_dice": None, "n": 0,
                "note": "no segmenter checkpoint available"}
    # masks come in [-1, 1]; rescale to {0, 1}
    m01 = ((masks + 1) / 2 > 0.5).float().to(device)
    f = fakes.to(device)
    dices = []
    bs = 8
    for i in range(0, len(f), bs):
        logits = segmenter(f[i:i+bs])
        pred = (torch.sigmoid(logits) > 0.5).float()
        tgt  = m01[i:i+bs]
        inter = (pred * tgt).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
        d = (2 * inter + 1e-6) / (union + 1e-6)
        dices.extend(d.cpu().tolist())
    arr = np.asarray(dices)
    return {
        "mean_mask_dice": float(arr.mean()),
        "std_mask_dice":  float(arr.std()),
        "n": int(len(arr)),
    }


def paired_metrics(reals: torch.Tensor, fakes: torch.Tensor,
                   device: str) -> dict:
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
    from torchmetrics.image.psnr import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    reals_dev = reals.to(device)
    fakes_dev = fakes.to(device)

    # SSIM/PSNR expect [0,1]
    reals_01 = (reals_dev + 1) / 2
    fakes_01 = (fakes_dev + 1) / 2

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    out = {
        "ssim": float(ssim(fakes_01, reals_01).cpu()),
        "psnr": float(psnr(fakes_01, reals_01).cpu()),
    }
    # LPIPS expects [-1,1]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex",
                                                  normalize=False).to(device)
    out["lpips"] = float(lpips(fakes_dev, reals_dev).cpu())
    return out


@torch.no_grad()
def privacy_nn_lpips(fakes: torch.Tensor, modality: str,
                     device: str, train_cap: int = 256) -> dict:
    """For each generated sample, find min LPIPS to train-set images.
    Reports the distribution of nearest-neighbor distances.
    """
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    train_ds = pd.get_pasd_dataset(modality, split="train",
                                   image_size=256, augment=False)
    n_train = min(train_cap, len(train_ds))
    train = torch.stack([train_ds[i]["image"] for i in range(n_train)])
    train_dev = train.to(device)
    fakes_dev = fakes.to(device)

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex",
                                                  normalize=False).to(device)

    nn_dists = []
    nn_idx = []
    # Batch one fake at a time vs all train
    for f in fakes_dev:
        f_rep = f.unsqueeze(0).expand_as(train_dev)
        # LPIPS expects 4D; compute per-sample by chunking through reduction='none'
        # torchmetrics LPIPS doesn't expose reduction; use per-batch loop instead.
        per = []
        for j in range(n_train):
            d = lpips(f.unsqueeze(0), train_dev[j:j+1])
            per.append(float(d.cpu()))
            lpips.reset()
        per = np.array(per)
        nn_dists.append(float(per.min()))
        nn_idx.append(int(per.argmin()))

    nn_dists = np.array(nn_dists)
    return {
        "nn_train_cap": n_train,
        "nn_lpips_min": float(nn_dists.min()),
        "nn_lpips_p05": float(np.percentile(nn_dists, 5)),
        "nn_lpips_p50": float(np.percentile(nn_dists, 50)),
        "nn_lpips_mean": float(nn_dists.mean()),
        "nn_lpips_max": float(nn_dists.max()),
        "nn_indices": nn_idx,
    }


# ── sample grid ──

def save_grid(reals, fakes, masks, path: Path, n: int = 8):
    """3-column grid: (mask | real | synthetic)."""
    n = min(n, len(reals))
    H = reals.shape[-2]
    W = reals.shape[-1]
    out = np.zeros((n * H, 3 * W, 3), dtype=np.uint8)
    for i in range(n):
        m = ((masks[i] + 1) / 2 * 255).byte().squeeze(0).cpu().numpy()
        out[i*H:(i+1)*H,       0:W,  :] = np.stack([m, m, m], axis=-1)
        out[i*H:(i+1)*H,       W:2*W] = ((reals[i] + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
        out[i*H:(i+1)*H,     2*W:3*W] = ((fakes[i] + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(out).save(path)


def save_nn_grid(fakes, modality: str, nn_indices: list[int],
                 path: Path, n: int = 4):
    """Show the worst (= closest to train) synthetic samples next to their NN."""
    n = min(n, len(fakes))
    train_ds = pd.get_pasd_dataset(modality, split="train",
                                   image_size=256, augment=False)
    H = fakes.shape[-2]
    W = fakes.shape[-1]
    out = np.zeros((n * H, 2 * W, 3), dtype=np.uint8)
    for i in range(n):
        out[i*H:(i+1)*H,    0:W,  :] = ((fakes[i] + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
        nn_t = train_ds[nn_indices[i]]["image"]
        out[i*H:(i+1)*H,    W:2*W] = ((nn_t + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(out).save(path)


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=("BTFE", "TSE"), default="BTFE")
    p.add_argument("--model", default="pix2pix")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--num", type=int, default=64,
                   help="N items from split to evaluate (capped at split size)")
    p.add_argument("--out", type=str, default="pasd_eval/pix2pix_btfe")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_privacy", action="store_true")
    p.add_argument("--privacy_cap", type=int, default=128,
                   help="Cap on training samples for NN-LPIPS (cost is O(num * cap))")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=25,
                   help="LDM-only: denoising steps")
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="LDM-only: classifier-free guidance scale")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[eval] modality={args.modality} model={args.model} "
          f"split={args.split} num={args.num} device={device}")

    t0 = time.time()
    reals, fakes, masks = generate_paired_set(
        args.modality, args.model, args.split, args.num, device, args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale)
    print(f"[eval] generated {len(reals)} pairs in {time.time()-t0:.1f}s "
          f"(steps={args.num_inference_steps}, guidance={args.guidance_scale})")

    report = {"args": vars(args)}

    t0 = time.time()
    report["fidelity"] = fidelity_metrics(reals, fakes, device)
    print(f"[eval] FID={report['fidelity']['fid']:.2f} "
          f"KID={report['fidelity']['kid_mean']:.4f}±{report['fidelity']['kid_std']:.4f} "
          f"({time.time()-t0:.1f}s)")

    t0 = time.time()
    report["paired"] = paired_metrics(reals, fakes, device)
    print(f"[eval] SSIM={report['paired']['ssim']:.3f}  "
          f"PSNR={report['paired']['psnr']:.2f}dB  "
          f"LPIPS={report['paired']['lpips']:.3f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    report["mask_faith"] = mask_dice_metric(fakes, masks, args.modality, device)
    if report["mask_faith"]["mean_mask_dice"] is not None:
        print(f"[eval] Mask Dice = {report['mask_faith']['mean_mask_dice']:.4f} "
              f"± {report['mask_faith']['std_mask_dice']:.4f} "
              f"(N={report['mask_faith']['n']}, {time.time()-t0:.1f}s)")
    else:
        print(f"[eval] Mask Dice: SKIPPED ({report['mask_faith'].get('note','')})")

    if not args.skip_privacy:
        t0 = time.time()
        report["privacy"] = privacy_nn_lpips(fakes, args.modality, device,
                                             train_cap=args.privacy_cap)
        p = report["privacy"]
        print(f"[eval] NN-LPIPS: min={p['nn_lpips_min']:.3f} "
              f"p05={p['nn_lpips_p05']:.3f} median={p['nn_lpips_p50']:.3f} "
              f"({time.time()-t0:.1f}s)")
        save_nn_grid(fakes, args.modality, p["nn_indices"],
                     out_dir / "nn_top.png", n=4)

    save_grid(reals, fakes, masks, out_dir / "samples.png", n=8)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] report written to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
