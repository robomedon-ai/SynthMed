"""TSTR (Train on Synthetic, Test on Real) evaluation for PASD generators.

Pipeline:
  1. prepare  - for a generator (pix2pix or LDM), generate a synthetic image
                for every real training mask. Cache (image, mask) to disk.
  2. train    - fit a small U-Net segmenter under a chosen data condition.
                Modes: real | real_p2p | real_ldm | p2p | ldm
  3. eval     - load each trained segmenter and report Dice / IoU on the
                held-out REAL test set. Output a markdown table.

Usage examples:
  python tstr.py prepare --model pix2pix --modality BTFE
  python tstr.py prepare --model ldm     --modality BTFE
  python tstr.py train --mode real         --modality BTFE
  python tstr.py train --mode real_ldm     --modality BTFE
  python tstr.py train --mode ldm          --modality BTFE
  python tstr.py eval --modality BTFE
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset

import pasd_data as pd
import pasd_seg as seg
import generation as gen


ROOT = Path(__file__).parent
SEG_MODELS_DIR = ROOT / "pasd_models" / "tstr"
SYNTH_DATA_DIR = ROOT / "pasd_synth"
MODES = ("real", "real_p2p", "real_ldm", "p2p", "ldm",
         "xmod", "real_xmod")


# ───────────────────────── prepare ─────────────────────────

@torch.no_grad()
def prepare(args):
    """Generate a synthetic image for every real training (mask, image)
    pair and write to pasd_synth/<model>_<modality>/{images,masks}/."""
    out_root = SYNTH_DATA_DIR / f"{args.model}_{args.modality}"
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)

    ds = pd.get_pasd_dataset(args.modality, split="train",
                             image_size=256, augment=False)
    print(f"[prepare] generating {len(ds)} synth images "
          f"({args.model}/{args.modality}) -> {out_root}", flush=True)
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        mask_t = item["mask"]
        mask_pil = Image.fromarray(
            ((mask_t + 1) / 2 * 255).byte().squeeze(0).cpu().numpy(), "L")
        out = gen.sample(mask_pil, modality=args.modality, model=args.model,
                         num_samples=1, seed=i, device=args.device,
                         watermark=False)
        stem = f"idx{i:05d}"
        out[0].save(out_root / "images" / f"{stem}.jpg", quality=92)
        Image.fromarray(((mask_t + 1) / 2 * 255).byte().squeeze(0).cpu().numpy(),
                        "L").save(out_root / "masks" / f"{stem}.png")
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(ds)}] {(i+1)/(time.time()-t0):.1f} img/s",
                  flush=True)
    print(f"[prepare] done in {time.time()-t0:.1f}s")


@torch.no_grad()
def prepare_xmod(args):
    """Cross-modality prep: translate every real <src> training slice (with its
    mask) into the <target> modality via CycleGAN, and save (synth, mask).

    Result feeds the `xmod` and `real_xmod` training modes. The mask is the
    *original* mask from the source slice — translation is anatomy-preserving
    so the mask still describes the placenta in the synthetic image.
    """
    target = args.modality                        # produced modality
    src = "BTFE" if target == "TSE" else "TSE"    # source modality
    out_root = SYNTH_DATA_DIR / f"xmod_{target}_from_{src}"
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "masks").mkdir(parents=True, exist_ok=True)

    ds = pd.get_pasd_dataset(src, split="train", image_size=256, augment=False)
    direction = f"{src}_to_{target}"
    print(f"[prepare-xmod] {len(ds)} {src}→{target} translations → {out_root}",
          flush=True)
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        # Real source image in [-1,1] (3,H,W) → PIL RGB
        img_t = (item["image"] + 1) / 2 * 255
        img_pil = Image.fromarray(img_t.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy(), "RGB")
        out_pil = gen.translate(img_pil, direction=direction,
                                device=args.device, watermark=False)
        stem = f"idx{i:05d}"
        out_pil.save(out_root / "images" / f"{stem}.jpg", quality=92)
        # Mask is preserved by cycle-consistent translation
        msk = ((item["mask"] + 1) / 2 * 255).clamp(0, 255).byte().squeeze(0).cpu().numpy()
        Image.fromarray(msk, "L").save(out_root / "masks" / f"{stem}.png")
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(ds)}] {(i+1)/(time.time()-t0):.1f} img/s",
                  flush=True)
    print(f"[prepare-xmod] done in {time.time()-t0:.1f}s")


# ───────────────────────── train ─────────────────────────

def _build_loaders(args):
    real_full = pd.get_pasd_dataset(args.modality, split="train",
                                    image_size=256, augment=True)
    val_real = pd.get_pasd_dataset(args.modality, split="val",
                                   image_size=256, augment=False)
    p2p_root = SYNTH_DATA_DIR / f"pix2pix_{args.modality}"
    ldm_root = SYNTH_DATA_DIR / f"ldm_{args.modality}"

    # Deterministic subset of real images (low-data sweep)
    if args.real_n is not None and args.real_n < len(real_full):
        gen = torch.Generator().manual_seed(args.real_n_seed)
        perm = torch.randperm(len(real_full), generator=gen).tolist()[:args.real_n]
        real_ds = torch.utils.data.Subset(real_full, perm)
        print(f"[data] real subset: {args.real_n}/{len(real_full)} "
              f"(seed={args.real_n_seed})")
    else:
        real_ds = real_full

    src_other = "BTFE" if args.modality == "TSE" else "TSE"
    xmod_root = SYNTH_DATA_DIR / f"xmod_{args.modality}_from_{src_other}"

    if args.mode == "real":
        train_ds = real_ds
    elif args.mode == "real_p2p":
        train_ds = ConcatDataset([real_ds, seg.SynthSegDataset(p2p_root)])
    elif args.mode == "real_ldm":
        train_ds = ConcatDataset([real_ds, seg.SynthSegDataset(ldm_root)])
    elif args.mode == "real_xmod":
        train_ds = ConcatDataset([real_ds, seg.SynthSegDataset(xmod_root)])
    elif args.mode == "p2p":
        train_ds = seg.SynthSegDataset(p2p_root)
    elif args.mode == "ldm":
        train_ds = seg.SynthSegDataset(ldm_root)
    elif args.mode == "xmod":
        train_ds = seg.SynthSegDataset(xmod_root)
    else:
        raise ValueError(f"unknown mode {args.mode}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_real, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    return train_loader, val_loader, len(train_ds), len(val_real)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suffix = f"_n{args.real_n}" if args.real_n is not None else ""
    out_dir = SEG_MODELS_DIR / f"{args.mode}_{args.modality}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, n_train, n_val = _build_loaders(args)
    print(f"[train] mode={args.mode} modality={args.modality} "
          f"train_n={n_train} val_n={n_val}")

    model = seg.SmallUNet(in_ch=3, out_ch=1, base=args.base).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"[train] U-Net params: {n_p/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    loss_fn = seg.DiceBCELoss()

    history = []
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "n": 0}
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            sums["loss"] += float(loss) * x.size(0)
            sums["n"] += x.size(0)
        sched.step()

        # Validate on REAL val set
        model.eval()
        v_dice, v_iou, v_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                if y.min() < 0:
                    y = (y + 1) / 2
                logits = model(x)
                v_dice += float(seg.dice_score(logits, y)) * x.size(0)
                v_iou  += float(seg.iou_score(logits, y))  * x.size(0)
                v_n += x.size(0)
        v_dice /= max(v_n, 1)
        v_iou /= max(v_n, 1)
        epoch_stats = {
            "epoch": epoch, "train_loss": sums["loss"] / sums["n"],
            "val_dice_real": v_dice, "val_iou_real": v_iou,
            "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0,
        }
        history.append(epoch_stats)
        print(f"[ep {epoch:2d}/{args.epochs}] "
              f"loss={epoch_stats['train_loss']:.4f}  "
              f"val_dice={v_dice:.4f}  val_iou={v_iou:.4f}  "
              f"({epoch_stats['time_s']:.1f}s)", flush=True)
        if v_dice > best_dice:
            best_dice = v_dice
            torch.save({"model": model.state_dict(),
                        "epoch": epoch, "args": vars(args),
                        "val_dice": v_dice, "val_iou": v_iou},
                       out_dir / "best.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    print(f"[done] best val_dice = {best_dice:.4f}  ({out_dir}/best.pt)")


# ───────────────────────── eval ─────────────────────────

@torch.no_grad()
def evaluate(args):
    """Auto-discover every trained segmenter for this modality and report
    Dice / IoU on the held-out REAL test set."""
    import re
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_ds = pd.get_pasd_dataset(args.modality, split="test",
                                  image_size=256, augment=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"[eval] real test set: {len(test_ds)} items")

    # Discover all subdirs of the form {mode}_{modality}(_nN)?
    pat = re.compile(rf"^(?P<mode>[a-z0-9_]+?)_{args.modality}(_n(?P<n>\d+))?$")
    rows = []
    for sub in sorted(SEG_MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        m = pat.match(sub.name)
        if not m:
            continue
        ckpt = sub / "best.pt"
        if not ckpt.exists():
            print(f"  {sub.name}: NO CHECKPOINT — skipped")
            continue
        sd = torch.load(str(ckpt), map_location=device, weights_only=False)
        model = seg.SmallUNet(in_ch=3, out_ch=1).to(device)
        model.load_state_dict(sd["model"])
        model.eval()

        d, i, n = 0.0, 0.0, 0
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            if y.min() < 0:
                y = (y + 1) / 2
            logits = model(x)
            d += float(seg.dice_score(logits, y)) * x.size(0)
            i += float(seg.iou_score(logits, y))  * x.size(0)
            n += x.size(0)
        rows.append({
            "name": sub.name,
            "mode": m["mode"],
            "real_n": int(m["n"]) if m["n"] else None,
            "dice": d/n, "iou": i/n,
            "best_val_dice": sd.get("val_dice", None),
            "best_epoch": sd.get("epoch", None),
        })

    # Sort: full-data first, then low-data ascending by N, then mode alphabetical
    def key(r):
        return (0 if r["real_n"] is None else 1,
                r["real_n"] or 0,
                r["mode"])
    rows.sort(key=key)

    print("\nTSTR results (held-out REAL test set):")
    print(f"{'name':<22} | {'Dice':>7} | {'IoU':>7} | val_dice | ep")
    print("-" * 64)
    for r in rows:
        bvd = f"{r['best_val_dice']:.4f}" if r["best_val_dice"] is not None else "  -   "
        ep  = f"{r['best_epoch']}"        if r["best_epoch"]    is not None else "-"
        print(f"{r['name']:<22} | {r['dice']:>7.4f} | {r['iou']:>7.4f} | "
              f"{bvd} | {ep}")

    out_path = SEG_MODELS_DIR / f"tstr_{args.modality}.json"
    with open(out_path, "w") as f:
        json.dump({"modality": args.modality, "rows": rows}, f, indent=2)
    print(f"\nReport: {out_path.relative_to(ROOT)}")


# ───────────────────────── CLI ─────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("--model", choices=("pix2pix", "ldm"), required=True)
    pp.add_argument("--modality", choices=("BTFE", "TSE"), default="BTFE")
    pp.add_argument("--device", default="cuda")

    px = sub.add_parser("prepare_xmod",
                        help="Translate training slices from the other modality "
                             "via CycleGAN (with their original masks).")
    px.add_argument("--modality", choices=("BTFE", "TSE"), required=True,
                    help="TARGET modality (source = the other one)")
    px.add_argument("--device", default="cuda")

    pt = sub.add_parser("train")
    pt.add_argument("--mode", choices=MODES, required=True)
    pt.add_argument("--modality", choices=("BTFE", "TSE"), default="BTFE")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch", type=int, default=16)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--base", type=int, default=32)
    pt.add_argument("--workers", type=int, default=4)
    pt.add_argument("--device", default="cuda")
    pt.add_argument("--amp", action="store_true", default=True)
    pt.add_argument("--no-amp", dest="amp", action="store_false")
    pt.add_argument("--real_n", type=int, default=None,
                    help="Cap number of real images (low-data sweep). None = use all.")
    pt.add_argument("--real_n_seed", type=int, default=0,
                    help="Seed for the deterministic real-subset permutation.")

    pe = sub.add_parser("eval")
    pe.add_argument("--modality", choices=("BTFE", "TSE"), default="BTFE")
    pe.add_argument("--device", default="cuda")

    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "prepare_xmod":
        prepare_xmod(args)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
