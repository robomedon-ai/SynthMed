"""Train pix2pix on PASD mask -> MRI.

Usage:
  python train_pasd_pix2pix.py --modality BTFE --epochs 60 --batch 8 \
                               --out pasd_models/pix2pix_btfe

Conventions follow the existing train_hovernet.py:
  - AdamW + cosine LR
  - AMP fp16
  - Checkpoint each epoch
  - Sample grid every --sample_every epochs (fixed val masks)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

import pasd_data as pd
import pasd_pix2pix as p2p


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] tensor -> [0,255] uint8 numpy (B, H, W, 3)."""
    x = x.clamp(-1, 1).add(1).div(2).mul(255).round()
    x = x.permute(0, 2, 3, 1).byte().cpu().numpy()
    return x


def save_sample_grid(real_imgs, fake_imgs, masks, path: Path):
    """Lay out (mask | real | fake) rows side by side into one PNG."""
    real_np = _denorm(real_imgs)
    fake_np = _denorm(fake_imgs)
    # masks: (B,1,H,W) in [-1,1] -> grayscale uint8
    msk_np = masks.clamp(-1, 1).add(1).div(2).mul(255).round().byte().squeeze(1).cpu().numpy()
    msk_np = np.stack([msk_np, msk_np, msk_np], axis=-1)  # (B, H, W, 3)
    n, H, W, _ = real_np.shape
    grid = np.zeros((n * H, 3 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,       0:W ] = msk_np[i]
        grid[i*H:(i+1)*H,       W:2*W] = real_np[i]
        grid[i*H:(i+1)*H,     2*W:3*W] = fake_np[i]
    Image.fromarray(grid).save(path)


def train(args):
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Data ──
    train_ds = pd.get_pasd_dataset(args.modality, "train",
                                   image_size=args.image_size, augment=True)
    val_ds   = pd.get_pasd_dataset(args.modality, "val",
                                   image_size=args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    print(f"[data] {args.modality} train={len(train_ds)} val={len(val_ds)}")

    # Fixed val batch for sampling
    val_batch = next(iter(DataLoader(val_ds, batch_size=min(8, len(val_ds)),
                                     shuffle=False)))
    val_masks = val_batch["mask"].to(device)
    val_reals = val_batch["image"].to(device)

    # ── Models ──
    G = p2p.UnetGenerator(in_ch=1, out_ch=3, ngf=64).to(device)
    D = p2p.PatchDiscriminator(in_ch=4, ndf=64).to(device)
    p2p.init_weights(G); p2p.init_weights(D)

    # ── Optimizers ──
    opt_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999),
                              weight_decay=1e-4)
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999),
                              weight_decay=1e-4)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=args.epochs)

    # ── Losses ──
    gan_loss = p2p.GANLoss()
    l1_loss = nn.L1Loss()
    scaler_G = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=args.amp)

    # ── Train loop ──
    history = []
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        ep_t = time.time()
        sums = {"g_total": 0.0, "g_gan": 0.0, "g_l1": 0.0,
                "d_real": 0.0, "d_fake": 0.0, "n": 0}

        for batch in train_loader:
            real = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # ── D step ──
            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake = G(mask).detach()
                d_real = D(torch.cat([mask, real], 1))
                d_fake = D(torch.cat([mask, fake], 1))
                loss_d_real = gan_loss(d_real, True)
                loss_d_fake = gan_loss(d_fake, False)
                loss_D = 0.5 * (loss_d_real + loss_d_fake)
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # ── G step ──
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake = G(mask)
                d_fake_for_g = D(torch.cat([mask, fake], 1))
                loss_g_gan = gan_loss(d_fake_for_g, True)
                loss_g_l1 = l1_loss(fake, real) * args.lambda_l1
                loss_G = loss_g_gan + loss_g_l1
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            n = real.size(0)
            sums["g_total"] += loss_G.item() * n
            sums["g_gan"]   += loss_g_gan.item() * n
            sums["g_l1"]    += loss_g_l1.item() * n
            sums["d_real"]  += loss_d_real.item() * n
            sums["d_fake"]  += loss_d_fake.item() * n
            sums["n"]       += n

        sched_G.step(); sched_D.step()
        n = sums["n"]
        epoch_stats = {
            "epoch": epoch,
            "g_total": sums["g_total"] / n,
            "g_gan":   sums["g_gan"]   / n,
            "g_l1":    sums["g_l1"]    / n,
            "d_real":  sums["d_real"]  / n,
            "d_fake":  sums["d_fake"]  / n,
            "lr":      opt_G.param_groups[0]["lr"],
            "time_s":  time.time() - ep_t,
        }
        history.append(epoch_stats)
        print(f"[ep {epoch:3d}/{args.epochs}] "
              f"G={epoch_stats['g_total']:.3f} (gan={epoch_stats['g_gan']:.3f} "
              f"l1={epoch_stats['g_l1']:.3f}) "
              f"D=(real={epoch_stats['d_real']:.3f} fake={epoch_stats['d_fake']:.3f}) "
              f"lr={epoch_stats['lr']:.2e} t={epoch_stats['time_s']:.1f}s", flush=True)

        # ── Sample + checkpoint ──
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                fake_val = G(val_masks)
            save_sample_grid(val_reals, fake_val, val_masks,
                             out_dir / "samples" / f"epoch_{epoch:03d}.png")
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({"G": G.state_dict(), "D": D.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       out_dir / f"ckpt_ep{epoch:03d}.pt")
            torch.save({"G": G.state_dict()}, out_dir / "latest_G.pt")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=("BTFE", "TSE"), default="BTFE")
    p.add_argument("--out", type=str, default="pasd_models/pix2pix_btfe")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--checkpoint_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
