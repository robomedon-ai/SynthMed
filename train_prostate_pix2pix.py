"""Pix2pix T2 → ADC training on prostate158 (Adams et al. 2022).

Mirrors train_cv_pix2pix.py one-to-one with:
  - input  = T2 slice  (1 channel, [-1, 1])
  - target = ADC slice (1 channel, [-1, 1])
  - discriminator sees concat(input, target) → 2 channels
  - sample grid: (real T2 | real ADC | fake ADC) rows

Usage:
    python train_prostate_pix2pix.py --epochs 60 --batch 8 \
        --out prostate_models/pix2pix_t2_adc
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
from torch.utils.data import DataLoader

import prostate_data
import pasd_pix2pix as p2p  # reuse generator + discriminator + losses


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(-1, 1).add(1).div(2).mul(255).round()
            .squeeze(1).byte().cpu().numpy())


def save_sample_grid(real_in, real_tg, fake_tg, path: Path,
                     col_labels=("T2", "ADC (real)", "ADC (fake)")):
    """3-col grid (input | real target | fake target)."""
    a, b, c = map(_to_uint8, (real_in, real_tg, fake_tg))
    n, H, W = b.shape
    grid = np.zeros((n * H, 3 * W), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,        0:W]   = a[i]
        grid[i*H:(i+1)*H,        W:2*W] = b[i]
        grid[i*H:(i+1)*H,      2*W:3*W] = c[i]
    Image.fromarray(grid, "L").save(path)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_ds = prostate_data.get_paired_2d_dataset(
        "train", input_modality=args.input, target_modality=args.target,
        augment=True, image_size=args.image_size)
    val_ds = prostate_data.get_paired_2d_dataset(
        "val", input_modality=args.input, target_modality=args.target,
        augment=False, image_size=args.image_size)
    print(f"[data] {args.input}→{args.target}  "
          f"train slices={len(train_ds)}  val slices={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=args.workers > 0)
    val_batch = next(iter(DataLoader(val_ds, batch_size=min(8, len(val_ds)),
                                     shuffle=False)))
    val_in = val_batch["input"].to(device)
    val_tg = val_batch["target"].to(device)

    G = p2p.UnetGenerator(in_ch=1, out_ch=1, ngf=64).to(device)
    D = p2p.PatchDiscriminator(in_ch=2, ndf=64).to(device)
    p2p.init_weights(G); p2p.init_weights(D)

    opt_G = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.5, 0.999),
                              weight_decay=1e-4)
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.lr, betas=(0.5, 0.999),
                              weight_decay=1e-4)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=args.epochs)
    gan = p2p.GANLoss(); l1 = nn.L1Loss()
    scaler_G = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=args.amp)

    history = []
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        ep_t = time.time()
        sums = {"g_total": 0, "g_gan": 0, "g_l1": 0,
                "d_real": 0, "d_fake": 0, "n": 0}
        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            # D step
            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake = G(x).detach()
                d_real = D(torch.cat([x, y],    1))
                d_fake = D(torch.cat([x, fake], 1))
                l_dr = gan(d_real, True); l_df = gan(d_fake, False)
                loss_D = 0.5 * (l_dr + l_df)
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D); scaler_D.update()

            # G step
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake = G(x)
                d_for_g = D(torch.cat([x, fake], 1))
                l_gan = gan(d_for_g, True)
                l_l1  = args.lambda_l1 * l1(fake, y)
                loss_G = l_gan + l_l1
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G); scaler_G.update()

            n = x.size(0)
            sums["g_total"] += loss_G.item() * n
            sums["g_gan"]   += l_gan.item()  * n
            sums["g_l1"]    += l_l1.item()   * n
            sums["d_real"]  += l_dr.item()   * n
            sums["d_fake"]  += l_df.item()   * n
            sums["n"]       += n
        sched_G.step(); sched_D.step()
        n = sums["n"]
        stats = {"epoch": epoch,
                 "g_total": sums["g_total"]/n,
                 "g_gan":   sums["g_gan"]/n,
                 "g_l1":    sums["g_l1"]/n,
                 "d_real":  sums["d_real"]/n,
                 "d_fake":  sums["d_fake"]/n,
                 "lr":      opt_G.param_groups[0]["lr"],
                 "time_s":  time.time() - ep_t}
        history.append(stats)
        print(f"[ep {epoch:3d}/{args.epochs}] "
              f"G={stats['g_total']:.3f} (gan={stats['g_gan']:.3f} "
              f"l1={stats['g_l1']:.3f})  "
              f"D=(real={stats['d_real']:.3f} fake={stats['d_fake']:.3f})  "
              f"t={stats['time_s']:.1f}s", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad():
                fake_val = G(val_in)
            save_sample_grid(val_in, val_tg, fake_val,
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
    p.add_argument("--out", default="prostate_models/pix2pix_t2_adc")
    p.add_argument("--input",  default="t2",  choices=("t2", "adc", "dwi"))
    p.add_argument("--target", default="adc", choices=("t2", "adc", "dwi"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--checkpoint_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
