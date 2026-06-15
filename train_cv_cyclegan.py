"""Train CycleGAN for bidirectional CT ↔ MRA translation on TopBrain.

Mirrors train_pasd_cyclegan.py one-to-one with two differences:
  1. in_ch=1, out_ch=1 (grayscale CT and MRA slices, not RGB)
  2. Reads paired_2d slices from cv_data, then samples each modality
     independently (CycleGAN's "unpaired" regime). Even though TopBrain
     IS paired per slice, training as if it were unpaired tests whether
     cycle-consistency learning generalizes the way it did on PASD
     (where xmod was the headline win — see REPORT.md).

Usage:
    python train_cv_cyclegan.py --epochs 100 --batch 4 \
        --out cv_models/cyclegan_mr_ct
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
from torch.utils.data import DataLoader, Dataset

import cv_data
import pasd_cyclegan as cg


# ── Single-modality wrappers (slice extraction reused from cv_data) ──

class _SingleModality2D(Dataset):
    """Yields one modality of slices from cv_data — for unpaired sampling."""

    def __init__(self, split: str, modality: str, image_size: int = 256,
                 augment: bool = True):
        self.base = cv_data.get_paired_2d_dataset(
            split, augment=augment, image_size=image_size)
        self.modality = modality   # "mr" or "ct"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        return {"image": item[self.modality]}    # (1, H, W) in [-1, 1]


# ── Helpers (mirror train_pasd_cyclegan) ──

def _to_uint8(x: torch.Tensor) -> np.ndarray:
    """(B, 1, H, W) in [-1, 1] -> (B, H, W) uint8."""
    return (x.clamp(-1, 1).add(1).div(2).mul(255).round()
            .squeeze(1).byte().cpu().numpy())


def save_translation_grid(real_A, fake_B, rec_A, real_B, fake_A, rec_B,
                          path: Path):
    """6-col grid (real_A | fake_B | rec_A | real_B | fake_A | rec_B).
    A = MR, B = CT."""
    cols = [_to_uint8(c) for c in (real_A, fake_B, rec_A, real_B, fake_A, rec_B)]
    n, H, W = cols[0].shape
    grid = np.zeros((n * H, 6 * W), dtype=np.uint8)
    for i in range(n):
        for j, c in enumerate(cols):
            grid[i*H:(i+1)*H, j*W:(j+1)*W] = c[i]
    Image.fromarray(grid, "L").save(path)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # A = MR, B = CT  (unpaired even though source data is paired)
    A_train = _SingleModality2D("train", "mr", args.image_size, augment=True)
    B_train = _SingleModality2D("train", "ct", args.image_size, augment=True)
    A_val   = _SingleModality2D("val",   "mr", args.image_size, augment=False)
    B_val   = _SingleModality2D("val",   "ct", args.image_size, augment=False)
    print(f"[data] MR: train={len(A_train)} val={len(A_val)}")
    print(f"[data] CT: train={len(B_train)} val={len(B_val)}")

    A_loader = DataLoader(A_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=args.workers > 0)
    B_loader = DataLoader(B_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=args.workers > 0)

    # Fixed val pair for sample grids (use paired samples for visual clarity)
    A_val_batch = next(iter(DataLoader(A_val, batch_size=min(4, len(A_val)),
                                       shuffle=False)))
    B_val_batch = next(iter(DataLoader(B_val, batch_size=min(4, len(B_val)),
                                       shuffle=False)))
    fixed_A = A_val_batch["image"].to(device)
    fixed_B = B_val_batch["image"].to(device)

    # ── Models — single-channel everywhere ──
    G_AB = cg.ResnetGenerator(1, 1, ngf=args.ngf, n_blocks=9).to(device)
    G_BA = cg.ResnetGenerator(1, 1, ngf=args.ngf, n_blocks=9).to(device)
    D_A  = cg.PatchDiscriminator(in_ch=1, ndf=64).to(device)
    D_B  = cg.PatchDiscriminator(in_ch=1, ndf=64).to(device)
    for m in (G_AB, G_BA, D_A, D_B):
        cg.init_weights(m)
    print(f"[arch] G: {sum(p.numel() for p in G_AB.parameters())/1e6:.1f}M each "
          f"(×2 directions)")

    lsgan = cg.LSGANLoss()
    l1 = nn.L1Loss()
    buf_A = cg.ImageBuffer(50)
    buf_B = cg.ImageBuffer(50)

    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()),
        lr=args.lr, betas=(0.5, 0.999))

    def lr_lambda(epoch):
        warmup = args.epochs // 2
        if epoch < warmup:
            return 1.0
        return max(0.0, 1.0 - (epoch - warmup) / max(1, args.epochs - warmup))
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    scaler_G = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=args.amp)

    history = []
    start_ep = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        G_AB.load_state_dict(ck["G_AB"]); G_BA.load_state_dict(ck["G_BA"])
        D_A.load_state_dict(ck["D_A"]);   D_B.load_state_dict(ck["D_B"])
        start_ep = int(ck.get("epoch", 0)) + 1
        for _ in range(start_ep - 1):
            sched_G.step(); sched_D.step()
        print(f"[resume] from {args.resume} at epoch {start_ep - 1}")

    # ── Best-checkpoint guardrail — saw GAN collapse on PASD CycleGAN ──
    best_balance = -1.0   # higher = healthier (low cycle, balanced D)
    for epoch in range(start_ep, args.epochs + 1):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        ep_t = time.time()
        sums = {"g_gan": 0, "g_cyc": 0, "g_id": 0,
                "d_a": 0, "d_b": 0, "n": 0}
        B_iter = iter(B_loader)
        for A_batch in A_loader:
            try:
                B_batch = next(B_iter)
            except StopIteration:
                B_iter = iter(B_loader)
                B_batch = next(B_iter)
            real_A = A_batch["image"].to(device, non_blocking=True)
            real_B = B_batch["image"].to(device, non_blocking=True)

            # G step
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake_B = G_AB(real_A); rec_A = G_BA(fake_B)
                fake_A = G_BA(real_B); rec_B = G_AB(fake_A)
                idt_A  = G_BA(real_A); idt_B  = G_AB(real_B)
                loss_g_gan = (lsgan(D_B(fake_B), True) +
                              lsgan(D_A(fake_A), True))
                loss_g_cyc = (l1(rec_A, real_A) + l1(rec_B, real_B)) * args.lambda_cycle
                loss_g_id  = (l1(idt_A, real_A) + l1(idt_B, real_B)) * args.lambda_identity
                loss_G = loss_g_gan + loss_g_cyc + loss_g_id
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G); scaler_G.update()

            # D step (with image buffer)
            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake_B_q = buf_B.query(fake_B.detach())
                fake_A_q = buf_A.query(fake_A.detach())
                loss_d_b = 0.5 * (lsgan(D_B(real_B), True) +
                                  lsgan(D_B(fake_B_q), False))
                loss_d_a = 0.5 * (lsgan(D_A(real_A), True) +
                                  lsgan(D_A(fake_A_q), False))
                loss_D = loss_d_a + loss_d_b
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D); scaler_D.update()

            n = real_A.size(0)
            sums["g_gan"] += loss_g_gan.item() * n
            sums["g_cyc"] += loss_g_cyc.item() * n
            sums["g_id"]  += loss_g_id.item()  * n
            sums["d_a"]   += loss_d_a.item()   * n
            sums["d_b"]   += loss_d_b.item()   * n
            sums["n"]     += n

        sched_G.step(); sched_D.step()
        n = sums["n"]
        stats = {"epoch": epoch,
                 "g_gan": sums["g_gan"]/n, "g_cyc": sums["g_cyc"]/n,
                 "g_id": sums["g_id"]/n,
                 "d_a": sums["d_a"]/n, "d_b": sums["d_b"]/n,
                 "lr": opt_G.param_groups[0]["lr"],
                 "time_s": time.time() - ep_t}
        history.append(stats)
        # Balance metric: small cycle loss + non-collapsed D (D > 0.05 each)
        balance = max(0.001, min(stats["d_a"], stats["d_b"])) / max(stats["g_cyc"], 0.01)
        print(f"[ep {epoch:3d}/{args.epochs}] "
              f"G(gan={stats['g_gan']:.3f} cyc={stats['g_cyc']:.3f} "
              f"id={stats['g_id']:.3f})  "
              f"D_A={stats['d_a']:.3f} D_B={stats['d_b']:.3f}  "
              f"lr={stats['lr']:.1e}  bal={balance:.3f}  t={stats['time_s']:.1f}s",
              flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            G_AB.eval(); G_BA.eval()
            with torch.no_grad():
                fB = G_AB(fixed_A); rA = G_BA(fB)
                fA = G_BA(fixed_B); rB = G_AB(fA)
            save_translation_grid(fixed_A, fB, rA, fixed_B, fA, rB,
                                  out_dir / "samples" / f"epoch_{epoch:03d}.png")
            G_AB.train(); G_BA.train()

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                        "D_A": D_A.state_dict(),   "D_B": D_B.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       out_dir / f"ckpt_ep{epoch:03d}.pt")
            torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       out_dir / "latest_G.pt")
            # Track best-balanced checkpoint to avoid PASD CycleGAN's
            # D-collapse-at-final-epoch trap.
            if balance > best_balance and min(stats["d_a"], stats["d_b"]) > 0.05:
                best_balance = balance
                torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                            "epoch": epoch, "args": vars(args),
                            "balance": balance},
                           out_dir / "best_G.pt")
                print(f"  → new best balance {balance:.3f} at ep {epoch}", flush=True)
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cv_models/cyclegan_mr_ct")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_identity", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--checkpoint_every", type=int, default=5)
    p.add_argument("--resume", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
