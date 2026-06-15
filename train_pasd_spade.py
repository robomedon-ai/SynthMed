"""Train SPADE-GAN on PASD mask -> MRI.

Usage:
    python train_pasd_spade.py --modality BTFE --epochs 80 --batch 8 \
        --out pasd_models/spade_btfe

Conventions follow train_pasd_pix2pix.py / train_pasd_ldm.py:
  - TTUR: G lr 1e-4, D lr 4e-4 (Heusel et al.; standard for SPADE)
  - Hinge GAN + feature matching (×10) + VGG perceptual (×10)
  - AMP fp16, cosine LR, EMA over G weights (decay 0.999)
  - Sample grid + checkpoint every --sample_every / --checkpoint_every epochs
  - --resume from a previous checkpoint
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

import pasd_data as pd
import pasd_spade as sp


def _denorm(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor -> uint8 numpy (B, H, W, 3)."""
    x = x.clamp(-1, 1).add(1).div(2).mul(255).round()
    return x.permute(0, 2, 3, 1).byte().cpu().numpy()


def save_sample_grid(masks, reals, fakes, path: Path):
    msk = ((masks + 1) / 2 * 255).clamp(0, 255).byte().squeeze(1).cpu().numpy()
    msk = np.stack([msk, msk, msk], axis=-1)
    real = _denorm(reals)
    fake = _denorm(fakes)
    n, H, W, _ = real.shape
    grid = np.zeros((n * H, 3 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,       0:W ] = msk[i]
        grid[i*H:(i+1)*H,       W:2*W] = real[i]
        grid[i*H:(i+1)*H,     2*W:3*W] = fake[i]
    Image.fromarray(grid).save(path)


class EMAWrap:
    """Tiny EMA-of-weights, kept on CPU until used."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().cpu().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None
    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach().cpu(),
                                                 alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self._backup = {n: p.detach().clone() for n, p in model.named_parameters()
                        if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n].to(p.device))
    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self._backup is None: return
        for n, p in model.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None
    def state_dict(self):
        return {"decay": self.decay,
                "shadow": {k: v for k, v in self.shadow.items()}}
    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = {k: v.cpu() for k, v in sd["shadow"].items()}


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=args.workers > 0)
    print(f"[data] {args.modality} train={len(train_ds)} val={len(val_ds)}")

    val_batch = next(iter(DataLoader(val_ds, batch_size=min(8, len(val_ds)),
                                     shuffle=False)))
    val_masks = val_batch["mask"].to(device)
    val_reals = val_batch["image"].to(device)
    # Fix one noise z for the sample grid so progress is comparable
    z_fixed = torch.randn(val_masks.size(0), 256, device=device)

    # ── Models ──
    G = sp.SPADEGenerator(mask_ch=1, image_ch=3, ngf=args.ngf, z_dim=256).to(device)
    D = sp.FeaturePatchDiscriminator(in_ch=4, ndf=64).to(device)
    sp.init_weights(G); sp.init_weights(D)
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"[arch] G={n_g/1e6:.1f}M  D={n_d/1e6:.1f}M  ngf={args.ngf}")

    # ── Losses ──
    gan_loss = sp.HingeGANLoss()
    fm_loss  = sp.FeatureMatchingLoss(weight=args.fm_weight)
    vgg_loss = sp.VGGPerceptualLoss(weight=args.vgg_weight).to(device) \
               if args.vgg_weight > 0 else None
    l1 = nn.L1Loss()

    # ── Optimizers (TTUR) ──
    opt_G = torch.optim.AdamW(G.parameters(), lr=args.g_lr,
                              betas=(0.0, 0.999), weight_decay=1e-4)
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.d_lr,
                              betas=(0.0, 0.999), weight_decay=1e-4)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=args.epochs)
    scaler_G = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = EMAWrap(G, decay=args.ema_decay)

    # ── Resume ──
    history = []
    start_ep = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        if "ema" in ck: ema.load_state_dict(ck["ema"])
        start_ep = int(ck.get("epoch", 0)) + 1
        for _ in range(start_ep - 1):
            sched_G.step(); sched_D.step()
        hist_p = out_dir / "history.json"
        if hist_p.exists():
            history = json.load(open(hist_p))
        print(f"[resume] from {args.resume} at epoch {start_ep - 1}")

    # ── Train loop ──
    for epoch in range(start_ep, args.epochs + 1):
        G.train(); D.train()
        ep_t = time.time()
        sums = {"g_total": 0.0, "g_gan": 0.0, "g_fm": 0.0, "g_vgg": 0.0, "g_l1": 0.0,
                "d_real": 0.0, "d_fake": 0.0, "n": 0}

        for batch in train_loader:
            real = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # ── D step (hinge) ──
            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                with torch.no_grad():
                    fake_for_d = G(mask)
                d_real_logits, _ = D(torch.cat([mask, real], 1))
                d_fake_logits, _ = D(torch.cat([mask, fake_for_d], 1))
                loss_d_real = gan_loss(d_real_logits, True, for_d=True)
                loss_d_fake = gan_loss(d_fake_logits, False, for_d=True)
                loss_D = 0.5 * (loss_d_real + loss_d_fake)
            scaler_D.scale(loss_D).backward()
            scaler_D.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            scaler_D.step(opt_D); scaler_D.update()

            # ── G step (GAN + FM + VGG + L1) ──
            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake = G(mask)
                g_fake_logits, fake_feats = D(torch.cat([mask, fake], 1))
                with torch.no_grad():
                    _, real_feats = D(torch.cat([mask, real], 1))
                loss_g_gan = gan_loss(g_fake_logits, True, for_d=False)
                loss_g_fm  = fm_loss(fake_feats, real_feats)
                loss_g_vgg = vgg_loss(fake, real) if vgg_loss is not None else torch.zeros((), device=device)
                loss_g_l1  = args.l1_weight * l1(fake, real)
                loss_G = loss_g_gan + loss_g_fm + loss_g_vgg + loss_g_l1
            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler_G.step(opt_G); scaler_G.update()
            ema.update(G)

            n = real.size(0)
            sums["g_total"] += loss_G.item() * n
            sums["g_gan"]   += loss_g_gan.item() * n
            sums["g_fm"]    += loss_g_fm.item() * n
            sums["g_vgg"]   += float(loss_g_vgg.detach()) * n
            sums["g_l1"]    += loss_g_l1.item() * n
            sums["d_real"]  += loss_d_real.item() * n
            sums["d_fake"]  += loss_d_fake.item() * n
            sums["n"]       += n

        sched_G.step(); sched_D.step()
        n = sums["n"]
        stats = {
            "epoch": epoch,
            "g_total": sums["g_total"]/n, "g_gan": sums["g_gan"]/n,
            "g_fm": sums["g_fm"]/n, "g_vgg": sums["g_vgg"]/n,
            "g_l1": sums["g_l1"]/n,
            "d_real": sums["d_real"]/n, "d_fake": sums["d_fake"]/n,
            "g_lr": opt_G.param_groups[0]["lr"],
            "d_lr": opt_D.param_groups[0]["lr"],
            "time_s": time.time() - ep_t,
        }
        history.append(stats)
        print(f"[ep {epoch:3d}/{args.epochs}] "
              f"G={stats['g_total']:.3f} (gan={stats['g_gan']:.3f} "
              f"fm={stats['g_fm']:.3f} vgg={stats['g_vgg']:.3f} l1={stats['g_l1']:.3f})  "
              f"D=(real={stats['d_real']:.3f} fake={stats['d_fake']:.3f})  "
              f"t={stats['time_s']:.1f}s", flush=True)

        # Sample + checkpoint
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            ema.apply_to(G); G.eval()
            with torch.no_grad():
                fake_val = G(val_masks, z=z_fixed)
            save_sample_grid(val_masks, val_reals, fake_val,
                             out_dir / "samples" / f"epoch_{epoch:03d}.png")
            ema.restore(G); G.train()

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({"G": G.state_dict(), "D": D.state_dict(),
                        "ema": ema.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       out_dir / f"ckpt_ep{epoch:03d}.pt")
            # latest.pt is the EMA copy ready for inference
            ema.apply_to(G)
            torch.save({"G": G.state_dict(), "ema": ema.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       out_dir / "latest_G.pt")
            ema.restore(G)
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=("BTFE", "TSE"), default="BTFE")
    p.add_argument("--out", default="pasd_models/spade_btfe")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ngf", type=int, default=32)
    p.add_argument("--g_lr", type=float, default=1e-4)
    p.add_argument("--d_lr", type=float, default=4e-4)
    p.add_argument("--fm_weight", type=float, default=10.0)
    p.add_argument("--vgg_weight", type=float, default=10.0)
    p.add_argument("--l1_weight", type=float, default=10.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
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
