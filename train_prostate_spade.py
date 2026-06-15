"""Train SPADE for anatomy_mask → T2 on prostate158.

Mirrors train_cv_spade.py with three differences:
  - label_nc = 3 (prostate has only bg / PZ / CG)
  - image target = T2 (not CT)
  - data via prostate_data with include_mask=True

Tests the prostate-vs-TopBrain SPADE hypothesis: prostate masks are far DENSER
(~30% of slice) than TopBrain vessel masks (~1-2%), so SPADE should have much
more conditioning signal per pixel — should perform meaningfully better here.

Usage:
    python train_prostate_spade.py --epochs 100 --batch 4 \
        --out prostate_models/spade_mask_t2
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
import prostate_spade as sp


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(-1, 1).add(1).div(2).mul(255).round()
            .squeeze(1).byte().cpu().numpy())


def _mask_vis(mask_long: torch.Tensor) -> np.ndarray:
    m = mask_long.clamp(0, sp.PROSTATE_LABEL_NC - 1).float()
    m = (m / max(sp.PROSTATE_LABEL_NC - 1, 1)) * 255
    return m.byte().cpu().numpy()


def save_sample_grid(masks, reals, fakes, path: Path):
    """3-col grid (mask | real T2 | fake T2)."""
    msk  = _mask_vis(masks)
    real = _to_uint8(reals)
    fake = _to_uint8(fakes)
    n, H, W = real.shape
    grid = np.zeros((n * H, 3 * W), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,        0:W ] = msk[i]
        grid[i*H:(i+1)*H,        W:2*W] = real[i]
        grid[i*H:(i+1)*H,      2*W:3*W] = fake[i]
    Image.fromarray(grid, "L").save(path)


class EMAWrap:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().cpu().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self._backup = None
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach().cpu(),
                                                 alpha=1 - self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        self._backup = {n: p.detach().clone()
                        for n, p in model.named_parameters()
                        if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n].to(p.device))
    @torch.no_grad()
    def restore(self, model):
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
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # target_modality="t2" — SPADE outputs T2 from anatomy mask
    train_ds = prostate_data.get_paired_2d_dataset(
        "train", input_modality="t2", target_modality="t2",
        augment=True, include_mask=True, image_size=args.image_size)
    val_ds = prostate_data.get_paired_2d_dataset(
        "val", input_modality="t2", target_modality="t2",
        augment=False, include_mask=True, image_size=args.image_size)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=args.workers > 0)
    val_batch = next(iter(DataLoader(val_ds, batch_size=min(8, len(val_ds)),
                                     shuffle=False)))
    val_masks_long = val_batch["mask"].to(device)
    val_masks_oh   = sp.one_hot_mask(val_masks_long)
    val_t2         = val_batch["target"].to(device)
    z_fixed = torch.randn(val_t2.size(0), 256, device=device)

    G = sp.ProstateSPADEGenerator(mask_nc=sp.PROSTATE_LABEL_NC, image_ch=1,
                                   ngf=args.ngf, z_dim=256).to(device)
    D = sp.FeaturePatchDiscriminator(in_ch=sp.PROSTATE_LABEL_NC + 1,
                                      ndf=64).to(device)
    sp.init_weights(G); sp.init_weights(D)
    print(f"[arch] G={sum(p.numel() for p in G.parameters())/1e6:.1f}M  "
          f"D={sum(p.numel() for p in D.parameters())/1e6:.1f}M  "
          f"(mask_nc={sp.PROSTATE_LABEL_NC})")

    gan_loss = sp.HingeGANLoss()
    fm_loss  = sp.FeatureMatchingLoss(weight=args.fm_weight)
    vgg_loss = (sp.VGGPerceptualLoss(weight=args.vgg_weight).to(device)
                if args.vgg_weight > 0 else None)
    l1 = nn.L1Loss()

    opt_G = torch.optim.AdamW(G.parameters(), lr=args.g_lr,
                              betas=(0.0, 0.999), weight_decay=1e-4)
    opt_D = torch.optim.AdamW(D.parameters(), lr=args.d_lr,
                              betas=(0.0, 0.999), weight_decay=1e-4)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.epochs)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=args.epochs)
    scaler_G = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = EMAWrap(G, decay=args.ema_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        ep_t = time.time()
        sums = {"g_total": 0, "g_gan": 0, "g_fm": 0, "g_vgg": 0, "g_l1": 0,
                "d_real": 0, "d_fake": 0, "n": 0}

        for batch in train_loader:
            real = batch["target"].to(device, non_blocking=True)
            mask_long = batch["mask"].to(device, non_blocking=True)
            mask_oh = sp.one_hot_mask(mask_long)

            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                with torch.no_grad():
                    fake_for_d = G(mask_oh)
                dr_logits, _ = D(torch.cat([mask_oh, real], 1))
                df_logits, _ = D(torch.cat([mask_oh, fake_for_d], 1))
                l_dr = gan_loss(dr_logits, True,  for_d=True)
                l_df = gan_loss(df_logits, False, for_d=True)
                loss_D = 0.5 * (l_dr + l_df)
            scaler_D.scale(loss_D).backward()
            scaler_D.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            scaler_D.step(opt_D); scaler_D.update()

            opt_G.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake = G(mask_oh)
                gf_logits, fake_feats = D(torch.cat([mask_oh, fake], 1))
                with torch.no_grad():
                    _, real_feats = D(torch.cat([mask_oh, real], 1))
                l_gan = gan_loss(gf_logits, True, for_d=False)
                l_fm  = fm_loss(fake_feats, real_feats)
                if vgg_loss is not None:
                    l_vgg = vgg_loss(fake.repeat(1, 3, 1, 1),
                                     real.repeat(1, 3, 1, 1))
                else:
                    l_vgg = torch.zeros((), device=device)
                l_l1 = args.l1_weight * l1(fake, real)
                loss_G = l_gan + l_fm + l_vgg + l_l1
            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            scaler_G.step(opt_G); scaler_G.update()
            ema.update(G)

            n = real.size(0)
            sums["g_total"] += loss_G.item() * n
            sums["g_gan"]   += l_gan.item()  * n
            sums["g_fm"]    += l_fm.item()   * n
            sums["g_vgg"]   += float(l_vgg.detach()) * n
            sums["g_l1"]    += l_l1.item()   * n
            sums["d_real"]  += l_dr.item()   * n
            sums["d_fake"]  += l_df.item()   * n
            sums["n"]       += n

        sched_G.step(); sched_D.step()
        n = sums["n"]
        stats = {"epoch": epoch,
                 "g_total": sums["g_total"]/n, "g_gan": sums["g_gan"]/n,
                 "g_fm": sums["g_fm"]/n, "g_vgg": sums["g_vgg"]/n,
                 "g_l1": sums["g_l1"]/n,
                 "d_real": sums["d_real"]/n, "d_fake": sums["d_fake"]/n,
                 "time_s": time.time() - ep_t}
        history.append(stats)
        print(f"[ep {epoch:3d}/{args.epochs}] "
              f"G={stats['g_total']:.3f} (gan={stats['g_gan']:.3f} "
              f"fm={stats['g_fm']:.3f} vgg={stats['g_vgg']:.3f} "
              f"l1={stats['g_l1']:.3f})  "
              f"D=(real={stats['d_real']:.3f} fake={stats['d_fake']:.3f})  "
              f"t={stats['time_s']:.1f}s", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            ema.apply_to(G); G.eval()
            with torch.no_grad():
                fake_val = G(val_masks_oh, z=z_fixed)
            save_sample_grid(val_masks_long, val_t2, fake_val,
                             out_dir / "samples" / f"epoch_{epoch:03d}.png")
            ema.restore(G); G.train()

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({"G": G.state_dict(), "D": D.state_dict(),
                        "ema": ema.state_dict(),
                        "epoch": epoch, "args": vars(args)},
                       out_dir / f"ckpt_ep{epoch:03d}.pt")
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
    p.add_argument("--out", default="prostate_models/spade_mask_t2")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ngf", type=int, default=32)
    p.add_argument("--g_lr", type=float, default=1e-4)
    p.add_argument("--d_lr", type=float, default=4e-4)
    p.add_argument("--fm_weight", type=float, default=10.0)
    p.add_argument("--vgg_weight", type=float, default=10.0)
    p.add_argument("--l1_weight", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--checkpoint_every", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
