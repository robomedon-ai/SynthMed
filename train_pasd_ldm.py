"""Stage 2 of the LDM: train the conditional UNet on PASD mask -> MRI.

Usage (after VAE Stage 1 is done):
    python train_pasd_ldm.py \
        --vae_ckpt pasd_models/vae_btfe/latest.pt \
        --steps 100000 --batch 16 \
        --out pasd_models/ldm_btfe \
        --modality BTFE

To train one model on both BTFE + TSE (recommended — modality is a class label):
    python train_pasd_ldm.py \
        --vae_ckpt pasd_models/vae_btfe/latest.pt \
        --steps 150000 --batch 16 \
        --out pasd_models/ldm_combined \
        --modality both

Expected wall time on RTX 5080 at batch=16: ~12-20h for 100k steps.

Exit criterion (per plan): FID < 108 (i.e. beats pix2pix baseline) on
the held-out test set, and visually plausible CFG samples at scale 3-5.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset

import pasd_data as pd
import pasd_vae as vae_mod
import pasd_ldm as ldm


# ── Helpers ──

def _denorm(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor -> uint8 (B, H, W, 3)."""
    x = x.clamp(-1, 1).add(1).div(2).mul(255).round()
    return x.permute(0, 2, 3, 1).byte().cpu().numpy()


def save_sample_grid(masks: torch.Tensor, reals: torch.Tensor,
                     fakes: torch.Tensor, path: Path):
    """3-column grid (mask | real | LDM sample)."""
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


def load_vae_with_finetuned_decoder(vae_ckpt: str, device: str):
    """Load the pretrained SD-VAE then overlay our fine-tuned decoder weights."""
    vae = vae_mod.load_vae(freeze_encoder=True)
    if vae_ckpt:
        sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
        print(f"[vae] loaded fine-tuned decoder from {vae_ckpt} (step {sd.get('step','?')})")
    else:
        print("[vae] using pretrained SD-VAE (no fine-tune)")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae.to(device)


def build_train_loader(modality: str, image_size: int, batch: int,
                       workers: int):
    """One or both modalities; loader yields image + mask + modality_label."""
    if modality == "both":
        parts = []
        for m in ("BTFE", "TSE"):
            ds = pd.get_pasd_dataset(m, "train", image_size=image_size,
                                     augment=True)
            parts.append(_TaggedDataset(ds, ldm.MODALITY_LABELS[m]))
        ds_all = ConcatDataset(parts)
    else:
        ds = pd.get_pasd_dataset(modality, "train", image_size=image_size,
                                 augment=True)
        ds_all = _TaggedDataset(ds, ldm.MODALITY_LABELS[modality])
    return DataLoader(ds_all, batch_size=batch, shuffle=True,
                      num_workers=workers, pin_memory=True, drop_last=True,
                      persistent_workers=workers > 0), len(ds_all)


class _TaggedDataset(torch.utils.data.Dataset):
    """Wraps a PASD Dataset to add a `modality_label` long tensor."""
    def __init__(self, base, modality_idx: int):
        self.base = base
        self.modality_idx = modality_idx
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        item = self.base[i]
        item["modality_label"] = torch.tensor(self.modality_idx, dtype=torch.long)
        return item


def fixed_val_sample(modality: str, image_size: int, n: int = 6, seed: int = 0):
    """Pick a deterministic small val set for the sample grid."""
    if modality == "both":
        modality = "BTFE"  # show BTFE in samples for both-modality runs
    ds = pd.get_pasd_dataset(modality, "val", image_size=image_size,
                             augment=False)
    n = min(n, len(ds))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), n, replace=False)
    images = torch.stack([ds[int(i)]["image"] for i in indices])
    masks  = torch.stack([ds[int(i)]["mask"]  for i in indices])
    return images, masks


# ── Training loop ──

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Data ──
    loader, n_total = build_train_loader(
        args.modality, args.image_size, args.batch, args.workers)
    print(f"[data] modality={args.modality} train items={n_total} "
          f"batch={args.batch} -> {n_total // args.batch} batches/epoch")

    # Fixed val for sample grids
    fixed_val_imgs, fixed_val_masks = fixed_val_sample(args.modality,
                                                       args.image_size, n=6)
    fixed_val_imgs = fixed_val_imgs.to(device)
    fixed_val_masks = fixed_val_masks.to(device)

    # ── Models ──
    vae = load_vae_with_finetuned_decoder(args.vae_ckpt, device)

    unet = ldm.build_unet().to(device)
    print(f"[unet] params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

    train_sched = ldm.build_train_scheduler()

    # ── Optimizer + scaler + EMA ──
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = ldm.EMAModel(unet, decay=args.ema_decay)

    # ── Resume? ──
    history = []
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ckpt["unet"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"], device=device)
        step = int(ckpt.get("step", 0))
        for _ in range(step):
            sched.step()
        hist_p = out_dir / "history.json"
        if hist_p.exists():
            history = json.load(open(hist_p))
        print(f"[resume] loaded {args.resume} at step {step}")

    # ── Train loop ──
    log_buf = {"mse": 0.0, "n": 0}
    t_log = time.time()
    unet.train()
    data_iter = iter(loader)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x = batch["image"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)
        lbl = batch["modality_label"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss, parts = ldm.training_step(
                unet, vae, x, m, lbl, train_sched,
                cfg_drop_p=args.cfg_drop_p)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(unet)

        log_buf["mse"] += parts["mse"] * x.size(0)
        log_buf["n"]   += x.size(0)
        step += 1

        if step % args.log_every == 0:
            n = log_buf["n"]
            dt = time.time() - t_log
            print(f"[step {step:6d}/{args.steps}] "
                  f"mse={log_buf['mse']/n:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"({n/dt:.1f} samples/s)", flush=True)
            log_buf = {"mse": 0.0, "n": 0}
            t_log = time.time()

        if step % args.sample_every == 0 or step == args.steps:
            ema.apply_to(unet)
            unet.eval()
            with torch.no_grad():
                fake = ldm.sample(unet, vae, fixed_val_masks,
                                  modality=("BTFE" if args.modality == "both"
                                            else args.modality),
                                  num_inference_steps=args.sample_steps,
                                  guidance_scale=args.sample_guidance,
                                  device=str(device))
            save_sample_grid(fixed_val_masks, fixed_val_imgs, fake,
                             out_dir / "samples" / f"step_{step:06d}.png")
            ema.restore(unet)
            unet.train()

        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save({"unet": unet.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step, "args": vars(args)},
                       out_dir / f"ldm_step{step:06d}.pt")
            torch.save({"unet": unet.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step, "args": vars(args)},
                       out_dir / "latest.pt")
            history.append({"step": step,
                            "lr": opt.param_groups[0]["lr"]})
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", required=True,
                   help="Path to fine-tuned VAE checkpoint (or '' to use SD-VAE as-is)")
    p.add_argument("--modality", default="BTFE",
                   choices=("BTFE", "TSE", "both"))
    p.add_argument("--out", default="pasd_models/ldm_btfe")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cfg_drop_p", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--sample_guidance", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--resume", default="",
                   help="Path to an LDM checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
