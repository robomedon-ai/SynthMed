"""Stage 1 of the LDM: domain-adapt the SD-VAE decoder on PASD MRI.

Usage:
    python train_pasd_vae.py --modality BTFE --steps 8000 \
        --out pasd_models/vae_btfe

Expected wall time on RTX 5080: ~1-2 hours for 8000 steps at batch=8.

Exit criterion (per plan): reconstruction PSNR >= 30 dB on val.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import pasd_data as pd
import pasd_vae as vae_mod


def _denorm_to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor -> uint8 numpy (B, H, W, 3)."""
    x = x.clamp(-1, 1).add(1).div(2).mul(255).round()
    return x.permute(0, 2, 3, 1).byte().cpu().numpy()


def save_recon_grid(originals: torch.Tensor, recons: torch.Tensor,
                    path: Path):
    """Save side-by-side (original | recon) rows as a single PNG."""
    o = _denorm_to_uint8(originals)
    r = _denorm_to_uint8(recons)
    n, H, W, _ = o.shape
    grid = np.zeros((n * H, 2 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,  0:W ] = o[i]
        grid[i*H:(i+1)*H,  W:2*W] = r[i]
    Image.fromarray(grid).save(path)


@torch.no_grad()
def evaluate(vae, val_loader, recon_loss_fn, device, max_batches: int = 8):
    """Compute mean L1 / LPIPS / PSNR on a few val batches."""
    vae.eval()
    sums = {"l1": 0.0, "lpips": 0.0, "psnr": 0.0, "n": 0}
    for batch in islice(val_loader, max_batches):
        x = batch["image"].to(device)
        recon, _ = vae_mod.vae_forward(vae, x, sample_posterior=False)
        _, parts = recon_loss_fn(recon, x)
        mse = torch.nn.functional.mse_loss(
            (recon + 1) / 2, (x + 1) / 2).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
        n = x.size(0)
        sums["l1"]    += parts["l1"]    * n
        sums["lpips"] += parts["lpips"] * n
        sums["psnr"]  += psnr           * n
        sums["n"]     += n
    vae.train()
    n = max(sums["n"], 1)
    return {"l1": sums["l1"]/n, "lpips": sums["lpips"]/n, "psnr": sums["psnr"]/n}


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
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    print(f"[data] {args.modality} train={len(train_ds)} val={len(val_ds)}")

    # Fixed val batch for sample grids
    fixed_val = next(iter(DataLoader(val_ds, batch_size=min(8, len(val_ds)),
                                     shuffle=False)))
    fixed_val_x = fixed_val["image"].to(device)

    # ── Model + loss ──
    print(f"[vae] loading pretrained {vae_mod.VAE_HF_ID}")
    vae = vae_mod.load_vae(freeze_encoder=True).to(device)
    n_train = sum(p.numel() for p in vae_mod.trainable_parameters(vae))
    n_total = sum(p.numel() for p in vae.parameters())
    print(f"[vae] params: trainable={n_train/1e6:.1f}M / total={n_total/1e6:.1f}M")

    recon_loss = vae_mod.VAEReconLoss(lpips_weight=args.lpips_weight).to(device)

    # ── Optimizer ──
    opt = torch.optim.AdamW(vae_mod.trainable_parameters(vae),
                            lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # ── Optional: resume from a previous checkpoint ──
    history = []
    step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume path not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        vae.decoder.load_state_dict(ckpt["decoder"])
        vae.post_quant_conv.load_state_dict(ckpt["post_quant_conv"])
        step = int(ckpt.get("step", 0))
        # Advance the LR scheduler to the saved step
        for _ in range(step):
            sched.step()
        print(f"[resume] loaded {resume_path} at step {step}; "
              f"LR now {opt.param_groups[0]['lr']:.2e}")
        # Restore history if present
        hist_path = out_dir / "history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            print(f"[resume] history: {len(history)} val records")

    # ── Train loop (step-based, not epoch-based) ──
    t0 = time.time()
    log_buf = {"l1": 0.0, "lpips": 0.0, "n": 0}
    vae.train()

    train_iter = iter(train_loader)
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch["image"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            recon, _ = vae_mod.vae_forward(vae, x, sample_posterior=True)
            loss, parts = recon_loss(recon, x)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        log_buf["l1"]    += parts["l1"]    * x.size(0)
        log_buf["lpips"] += parts["lpips"] * x.size(0)
        log_buf["n"]     += x.size(0)
        step += 1

        if step % args.log_every == 0:
            n = log_buf["n"]
            dt = time.time() - t0
            print(f"[step {step:5d}/{args.steps}] "
                  f"l1={log_buf['l1']/n:.4f} lpips={log_buf['lpips']/n:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} "
                  f"({n/dt:.1f} samples/s)", flush=True)
            log_buf = {"l1": 0.0, "lpips": 0.0, "n": 0}
            t0 = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            val_metrics = evaluate(vae, val_loader, recon_loss, device,
                                   max_batches=args.eval_batches)
            print(f"  [val] l1={val_metrics['l1']:.4f} "
                  f"lpips={val_metrics['lpips']:.4f} "
                  f"PSNR={val_metrics['psnr']:.2f}dB", flush=True)
            history.append({"step": step, **val_metrics})
            with torch.no_grad():
                recon_val, _ = vae_mod.vae_forward(vae, fixed_val_x,
                                                   sample_posterior=False)
            save_recon_grid(fixed_val_x, recon_val,
                            out_dir / "samples" / f"step_{step:05d}.png")
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        if step % args.checkpoint_every == 0 or step == args.steps:
            # Save only the decoder + post_quant_conv (the trainable parts);
            # at inference we reload pretrained encoder.
            ckpt = {
                "decoder": vae.decoder.state_dict(),
                "post_quant_conv": vae.post_quant_conv.state_dict(),
                "step": step,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / f"vae_step{step:05d}.pt")
            torch.save(ckpt, out_dir / "latest.pt")

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=("BTFE", "TSE"), default="BTFE")
    p.add_argument("--out", type=str, default="pasd_models/vae_btfe")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lpips_weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--resume", type=str, default="",
                   help="Path to a checkpoint to resume from (decoder + post_quant_conv)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
