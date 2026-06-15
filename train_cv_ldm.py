"""Train the CV LDM (MRA → CTA via latent diffusion) on TopBrain.

Usage (after train_cv_vae.py finishes):
    python train_cv_ldm.py \
        --vae_ckpt cv_models/vae/latest.pt \
        --steps 30000 --batch 16 \
        --out cv_models/ldm_mr2ct

Mirrors train_pasd_ldm.py — v-prediction, AMP, EMA, sample grids, --resume.
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

import cv_data
import cv_vae as vae_mod
import cv_ldm as ldm


def _denorm_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] tensor → uint8 (B, H, W, 3)."""
    return x.clamp(-1, 1).add(1).div(2).mul(255).round().permute(
        0, 2, 3, 1).byte().cpu().numpy()


def save_grid(mrs, cts, fakes, path: Path):
    """3-col grid: (real MR | real CT | fake CT). Inputs are 3-channel."""
    mr_np = _denorm_uint8(mrs)
    ct_np = _denorm_uint8(cts)
    fk_np = _denorm_uint8(fakes)
    n, H, W, _ = ct_np.shape
    grid = np.zeros((n * H, 3 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,        0:W ] = mr_np[i]
        grid[i*H:(i+1)*H,        W:2*W] = ct_np[i]
        grid[i*H:(i+1)*H,      2*W:3*W] = fk_np[i]
    Image.fromarray(grid).save(path)


def load_vae_with_finetuned_decoder(vae_ckpt: str, device: str):
    """Load SD-VAE then overlay our fine-tuned CV decoder weights."""
    vae = vae_mod.load_vae(freeze_encoder=True)
    if vae_ckpt:
        sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
        print(f"[vae] loaded fine-tuned decoder from {vae_ckpt} "
              f"(step {sd.get('step', '?')})")
    else:
        print("[vae] using pretrained SD-VAE as-is")
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    return vae.to(device)


def get_paired_3ch_loader(split: str, image_size: int, batch: int, workers: int):
    """Wrap cv_data with grayscale→RGB replication for SD-VAE."""
    base = cv_data.get_paired_2d_dataset(split, augment=(split == "train"),
                                         image_size=image_size)

    class _Paired3ch(torch.utils.data.Dataset):
        def __len__(self): return len(base)
        def __getitem__(self, i):
            item = base[i]
            return {"mr": item["mr"].repeat(3, 1, 1),
                    "ct": item["ct"].repeat(3, 1, 1)}

    return DataLoader(_Paired3ch(), batch_size=batch, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=True,
                      drop_last=(split == "train"),
                      persistent_workers=workers > 0), len(base)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    loader, n_train = get_paired_3ch_loader(
        "train", args.image_size, args.batch, args.workers)
    val_loader, _ = get_paired_3ch_loader(
        "val", args.image_size, min(args.batch, 4), args.workers)
    print(f"[data] train slices={n_train}  batch={args.batch}")

    fixed_val = next(iter(val_loader))
    fixed_mr = fixed_val["mr"].to(device)
    fixed_ct = fixed_val["ct"].to(device)

    vae = load_vae_with_finetuned_decoder(args.vae_ckpt, device)
    unet = ldm.build_unet().to(device)
    print(f"[unet] params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")
    train_sched = ldm.build_train_scheduler()

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = ldm.EMAModel(unet, decay=args.ema_decay)

    history = []
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ck["unet"])
        if "ema" in ck: ema.load_state_dict(ck["ema"], device=device)
        step = int(ck.get("step", 0))
        for _ in range(step): sched.step()
        print(f"[resume] from {args.resume} at step {step}")

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
        mr = batch["mr"].to(device, non_blocking=True)
        ct = batch["ct"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss, parts = ldm.training_step(unet, vae, mr, ct, train_sched,
                                            cfg_drop_p=args.cfg_drop_p)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        ema.update(unet)
        log_buf["mse"] += parts["mse"] * mr.size(0); log_buf["n"] += mr.size(0)
        step += 1

        if step % args.log_every == 0:
            n = log_buf["n"]; dt = time.time() - t_log
            print(f"[step {step:6d}/{args.steps}] mse={log_buf['mse']/n:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  ({n/dt:.1f} samples/s)",
                  flush=True)
            log_buf = {"mse": 0.0, "n": 0}; t_log = time.time()

        if step % args.sample_every == 0 or step == args.steps:
            ema.apply_to(unet); unet.eval()
            with torch.no_grad():
                fake = ldm.sample(unet, vae, fixed_mr,
                                  num_inference_steps=args.sample_steps,
                                  guidance_scale=args.sample_guidance,
                                  device=str(device))
            save_grid(fixed_mr, fixed_ct, fake,
                      out_dir / "samples" / f"step_{step:06d}.png")
            ema.restore(unet); unet.train()

        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save({"unet": unet.state_dict(), "ema": ema.state_dict(),
                        "step": step, "args": vars(args)},
                       out_dir / f"ldm_step{step:06d}.pt")
            torch.save({"unet": unet.state_dict(), "ema": ema.state_dict(),
                        "step": step, "args": vars(args)},
                       out_dir / "latest.pt")
            history.append({"step": step, "lr": opt.param_groups[0]["lr"]})
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", required=True,
                   help="Path to fine-tuned CV VAE (or '' to use pretrained SD-VAE)")
    p.add_argument("--out", default="cv_models/ldm_mr2ct")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cfg_drop_p", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--sample_guidance", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--resume", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
