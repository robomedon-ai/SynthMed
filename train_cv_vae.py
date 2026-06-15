"""Stage 1 of the CV LDM: fine-tune SD-VAE decoder on MR + CT slices.

Trains a SINGLE VAE on the UNION of MR and CT slices, giving the LDM a
shared latent space. The model can encode either modality and decode back;
we use it for both the conditioning side (encode real MR) and the output
side (decode generated CT latent).

Usage:
    python train_cv_vae.py --steps 8000 --batch 8 --out cv_models/vae

Exit criterion: PSNR ≥ 30 dB on val (matches PASD VAE target).
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import cv_data
import cv_vae as vae_mod


# ── Dataset that yields a single-modality slice as 3-channel [-1,1] ──

class _SingleModalitySlices(Dataset):
    """Wraps cv_data.get_paired_2d_dataset and emits ONE modality per item.

    SD-VAE expects 3-channel input, so we replicate grayscale → RGB.
    """
    def __init__(self, split: str, modality: str, image_size: int = 256,
                 augment: bool = True):
        self.base = cv_data.get_paired_2d_dataset(
            split, augment=augment, image_size=image_size)
        self.modality = modality  # "mr" or "ct"

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        t = item[self.modality]            # (1, H, W) in [-1, 1]
        return {"image": t.repeat(3, 1, 1)}  # (3, H, W) in [-1, 1]


def _denorm_uint8(x: torch.Tensor) -> np.ndarray:
    return x.clamp(-1, 1).add(1).div(2).mul(255).round().permute(
        0, 2, 3, 1).byte().cpu().numpy()


def save_recon_grid(originals: torch.Tensor, recons: torch.Tensor, path: Path):
    o = _denorm_uint8(originals)
    r = _denorm_uint8(recons)
    n, H, W, _ = o.shape
    grid = np.zeros((n * H, 2 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,  0:W ] = o[i]
        grid[i*H:(i+1)*H,  W:2*W] = r[i]
    Image.fromarray(grid).save(path)


@torch.no_grad()
def evaluate(vae, loaders, recon_fn, device, max_batches: int = 4):
    """Compute mean L1 / LPIPS / PSNR on N val batches from EACH loader."""
    vae.eval()
    out_total = {"l1": 0.0, "lpips": 0.0, "psnr": 0.0, "n": 0}
    for loader in loaders:
        for batch in islice(loader, max_batches):
            x = batch["image"].to(device)
            recon, _ = vae_mod.vae_forward(vae, x, sample_posterior=False)
            _, parts = recon_fn(recon, x)
            mse = F.mse_loss((recon + 1) / 2, (x + 1) / 2).item()
            psnr = 10 * math.log10(1.0 / max(mse, 1e-10))
            n = x.size(0)
            out_total["l1"]    += parts["l1"]    * n
            out_total["lpips"] += parts["lpips"] * n
            out_total["psnr"]  += psnr           * n
            out_total["n"]     += n
    vae.train()
    n = max(out_total["n"], 1)
    return {k: out_total[k] / n if k != "n" else out_total["n"] for k in out_total}


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Train on UNION of MR + CT slices
    mr_train = _SingleModalitySlices("train", "mr", args.image_size, augment=True)
    ct_train = _SingleModalitySlices("train", "ct", args.image_size, augment=True)
    train_ds = ConcatDataset([mr_train, ct_train])
    mr_val   = _SingleModalitySlices("val",   "mr", args.image_size, augment=False)
    ct_val   = _SingleModalitySlices("val",   "ct", args.image_size, augment=False)
    print(f"[data] train={len(train_ds)} ({len(mr_train)} MR + {len(ct_train)} CT)  "
          f"val MR={len(mr_val)} CT={len(ct_val)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader_mr = DataLoader(mr_val, batch_size=args.batch, shuffle=False,
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=args.workers > 0)
    val_loader_ct = DataLoader(ct_val, batch_size=args.batch, shuffle=False,
                               num_workers=args.workers, pin_memory=True,
                               persistent_workers=args.workers > 0)

    fixed_mr = next(iter(DataLoader(mr_val, batch_size=min(4, len(mr_val)),
                                    shuffle=False)))["image"].to(device)
    fixed_ct = next(iter(DataLoader(ct_val, batch_size=min(4, len(ct_val)),
                                    shuffle=False)))["image"].to(device)
    fixed_all = torch.cat([fixed_mr, fixed_ct], dim=0)

    print(f"[vae] loading pretrained {vae_mod.VAE_HF_ID}")
    vae = vae_mod.load_vae(freeze_encoder=True).to(device)
    n_t = sum(p.numel() for p in vae_mod.trainable_parameters(vae))
    n_a = sum(p.numel() for p in vae.parameters())
    print(f"[vae] trainable {n_t/1e6:.1f}M / total {n_a/1e6:.1f}M")

    recon_loss = vae_mod.VAEReconLoss(lpips_weight=args.lpips_weight).to(device)

    opt = torch.optim.AdamW(vae_mod.trainable_parameters(vae),
                            lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    history = []
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        vae.decoder.load_state_dict(ck["decoder"])
        vae.post_quant_conv.load_state_dict(ck["post_quant_conv"])
        step = int(ck.get("step", 0))
        for _ in range(step): sched.step()
        print(f"[resume] from {args.resume} at step {step}")

    log_buf = {"l1": 0.0, "lpips": 0.0, "n": 0}
    t0 = time.time()
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
        scaler.step(opt); scaler.update(); sched.step()

        log_buf["l1"]    += parts["l1"]    * x.size(0)
        log_buf["lpips"] += parts["lpips"] * x.size(0)
        log_buf["n"]     += x.size(0)
        step += 1

        if step % args.log_every == 0:
            n = log_buf["n"]; dt = time.time() - t0
            print(f"[step {step:5d}/{args.steps}] "
                  f"l1={log_buf['l1']/n:.4f} lpips={log_buf['lpips']/n:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} "
                  f"({n/dt:.1f} samples/s)", flush=True)
            log_buf = {"l1": 0.0, "lpips": 0.0, "n": 0}
            t0 = time.time()

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(vae, [val_loader_mr, val_loader_ct], recon_loss,
                         device, max_batches=args.eval_batches)
            print(f"  [val] l1={m['l1']:.4f} lpips={m['lpips']:.4f} "
                  f"PSNR={m['psnr']:.2f}dB", flush=True)
            history.append({"step": step, **m})
            with torch.no_grad():
                rec, _ = vae_mod.vae_forward(vae, fixed_all,
                                             sample_posterior=False)
            save_recon_grid(fixed_all, rec,
                            out_dir / "samples" / f"step_{step:05d}.png")
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save({"decoder": vae.decoder.state_dict(),
                        "post_quant_conv": vae.post_quant_conv.state_dict(),
                        "step": step, "args": vars(args)},
                       out_dir / f"vae_step{step:05d}.pt")
            torch.save({"decoder": vae.decoder.state_dict(),
                        "post_quant_conv": vae.post_quant_conv.state_dict(),
                        "step": step, "args": vars(args)},
                       out_dir / "latest.pt")

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="cv_models/vae")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lpips_weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=4)
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--resume", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
