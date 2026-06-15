"""Train the combined prostate LDM (T2 + anatomy mask → ADC).

Mirrors train_prostate_ldm.py with mask loading. Warm-starts the UNet's shared
weights is NOT possible (in_channels differ: 8 vs 11), so trains from scratch.

Usage:
    python train_prostate_ldm_combined.py \
        --vae_ckpt prostate_models/vae/latest.pt \
        --steps 30000 --batch 16 \
        --out prostate_models/ldm_combined_t2_adc
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import prostate_data
import prostate_vae as vae_mod
import prostate_ldm_combined as ldm


def _denorm_uint8(x):
    return x.clamp(-1,1).add(1).div(2).mul(255).round().permute(0,2,3,1).byte().cpu().numpy()


def save_grid(t2s, adcs, fakes, path):
    a, b, f = _denorm_uint8(t2s), _denorm_uint8(adcs), _denorm_uint8(fakes)
    n, H, W, _ = b.shape
    grid = np.zeros((n*H, 3*W, 3), np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H, 0:W] = a[i]
        grid[i*H:(i+1)*H, W:2*W] = b[i]
        grid[i*H:(i+1)*H, 2*W:3*W] = f[i]
    Image.fromarray(grid).save(path)


def load_vae(vae_ckpt, device):
    vae = vae_mod.load_vae(freeze_encoder=True)
    if vae_ckpt:
        sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
        print(f"[vae] loaded decoder from {vae_ckpt}")
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    return vae.to(device)


def get_loader(split, image_size, batch, workers):
    base = prostate_data.get_paired_2d_dataset(
        split, input_modality="t2", target_modality="adc",
        augment=(split == "train"), include_mask=True, image_size=image_size)

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(base)
        def __getitem__(self, i):
            it = base[i]
            return {"t2": it["input"].repeat(3,1,1),
                    "adc": it["target"].repeat(3,1,1),
                    "mask": it["mask"]}   # (H,W) long

    return DataLoader(_DS(), batch_size=batch, shuffle=(split=="train"),
                      num_workers=workers, pin_memory=True,
                      drop_last=(split=="train"), persistent_workers=workers>0), len(base)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/"samples").mkdir(exist_ok=True)
    json.dump(vars(args), open(out/"config.json","w"), indent=2)

    loader, n = get_loader("train", args.image_size, args.batch, args.workers)
    vloader, _ = get_loader("val", args.image_size, min(args.batch,4), args.workers)
    print(f"[data] train slices={n} batch={args.batch}")
    fv = next(iter(vloader))
    fixed_t2 = fv["t2"].to(device); fixed_adc = fv["adc"].to(device)
    fixed_mask = fv["mask"].to(device)

    vae = load_vae(args.vae_ckpt, device)
    unet = ldm.build_unet().to(device)
    print(f"[unet] params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M (in_ch={ldm.IN_CHANNELS})")
    tsched = ldm.build_train_scheduler()

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = ldm.EMAModel(unet, decay=args.ema_decay)

    hist=[]; step=0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ck["unet"])
        if "ema" in ck: ema.load_state_dict(ck["ema"], device=device)
        step = int(ck.get("step",0))
        for _ in range(step): sch.step()
        print(f"[resume] step {step}")

    log={"mse":0,"n":0}; t0=time.time(); unet.train()
    it = iter(loader)
    while step < args.steps:
        try: b = next(it)
        except StopIteration: it = iter(loader); b = next(it)
        t2 = b["t2"].to(device, non_blocking=True)
        adc = b["adc"].to(device, non_blocking=True)
        mask = b["mask"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss, parts = ldm.training_step(unet, vae, t2, adc, mask, tsched,
                                            cfg_drop_p=args.cfg_drop_p)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        ema.update(unet)
        log["mse"]+=parts["mse"]*t2.size(0); log["n"]+=t2.size(0); step+=1

        if step % args.log_every == 0:
            nn=log["n"]; dt=time.time()-t0
            print(f"[step {step:6d}/{args.steps}] mse={log['mse']/nn:.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} ({nn/dt:.1f}/s)", flush=True)
            log={"mse":0,"n":0}; t0=time.time()

        if step % args.sample_every == 0 or step == args.steps:
            ema.apply_to(unet); unet.eval()
            with torch.no_grad():
                fake = ldm.sample(unet, vae, fixed_t2, fixed_mask,
                                  num_inference_steps=args.sample_steps,
                                  guidance_scale=args.sample_guidance, device=str(device))
            save_grid(fixed_t2, fixed_adc, fake, out/"samples"/f"step_{step:06d}.png")
            ema.restore(unet); unet.train()

        if step % args.checkpoint_every == 0 or step == args.steps:
            for name in (f"ldm_step{step:06d}.pt", "latest.pt"):
                torch.save({"unet":unet.state_dict(),"ema":ema.state_dict(),
                            "step":step,"args":vars(args)}, out/name)
            hist.append({"step":step,"lr":opt.param_groups[0]["lr"]})
            json.dump(hist, open(out/"history.json","w"), indent=2)

    print(f"[done] {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--out", default="prostate_models/ldm_combined_t2_adc")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cfg_drop_p", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--sample_guidance", type=float, default=1.0)
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
