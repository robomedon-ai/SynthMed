"""Train LDM++ (adversarial latent diffusion) for T2 → ADC on prostate158.

Warm-starts from the vanilla prostate LDM (so we refine, not retrain from
scratch) and adds a conditional PatchGAN discriminator + L1 + LPIPS pixel
supervision on the decoded x̂₀ at low-noise timesteps.

Usage:
    python train_prostate_ldm_plus.py \
        --vae_ckpt prostate_models/vae/latest.pt \
        --init_ldm prostate_models/ldm_t2_adc/latest.pt \
        --steps 15000 --batch 12 \
        --out prostate_models/ldm_plus_t2_adc
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
import prostate_ldm_plus as ldmp
import pasd_pix2pix as p2p


def _denorm_uint8(x: torch.Tensor) -> np.ndarray:
    return x.clamp(-1, 1).add(1).div(2).mul(255).round().permute(
        0, 2, 3, 1).byte().cpu().numpy()


def save_grid(t2s, adcs, fakes, path: Path):
    t2_np, adc_np, fk_np = map(_denorm_uint8, (t2s, adcs, fakes))
    n, H, W, _ = adc_np.shape
    grid = np.zeros((n * H, 3 * W, 3), dtype=np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H,      0:W]   = t2_np[i]
        grid[i*H:(i+1)*H,      W:2*W] = adc_np[i]
        grid[i*H:(i+1)*H,    2*W:3*W] = fk_np[i]
    Image.fromarray(grid).save(path)


def load_vae(vae_ckpt: str, device: str):
    vae = vae_mod.load_vae(freeze_encoder=True)
    if vae_ckpt:
        sd = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
        print(f"[vae] loaded fine-tuned decoder from {vae_ckpt}")
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False
    return vae.to(device)


def get_loader(split, image_size, batch, workers):
    base = prostate_data.get_paired_2d_dataset(
        split, input_modality="t2", target_modality="adc",
        augment=(split == "train"), image_size=image_size)

    class _P3(torch.utils.data.Dataset):
        def __len__(self): return len(base)
        def __getitem__(self, i):
            it = base[i]
            return {"t2": it["input"].repeat(3, 1, 1),
                    "adc": it["target"].repeat(3, 1, 1)}

    return DataLoader(_P3(), batch_size=batch, shuffle=(split == "train"),
                      num_workers=workers, pin_memory=True,
                      drop_last=(split == "train"),
                      persistent_workers=workers > 0), len(base)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    loader, n_train = get_loader("train", args.image_size, args.batch, args.workers)
    val_loader, _ = get_loader("val", args.image_size, min(args.batch, 4), args.workers)
    print(f"[data] train slices={n_train}  batch={args.batch}")

    fixed = next(iter(val_loader))
    fixed_t2 = fixed["t2"].to(device); fixed_adc = fixed["adc"].to(device)

    vae = load_vae(args.vae_ckpt, device)
    unet = ldmp.build_unet().to(device)
    if args.init_ldm:
        sd = torch.load(args.init_ldm, map_location="cpu", weights_only=False)
        unet.load_state_dict(sd["unet"])
        print(f"[unet] warm-started from {args.init_ldm} (step {sd.get('step','?')})")
    print(f"[unet] params: {sum(p.numel() for p in unet.parameters())/1e6:.1f}M")

    # Conditional PatchGAN: sees (T2 3ch, ADC 3ch) = 6 channels
    disc = p2p.PatchDiscriminator(in_ch=6, ndf=64).to(device)
    p2p.init_weights(disc)
    print(f"[disc] params: {sum(p.numel() for p in disc.parameters())/1e6:.1f}M")

    train_sched = ldmp.build_train_scheduler()
    gan_loss = p2p.GANLoss()
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    for p in lpips_fn.parameters(): p.requires_grad = False

    opt_G = torch.optim.AdamW(unet.parameters(), lr=args.lr,
                              betas=(0.9, 0.999), weight_decay=1e-4)
    opt_D = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.steps)
    scaler_G = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler_D = torch.amp.GradScaler("cuda", enabled=args.amp)
    ema = ldmp.EMAModel(unet, decay=args.ema_decay)

    history = []
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        unet.load_state_dict(ck["unet"]); disc.load_state_dict(ck["disc"])
        if "ema" in ck: ema.load_state_dict(ck["ema"], device=device)
        step = int(ck.get("step", 0))
        for _ in range(step): sched_G.step()
        print(f"[resume] from {args.resume} at step {step}")

    log = {"mse": 0, "pix": 0, "perc": 0, "adv": 0,
           "d_real": 0, "d_fake": 0, "n": 0, "nlow": 0}
    t_log = time.time()
    unet.train(); disc.train()
    data_iter = iter(loader)
    best_balance = -1.0
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch = next(data_iter)
        t2  = batch["t2"].to(device, non_blocking=True)
        adc = batch["adc"].to(device, non_blocking=True)

        # ── G step ──
        opt_G.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp):
            loss_G, parts, fake_for_D = ldmp.training_step_plus(
                unet, vae, disc, t2, adc, train_sched, gan_loss, lpips_fn,
                cfg_drop_p=args.cfg_drop_p, pix_t_threshold=args.pix_t_threshold,
                lambda_pix=args.lambda_pix, lambda_perc=args.lambda_perc,
                lambda_adv=args.lambda_adv)
        scaler_G.scale(loss_G).backward()
        scaler_G.unscale_(opt_G)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        scaler_G.step(opt_G); scaler_G.update(); sched_G.step()
        ema.update(unet)

        # ── D step (only when G produced decoded fakes) ──
        d_parts = {"d_real": 0.0, "d_fake": 0.0}
        if fake_for_D is not None:
            opt_D.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                loss_D, d_parts = ldmp.discriminator_step(disc, gan_loss, fake_for_D)
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D); scaler_D.update()

        bs = t2.size(0)
        log["mse"]  += parts["mse"] * bs
        log["pix"]  += parts.get("pix", 0) * bs
        log["perc"] += parts.get("perc", 0) * bs
        log["adv"]  += parts.get("adv", 0) * bs
        log["d_real"] += d_parts["d_real"] * bs
        log["d_fake"] += d_parts["d_fake"] * bs
        log["nlow"] += parts.get("n_low", 0)
        log["n"]    += bs
        step += 1

        if step % args.log_every == 0:
            n = log["n"]; dt = time.time() - t_log
            print(f"[step {step:6d}/{args.steps}] mse={log['mse']/n:.4f} "
                  f"pix={log['pix']/n:.4f} perc={log['perc']/n:.4f} "
                  f"adv={log['adv']/n:.3f} D(r={log['d_real']/n:.3f} "
                  f"f={log['d_fake']/n:.3f}) nlow={log['nlow']} "
                  f"({n/dt:.1f}/s)", flush=True)
            log = {k: 0 for k in log}; t_log = time.time()

        if step % args.sample_every == 0 or step == args.steps:
            ema.apply_to(unet); unet.eval()
            with torch.no_grad():
                fake = ldmp.sample(unet, vae, fixed_t2,
                                   num_inference_steps=args.sample_steps,
                                   guidance_scale=args.sample_guidance,
                                   device=str(device))
            save_grid(fixed_t2, fixed_adc, fake,
                      out_dir / "samples" / f"step_{step:06d}.png")
            ema.restore(unet); unet.train()

        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save({"unet": unet.state_dict(), "disc": disc.state_dict(),
                        "ema": ema.state_dict(), "step": step, "args": vars(args)},
                       out_dir / f"ldm_step{step:06d}.pt")
            torch.save({"unet": unet.state_dict(), "disc": disc.state_dict(),
                        "ema": ema.state_dict(), "step": step, "args": vars(args)},
                       out_dir / "latest.pt")
            history.append({"step": step, "lr": opt_G.param_groups[0]["lr"]})
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"[done] checkpoints in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--init_ldm", default="",
                   help="Warm-start UNet from a trained vanilla LDM checkpoint.")
    p.add_argument("--out", default="prostate_models/ldm_plus_t2_adc")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-5)        # lower than scratch LDM (1e-4) since warm-started
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--cfg_drop_p", type=float, default=0.1)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--pix_t_threshold", type=int, default=250)
    p.add_argument("--lambda_pix", type=float, default=1.0)
    p.add_argument("--lambda_perc", type=float, default=1.0)
    p.add_argument("--lambda_adv", type=float, default=0.5)
    p.add_argument("--sample_steps", type=int, default=25)
    p.add_argument("--sample_guidance", type=float, default=1.5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--resume", default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
