"""Train a paired CUT (contrastive GAN) for T2 → ADC on prostate158.

Losses:
  - GAN (conditional PatchGAN, sees T2+ADC) — realism
  - PatchNCE (input↔output feature correspondence) — content + EPI-distortion robustness
  - identity PatchNCE (real_B↔G(real_B)) — color/structure preservation
  - modest L1(fake_B, real_B) — exploits pairing without dominating into mean-blur

Usage:
    python train_prostate_cut.py --epochs 80 --batch 4 \
        --out prostate_models/cut_t2_adc
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

import prostate_data
import prostate_cut as cut
import pasd_pix2pix as p2p   # reuse PatchDiscriminator + GANLoss


class _Paired2D(Dataset):
    def __init__(self, split, image_size=256, augment=True):
        self.base = prostate_data.get_paired_2d_dataset(
            split, input_modality="t2", target_modality="adc",
            augment=augment, image_size=image_size)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        it = self.base[i]
        return {"A": it["input"], "B": it["target"]}   # (1,H,W) in [-1,1]


def _to_uint8(x): return (x.clamp(-1,1).add(1).div(2).mul(255).round()
                          .squeeze(1).byte().cpu().numpy())


def save_grid(A, fake, B, path):
    a, f, b = _to_uint8(A), _to_uint8(fake), _to_uint8(B)
    n, H, W = a.shape
    grid = np.zeros((n*H, 3*W), np.uint8)
    for i in range(n):
        grid[i*H:(i+1)*H, 0:W] = a[i]
        grid[i*H:(i+1)*H, W:2*W] = f[i]
        grid[i*H:(i+1)*H, 2*W:3*W] = b[i]
    Image.fromarray(grid, "L").save(path)


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    tr = _Paired2D("train", args.image_size, augment=True)
    va = _Paired2D("val", args.image_size, augment=False)
    print(f"[data] train={len(tr)} val={len(va)}")
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers>0)
    vb = next(iter(DataLoader(va, batch_size=min(4,len(va)), shuffle=False)))
    fixed_A = vb["A"].to(device); fixed_B = vb["B"].to(device)

    nce_layers = cut.DEFAULT_NCE_LAYERS
    G = cut.ResnetGenerator(1, 1, ngf=args.ngf, n_blocks=9).to(device)
    D = p2p.PatchDiscriminator(in_ch=2, ndf=64).to(device)   # conditional: T2+ADC
    F_net = cut.PatchSampleF(use_mlp=True, nc=256).to(device)
    cut.init_weights(G); p2p.init_weights(D)

    gan = p2p.GANLoss()
    l1 = nn.L1Loss()
    nce_losses = [cut.PatchNCELoss(nce_T=args.nce_T).to(device) for _ in nce_layers]

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))
    # F's MLPs are built lazily on first forward; its optimizer is created then.
    opt_F = None

    def lr_lambda(ep):
        warm = args.epochs // 2
        return 1.0 if ep < warm else max(0.0, 1 - (ep-warm)/max(1,args.epochs-warm))
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    def compute_nce(real, fake):
        feat_k = G(real, layers=nce_layers, encode_only=True)
        feat_q = G(fake, layers=nce_layers, encode_only=True)
        k_pool, ids = F_net(feat_k, args.num_patches, None)
        q_pool, _   = F_net(feat_q, args.num_patches, ids)
        total = 0.0
        for fq, fk, crit in zip(q_pool, k_pool, nce_losses):
            crit.num_patches = fq.shape[0] // args.batch
            total = total + crit(fq, fk)
        return total / len(nce_layers)

    history = []
    start_ep = 1
    best_balance = -1.0
    for epoch in range(start_ep, args.epochs+1):
        G.train(); D.train()
        t0 = time.time()
        s = {"gan":0,"nce":0,"l1":0,"d":0,"n":0}
        for batch in loader:
            A = batch["A"].to(device, non_blocking=True)
            B = batch["B"].to(device, non_blocking=True)

            fake = G(A)

            # lazily create F optimizer after MLPs exist
            if opt_F is None:
                _ = compute_nce(A, fake)   # triggers MLP build
                if len(list(F_net.parameters())) > 0:
                    opt_F = torch.optim.Adam(F_net.parameters(), lr=args.lr, betas=(0.5,0.999))

            # ── D step ──
            opt_D.zero_grad(set_to_none=True)
            d_real = D(torch.cat([A, B], 1))
            d_fake = D(torch.cat([A, fake.detach()], 1))
            loss_D = 0.5*(gan(d_real, True) + gan(d_fake, False))
            loss_D.backward()
            opt_D.step()

            # ── G + F step ──
            opt_G.zero_grad(set_to_none=True)
            if opt_F is not None: opt_F.zero_grad(set_to_none=True)
            d_for_g = D(torch.cat([A, fake], 1))
            l_gan = gan(d_for_g, True) * args.lambda_gan
            l_nce = compute_nce(A, fake) * args.lambda_nce
            if args.nce_idt:
                idt_B = G(B)
                l_nce = 0.5*(l_nce + compute_nce(B, idt_B) * args.lambda_nce)
            l_l1 = l1(fake, B) * args.lambda_l1
            loss_G = l_gan + l_nce + l_l1
            loss_G.backward()
            opt_G.step()
            if opt_F is not None: opt_F.step()

            n = A.size(0)
            s["gan"]+=l_gan.item()*n; s["nce"]+=l_nce.item()*n
            s["l1"]+=l_l1.item()*n; s["d"]+=loss_D.item()*n; s["n"]+=n
        sch_G.step(); sch_D.step()
        n=s["n"]
        stats={"epoch":epoch,"gan":s["gan"]/n,"nce":s["nce"]/n,"l1":s["l1"]/n,
               "d":s["d"]/n,"lr":opt_G.param_groups[0]["lr"],"t":time.time()-t0}
        history.append(stats)
        print(f"[ep {epoch:3d}/{args.epochs}] G(gan={stats['gan']:.3f} "
              f"nce={stats['nce']:.3f} l1={stats['l1']:.3f}) D={stats['d']:.3f} "
              f"lr={stats['lr']:.1e} t={stats['t']:.0f}s", flush=True)

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            G.eval()
            with torch.no_grad(): fv = G(fixed_A)
            save_grid(fixed_A, fv, fixed_B, out/"samples"/f"epoch_{epoch:03d}.png")
            G.train()
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            torch.save({"G":G.state_dict(),"epoch":epoch,"args":vars(args)},
                       out/"latest_G.pt")
            # best-balance guardrail (D near 0.25 = healthy for BCE 0.5-ish split)
            bal = 1.0 - abs(stats["d"] - 0.5)
            if bal > best_balance and stats["d"] > 0.15:
                best_balance = bal
                torch.save({"G":G.state_dict(),"epoch":epoch,"args":vars(args)},
                           out/"best_G.pt")
                print(f"  -> best_G @ ep{epoch} (D={stats['d']:.3f})", flush=True)
            json.dump(history, open(out/"history.json","w"), indent=2)
    # ensure best exists
    if not (out/"best_G.pt").exists():
        torch.save({"G":G.state_dict(),"epoch":args.epochs,"args":vars(args)},
                   out/"best_G.pt")
    print(f"[done] {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="prostate_models/cut_t2_adc")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_gan", type=float, default=1.0)
    p.add_argument("--lambda_nce", type=float, default=1.0)
    p.add_argument("--lambda_l1", type=float, default=2.0)
    p.add_argument("--nce_idt", action="store_true", default=True)
    p.add_argument("--nce_T", type=float, default=0.07)
    p.add_argument("--num_patches", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--checkpoint_every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
