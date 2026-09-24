"""Cross-dataset TSTR (Train-on-Synthetic, Test-on-Real) utility study.

The paper's "realism → relevance" pillar: does a generator's IMAGE QUALITY
(FID/SSIM) predict its DOWNSTREAM UTILITY? We segment the TARGET modality and
compare segmenters trained on real vs each generator's synthetic images.

For an image→image dataset (prostate T2→ADC, TopBrain MRA→CTA):
  - downstream task = binary segmentation of the target modality
    (prostate: whole-gland from ADC; topbrain: vessels from CTA)
  - real condition   : train on real (target, mask)
  - synth_<gen>       : train on (generated target, real mask)
  - real+synth_<gen>  : augmentation (union)
  - ALL conditions tested on the held-out REAL test set → Dice/IoU

Subcommands:
  prepare --dataset prostate --gen pix2pix|ldm|cyclegan|combined
  train   --dataset prostate --mode real|synth_pix2pix|real_synth_ldm|...
  eval    --dataset prostate
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import pasd_seg as seg
import generation as gen

ROOT = Path(__file__).parent
SYNTH_DIR = ROOT / "tstr_synth_data"
SEG_DIR   = ROOT / "tstr_synth_models"


# ── dataset registry ──

def _prostate_loader(split):
    import prostate_data
    return prostate_data.get_paired_2d_dataset(
        split, input_modality="t2", target_modality="adc",
        augment=False, include_mask=True, image_size=256)

def _topbrain_loader(split):
    import cv_data
    return cv_data.get_paired_2d_dataset(split, augment=False,
                                         include_masks=True, image_size=256)

def _cond_pil_prostate(item):
    arr = ((item["input"][0] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr, "L")

def _target_arr_prostate(item):
    return ((item["target"][0] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()

def _mask_arr_prostate(item):
    return (item["mask"].cpu().numpy() > 0).astype(np.uint8) * 255

def _cond_pil_topbrain(item):
    arr = ((item["mr"][0] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr, "L")

def _target_arr_topbrain(item):
    return ((item["ct"][0] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()

def _mask_arr_topbrain(item):
    return (item["mask_ct"].cpu().numpy() > 0).astype(np.uint8) * 255


def _gen_prostate(name, cond_pil, item, device, seed):
    if name == "pix2pix":
        return gen.prostate_translate_t2_to_adc(cond_pil, device=device, watermark=False)
    if name == "ldm":
        return gen.prostate_translate_t2_to_adc_ldm(
            cond_pil, num_inference_steps=25, guidance_scale=1.0,
            seed=seed, device=device, watermark=False)
    if name == "cyclegan":
        return gen.prostate_translate_cyclegan(cond_pil, direction="T2_to_ADC",
                                               device=device, watermark=False)
    if name == "combined":
        mask_t = item["mask"].long()
        return gen.prostate_translate_t2_mask_to_adc_ldm(
            cond_pil, mask_t, num_inference_steps=25, guidance_scale=1.0,
            seed=seed, device=device, watermark=False)
    raise ValueError(name)


def _gen_topbrain(name, cond_pil, item, device, seed):
    if name == "pix2pix":
        return gen.cv_translate_mr2ct(cond_pil, device=device, watermark=False)
    if name == "ldm":
        return gen.cv_translate_mr2ct_ldm(cond_pil, num_samples=1,
            num_inference_steps=25, guidance_scale=1.5, seed=seed,
            device=device, watermark=False)[0]
    if name == "cyclegan":
        return gen.cv_translate_cyclegan(cond_pil, direction="MR_to_CT",
                                         device=device, watermark=False)
    if name == "combined":
        mask_t = item["mask_ct"].long()
        return gen.cv_translate_mr_mask_to_ct_ldm(
            cond_pil, mask_t, num_inference_steps=25, guidance_scale=1.5,
            seed=seed, device=device, watermark=False)
    raise ValueError(name)


# ── PASD (mask → MRI; placenta segmentation). Different task structure:
#    the generator INPUT is the mask, the target is the real BTFE image,
#    and the seg label is the same placenta mask. Comparable generators are
#    the mask-conditional ones: pix2pix, ldm, spade. ──

def _pasd_loader(split):
    import pasd_data
    return pasd_data.get_pasd_dataset("BTFE", split=split, augment=False,
                                      image_size=256)

def _cond_pil_pasd(item):
    # generator input = the placenta mask
    m = (item["mask"][0] > 0).cpu().numpy().astype(np.uint8) * 255
    return Image.fromarray(m, "L")

def _target_arr_pasd(item):
    # real BTFE image (the modality the segmenter operates on)
    return ((item["image"][0] + 1) / 2 * 255).clamp(0, 255).byte().cpu().numpy()

def _mask_arr_pasd(item):
    return (item["mask"][0] > 0).cpu().numpy().astype(np.uint8) * 255

def _gen_pasd(name, cond_pil, item, device, seed):
    # cond_pil is the mask; all PASD generators are mask→BTFE
    if name in ("pix2pix", "ldm", "spade"):
        return gen.sample(cond_pil, modality="BTFE", model=name, num_samples=1,
                          seed=seed, device=device, watermark=False)[0]
    raise ValueError(name)


DATASETS = {
    "prostate": dict(loader=_prostate_loader, cond_pil=_cond_pil_prostate,
                     target_arr=_target_arr_prostate, mask_arr=_mask_arr_prostate,
                     generate=_gen_prostate,
                     gens=("pix2pix", "ldm", "cyclegan", "combined")),
    "topbrain": dict(loader=_topbrain_loader, cond_pil=_cond_pil_topbrain,
                     target_arr=_target_arr_topbrain, mask_arr=_mask_arr_topbrain,
                     generate=_gen_topbrain,
                     gens=("pix2pix", "ldm", "cyclegan", "combined")),
    "pasd":     dict(loader=_pasd_loader, cond_pil=_cond_pil_pasd,
                     target_arr=_target_arr_pasd, mask_arr=_mask_arr_pasd,
                     generate=_gen_pasd,
                     gens=("pix2pix", "ldm", "spade")),
}


# ── disk dataset for a prepared condition ──

class FolderSeg(Dataset):
    """Loads (target_image, mask) jpg/png pairs from a folder; returns
    in_ch=1 image in [-1,1] and mask in {0,1}."""
    def __init__(self, root, augment=False):
        self.imgs = sorted((Path(root) / "images").glob("*.jpg"))
        self.root = Path(root); self.augment = augment
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        ip = self.imgs[i]
        mp = self.root / "masks" / (ip.stem + ".png")
        img = np.asarray(Image.open(ip).convert("L"), np.float32) / 255.0
        msk = (np.asarray(Image.open(mp).convert("L"), np.float32) > 127).astype(np.float32)
        if self.augment and np.random.rand() < 0.5:
            img = img[:, ::-1].copy(); msk = msk[:, ::-1].copy()
        x = torch.from_numpy(img).unsqueeze(0) * 2 - 1     # (1,H,W) [-1,1]
        y = torch.from_numpy(msk).unsqueeze(0)             # (1,H,W) {0,1}
        return {"image": x, "mask": y}


# ───────── prepare ─────────

@torch.no_grad()
def prepare(args):
    cfg = DATASETS[args.dataset]
    device = args.device
    # real condition (once)
    if args.gen == "real" or args.also_real:
        _prepare_real(args.dataset, cfg)
        if args.gen == "real":
            return
    out = SYNTH_DIR / f"{args.dataset}_{args.gen}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    ds = cfg["loader"]("train")
    print(f"[prep] {args.dataset}/{args.gen}: {len(ds)} slices -> {out}", flush=True)
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        cond = cfg["cond_pil"](item)
        synth = cfg["generate"](args.gen, cond, item, device, i)
        stem = f"idx{i:05d}"
        synth.convert("L").save(out / "images" / f"{stem}.jpg", quality=92)
        Image.fromarray(cfg["mask_arr"](item), "L").save(out / "masks" / f"{stem}.png")
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(ds)}] {(i+1)/(time.time()-t0):.1f}/s", flush=True)
    print(f"[prep] done {time.time()-t0:.0f}s")


def _prepare_real(dataset, cfg):
    for split in ("train", "val", "test"):
        out = SYNTH_DIR / f"{dataset}_real_{split}"
        if (out / "images").exists() and list((out / "images").glob("*.jpg")):
            continue
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        ds = cfg["loader"](split)
        print(f"[prep-real] {dataset}/{split}: {len(ds)} slices", flush=True)
        for i in range(len(ds)):
            item = ds[i]
            stem = f"idx{i:05d}"
            Image.fromarray(cfg["target_arr"](item), "L").save(out/"images"/f"{stem}.jpg", quality=92)
            Image.fromarray(cfg["mask_arr"](item), "L").save(out/"masks"/f"{stem}.png")


# ───────── train ─────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mode = args.mode
    seed = int(getattr(args, "seed", 0))
    # Seed all RNGs so multi-seed runs are reproducible and capture training variance
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    suffix = f"_s{seed}" if getattr(args, "seeded_dir", False) else ""
    arch = getattr(args, "arch", "unet")
    archsuf = "" if arch == "unet" else f"_{arch}"
    out_dir = SEG_DIR / f"{args.dataset}_{mode}{archsuf}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # assemble training set from mode
    parts = []
    if mode == "real" or mode.startswith("real_synth_"):
        parts.append(FolderSeg(SYNTH_DIR / f"{args.dataset}_real_train", augment=True))
    if mode.startswith("synth_"):
        parts.append(FolderSeg(SYNTH_DIR / f"{args.dataset}_{mode[6:]}", augment=True))
    if mode.startswith("real_synth_"):
        parts.append(FolderSeg(SYNTH_DIR / f"{args.dataset}_{mode[11:]}", augment=True))
    if not parts:
        raise ValueError(f"bad mode {mode}")
    train_ds = ConcatDataset(parts) if len(parts) > 1 else parts[0]
    # VAL-BASED SELECTION: choose the checkpoint on a real validation split,
    # then score it once on the untouched real test split.
    val_ds = FolderSeg(SYNTH_DIR / f"{args.dataset}_real_val", augment=False)
    test_ds = FolderSeg(SYNTH_DIR / f"{args.dataset}_real_test", augment=False)
    print(f"[train] {args.dataset}/{mode}: train={len(train_ds)} "
          f"val(real val)={len(val_ds)} test(real test)={len(test_ds)}")

    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    testl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    if arch == "resunet":
        model = seg.ResUNet(in_ch=1, out_ch=1, base=args.base).to(device)
    else:
        model = seg.SmallUNet(in_ch=1, out_ch=1, base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = seg.DiceBCELoss()

    best = -1.0
    best_ep = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for b in tl:
            x = b["image"].to(device); y = b["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        sch.step()
        # val
        model.eval(); d = n = 0.0
        with torch.no_grad():
            for b in vl:
                x = b["image"].to(device); y = b["mask"].to(device)
                d += float(seg.dice_score(model(x), y)) * x.size(0); n += x.size(0)
        d /= max(n, 1)
        if d > best:
            best = d
            best_ep = ep
            torch.save({"model": model.state_dict(), "val_dice": d, "ep": ep,
                        "args": vars(args)}, out_dir / "best.pt")
        print(f"[ep {ep:2d}/{args.epochs}] real-val Dice={d:.4f} (best {best:.4f})", flush=True)

    # single test evaluation of the val-selected checkpoint
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval(); td = tn = 0.0
    with torch.no_grad():
        for b in testl:
            x = b["image"].to(device); y = b["mask"].to(device)
            td += float(seg.dice_score(model(x), y)) * x.size(0); tn += x.size(0)
    td /= max(tn, 1)
    ck["test_dice"] = td
    torch.save(ck, out_dir / "best.pt")
    print(f"[done] {args.dataset}/{mode} val={best:.4f} (ep {best_ep}) TEST={td:.4f}")


# ───────── eval ─────────

def evaluate(args):
    rows = []
    for sub in sorted((SEG_DIR).glob(f"{args.dataset}_*")):
        ck = sub / "best.pt"
        if not ck.exists(): continue
        d = torch.load(ck, map_location="cpu", weights_only=False)
        if "test_dice" not in d:
            print(f"[skip] {sub.name}: no test_dice (train it with the patched script)")
            continue
        rows.append((sub.name.replace(f"{args.dataset}_", ""), d["test_dice"], d["ep"]))
    rows.sort(key=lambda r: -r[1])
    print(f"\n=== TSTR utility — {args.dataset} (real-test Dice, val-selected ckpt) ===")
    print(f"{'condition':<24} {'Dice':>7} {'ep':>4}")
    for name, dice, ep in rows:
        print(f"{name:<24} {dice:>7.4f} {ep:>4}")
    json.dump([{"condition": r[0], "dice": r[1]} for r in rows],
              open(SEG_DIR / f"{args.dataset}_results.json", "w"), indent=2)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare"); pp.add_argument("--dataset", required=True)
    pp.add_argument("--gen", required=True); pp.add_argument("--device", default="cuda")
    pp.add_argument("--also_real", action="store_true")
    tp = sub.add_parser("train"); tp.add_argument("--dataset", required=True)
    tp.add_argument("--mode", required=True); tp.add_argument("--epochs", type=int, default=25)
    tp.add_argument("--batch", type=int, default=16); tp.add_argument("--base", type=int, default=32)
    tp.add_argument("--lr", type=float, default=2e-4); tp.add_argument("--workers", type=int, default=4)
    tp.add_argument("--device", default="cuda")
    tp.add_argument("--seed", type=int, default=0)
    tp.add_argument("--seeded_dir", action="store_true",
                    help="suffix output dir with _s<seed> for multi-seed runs")
    tp.add_argument("--arch", default="unet", choices=("unet", "resunet"),
                    help="segmenter architecture (resunet = #2 robustness check)")
    ep = sub.add_parser("eval"); ep.add_argument("--dataset", required=True)
    args = p.parse_args()
    {"prepare": prepare, "train": train, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
