"""Aggregate multi-seed TSTR results into mean +/- std per condition.

Reads tstr_synth_models/<dataset>_<mode>[_<arch>]_s<seed>/best.pt and groups by
(dataset, mode, arch), reporting mean and std of real-test Dice across seeds.
Covers the main 3-seed sweep (#1), the real+synth augmentation runs (#5), and
the ResUNet robustness runs (#2).

Usage:  python aggregate_tstr.py
"""
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

SEG = Path(__file__).parent / "tstr_synth_models"
pat = re.compile(r"^(prostate|topbrain|pasd)_(.+?)(_resunet)?_s(\d+)$")

groups = defaultdict(list)   # (dataset, mode, arch) -> [dice,...]
for d in sorted(SEG.glob("*_s*")):
    m = pat.match(d.name)
    ck = d / "best.pt"
    if not m or not ck.exists():
        continue
    ds, mode, arch, seed = m.group(1), m.group(2), m.group(3), m.group(4)
    arch = "resunet" if arch else "unet"
    try:
        dice = float(torch.load(ck, map_location="cpu", weights_only=False)["dice"])
    except Exception:
        continue
    groups[(ds, mode, arch)].append(dice)

def show(title, filt):
    print(f"\n=== {title} ===")
    print(f"{'dataset':<10}{'condition':<22}{'arch':<9}{'n':>3}  {'mean':>7}  {'std':>6}")
    for (ds, mode, arch), vals in sorted(groups.items()):
        if not filt(ds, mode, arch):
            continue
        v = np.array(vals)
        print(f"{ds:<10}{mode:<22}{arch:<9}{len(v):>3}  {v.mean():>7.4f}  {v.std():>6.4f}")

# #1 main synth-only, UNet
show("Main TSTR (synth-only, UNet, multi-seed)",
     lambda ds, mode, arch: arch == "unet" and not mode.startswith("real_synth_"))
# #5 augmentation
show("Augmentation (real+synth, UNet, multi-seed)",
     lambda ds, mode, arch: arch == "unet" and mode.startswith("real_synth_"))
# #2 robustness
show("Robustness (ResUNet, synth-only)",
     lambda ds, mode, arch: arch == "resunet")
