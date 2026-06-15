"""Shared data layer for the PASD MRI viewer and generation modules.

Dataset: Placenta Accreta Spectrum Disorders (PASDs), Mendeley
https://data.mendeley.com/datasets/284gwmf5bh/1

Layout after extracting the 4 RARs:
    Placenta accreta spectrum disorders(PASDs)/extracted/
        BTFE/JPEGImages/sub###_NN.jpg   + SegmentationClass/sub###_NN.png
        TSE /JPEGImages/sub###_NN.jpg   + SegmentationClass/sub###_NN.png

Verified facts (see scripts/inspect_pasd.py history):
  - 131 BTFE subjects, 132 TSE subjects
  - Images: 512x512 grayscale (PIL mode L), uint8
  - Masks : 512x512 palette PNG with binary {0,1} labels
            (0 = background, 1 = placenta extent — NOT PAS-only)
"""

import csv
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ── Paths ──

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "Placenta accreta spectrum disorders(PASDs)" / "extracted"
INDEX_CSV = ROOT / "pasd_index.csv"

MODALITIES = ("BTFE", "TSE")

# Slice spacing is NOT documented in the dataset filenames.
# Defaults below are typical for placental MRI (in-plane ~1.5 mm,
# slice thickness ~4-5 mm). Adjust if the source publication clarifies.
DEFAULT_SPACING_MM = {
    "BTFE": (1.5, 1.5, 4.0),   # (x, y, z)
    "TSE":  (1.5, 1.5, 5.0),
}


# ── Index ──

_SLICE_RE = re.compile(r"^(?P<subject>[A-Za-z]+\d+)_(?P<idx>\d+)\.jpg$")


def build_index(force: bool = False) -> list[dict]:
    """Scan the extracted dataset, build/load the per-slice index CSV.

    Returns a list of dicts: {modality, subject, slice_idx, image, mask}.
    Caches to INDEX_CSV; pass force=True to rebuild.
    """
    if INDEX_CSV.exists() and not force:
        with open(INDEX_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["slice_idx"] = int(r["slice_idx"])
        return rows

    rows = []
    for mod in MODALITIES:
        img_dir = DATASET_DIR / mod / "JPEGImages"
        msk_dir = DATASET_DIR / mod / "SegmentationClass"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            m = _SLICE_RE.match(img_path.name)
            if not m:
                continue
            subject = m["subject"]
            idx = int(m["idx"])
            mask_path = msk_dir / f"{subject}_{idx}.png"
            rows.append({
                "modality": mod,
                "subject": subject,
                "slice_idx": idx,
                "image": str(img_path.relative_to(ROOT)),
                "mask": str(mask_path.relative_to(ROOT)) if mask_path.exists() else "",
            })

    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "subject", "slice_idx",
                                          "image", "mask"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


# ── Subject-level split ──

def subject_split(rows: list[dict], seed: int = 42,
                  ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
                  ) -> dict[str, dict[str, list[str]]]:
    """Subject-level 80/10/10 split per modality.

    Returns {modality: {"train": [...], "val": [...], "test": [...]}}.
    Subject IDs are split, not slices — prevents adjacent-slice leakage.
    """
    rng = np.random.default_rng(seed)
    by_mod = defaultdict(set)
    for r in rows:
        by_mod[r["modality"]].add(r["subject"])

    out = {}
    for mod, subj_set in by_mod.items():
        subjs = sorted(subj_set)
        rng.shuffle(subjs)
        n = len(subjs)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        out[mod] = {
            "train": subjs[:n_train],
            "val":   subjs[n_train:n_train + n_val],
            "test":  subjs[n_train + n_val:],
        }
    return out


# ── Volume assembly (lazy-cached) ──

_volume_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 6   # keep at most N volumes in memory


def _evict_if_needed():
    """LRU-ish: drop the oldest cached volume if we exceed the cap."""
    while len(_volume_cache) > _CACHE_MAX:
        _volume_cache.pop(next(iter(_volume_cache)))


def list_subjects(modality: str) -> list[dict]:
    """Return [{subject, slice_count, has_mask, first_slice, last_slice}]."""
    rows = build_index()
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["modality"] == modality:
            by_subj[r["subject"]].append(r)
    out = []
    for subj, items in sorted(by_subj.items()):
        items.sort(key=lambda x: x["slice_idx"])
        out.append({
            "subject": subj,
            "slice_count": len(items),
            "has_mask": all(bool(it["mask"]) for it in items),
            "first_slice": items[0]["slice_idx"],
            "last_slice": items[-1]["slice_idx"],
        })
    return out


def get_volume(modality: str, subject: str
               ) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (image_vol, mask_vol), both shape (H, W, D), dtype uint8.

    D = number of slices, sorted by slice_idx ascending.
    mask_vol is all-zero if no masks are available for this subject.
    Cached; thread-safe.
    """
    key = (modality, subject)
    with _cache_lock:
        if key in _volume_cache:
            # bump to "most recent" by re-inserting
            vol = _volume_cache.pop(key)
            _volume_cache[key] = vol
            return vol

    rows = [r for r in build_index()
            if r["modality"] == modality and r["subject"] == subject]
    if not rows:
        raise ValueError(f"No slices for {modality}/{subject}")
    rows.sort(key=lambda r: r["slice_idx"])

    img_stack, msk_stack = [], []
    for r in rows:
        img = np.array(Image.open(ROOT / r["image"]).convert("L"))
        img_stack.append(img)
        if r["mask"]:
            # palette PNG → label array via direct uint8 channel
            msk = np.array(Image.open(ROOT / r["mask"]))
            if msk.ndim == 3:
                msk = msk[..., 0]
            msk = (msk > 0).astype(np.uint8)  # binarize
        else:
            msk = np.zeros_like(img_stack[-1], dtype=np.uint8)
        msk_stack.append(msk)

    # Stack along D axis → (H, W, D)
    image_vol = np.stack(img_stack, axis=-1)
    mask_vol = np.stack(msk_stack, axis=-1)

    with _cache_lock:
        _volume_cache[key] = (image_vol, mask_vol)
        _evict_if_needed()
    return image_vol, mask_vol


def get_spacing(modality: str) -> tuple[float, float, float]:
    return DEFAULT_SPACING_MM.get(modality, (1.0, 1.0, 1.0))


# ── PyTorch Dataset (for pix2pix / VAE / LDM training) ──

def _try_import_torch():
    """Imported lazily so the viewer never pays the torch import cost."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    return torch, F, Dataset


def get_pasd_dataset(modality: str, split: str = "train",
                     image_size: int = 256, augment: bool = True,
                     seed: int = 42):
    """Build a PyTorch Dataset of (image, mask) pairs.

    Returns tensors:
      image: (3, image_size, image_size) float32, range [-1, 1] (grayscale replicated)
      mask : (1, image_size, image_size) float32, range [-1, 1] (binary)

    Subject-level split (train/val/test) using subject_split().
    Augmentations (train only): horizontal flip, mild affine, intensity jitter.
    """
    torch, F, Dataset = _try_import_torch()
    from torchvision import transforms
    from torchvision.transforms import functional as TF

    rows = build_index()
    splits = subject_split(rows, seed=seed)
    subj_set = set(splits[modality][split])
    items = [r for r in rows
    
             if r["modality"] == modality and r["subject"] in subj_set]

    rng = np.random.default_rng(seed)

    class _PasdDataset(Dataset):
        def __init__(self):
            self.items = items
            self.image_size = image_size
            self.augment = augment

        def __len__(self):
            return len(self.items)

        def _load_pair(self, idx):
            r = self.items[idx]
            img = Image.open(ROOT / r["image"]).convert("L")
            msk = Image.open(ROOT / r["mask"]) if r["mask"] else None
            if msk is None:
                msk = Image.new("L", img.size, 0)
            elif msk.mode != "L":
                msk = msk.convert("L")
            # Binarize the palette mask
            msk_arr = (np.array(msk) > 0).astype(np.uint8) * 255
            msk = Image.fromarray(msk_arr, "L")
            return img, msk

        def __getitem__(self, idx):
            img, msk = self._load_pair(idx)

            # Resize (image: bilinear, mask: nearest)
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            msk = msk.resize((self.image_size, self.image_size), Image.NEAREST)

            if self.augment:
                # H-flip together
                if rng.random() < 0.5:
                    img = TF.hflip(img); msk = TF.hflip(msk)
                # Mild affine together
                if rng.random() < 0.5:
                    angle = float(rng.uniform(-5, 5))
                    tx = float(rng.uniform(-0.05, 0.05) * self.image_size)
                    ty = float(rng.uniform(-0.05, 0.05) * self.image_size)
                    img = TF.affine(img, angle=angle, translate=(tx, ty),
                                    scale=1.0, shear=0.0,
                                    interpolation=TF.InterpolationMode.BILINEAR,
                                    fill=0)
                    msk = TF.affine(msk, angle=angle, translate=(tx, ty),
                                    scale=1.0, shear=0.0,
                                    interpolation=TF.InterpolationMode.NEAREST,
                                    fill=0)
                # Intensity jitter (image only)
                if rng.random() < 0.5:
                    factor = float(rng.uniform(0.9, 1.1))
                    img = TF.adjust_brightness(img, factor)

            # Convert to tensors in [-1, 1]
            img_t = TF.to_tensor(img)        # (1, H, W) in [0,1]
            img_t = img_t.repeat(3, 1, 1)    # (3, H, W) grayscale->RGB
            img_t = img_t * 2.0 - 1.0

            msk_t = TF.to_tensor(msk)        # (1, H, W) in [0,1]
            msk_t = (msk_t > 0.5).float()
            msk_t = msk_t * 2.0 - 1.0

            return {"image": img_t, "mask": msk_t,
                    "subject": self.items[idx]["subject"],
                    "slice_idx": self.items[idx]["slice_idx"]}

    return _PasdDataset()


# ── CLI: build index & print summary ──

if __name__ == "__main__":
    rows = build_index(force=True)
    by_mod = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_mod[r["modality"]][r["subject"]] += 1
    for mod, subjs in by_mod.items():
        n_subj = len(subjs)
        n_slice = sum(subjs.values())
        print(f"{mod:5s}: {n_subj:3d} subjects, {n_slice:5d} slices "
              f"(avg {n_slice/n_subj:.1f}/subj)")
    splits = subject_split(rows)
    for mod, s in splits.items():
        print(f"  {mod} split: train={len(s['train'])} "
              f"val={len(s['val'])} test={len(s['test'])}")
    print(f"\nIndex written to {INDEX_CSV}")
