"""Prostate158 paired 2D axial dataset loader.

Mirrors cv_data.py but for prostate158 — paired same-grid (T2, ADC, DWI) with
multi-class anatomy mask (0/1/2 = bg / peripheral zone / central gland).

Splits use the official train.csv / valid.csv from the dataset; test set is the
separate prostate158_test release.
"""

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "prostate158_preprocessed"
INDEX_CSV = ROOT / "prostate_index.csv"

# Intensity clamping for [-1, 1] normalization. T2 uses 99th-percentile per
# volume; ADC clamps to a generous physiological range; DWI uses 99th-percentile.
T2_PCTL   = 99.0
ADC_CLAMP = (0.0, 3500.0)
DWI_PCTL  = 99.0

# Skip 2D slices with too little tissue (mostly air/background outside body)
MIN_TISSUE_FRACTION = 0.05

# Modality kinds available in the preprocessed dir
KINDS = ("t2", "adc", "dwi", "t2_anatomy")


# ── Lazy nibabel + per-volume cache ───────────────────────────────────────

_volume_cache: dict[tuple[str, str], np.ndarray] = {}


def _load(path: Path) -> np.ndarray:
    import nibabel as nib
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _vol(pid: str, kind: str) -> np.ndarray:
    """Cached load of one volume. kind ∈ KINDS."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind}")
    key = (pid, kind)
    if key in _volume_cache:
        return _volume_cache[key]
    p = DATA_DIR / kind / f"case_{pid}.nii.gz"
    arr = _load(p)
    _volume_cache[key] = arr
    return arr


def clear_cache():
    _volume_cache.clear()


# ── Patient index ─────────────────────────────────────────────────────────

def _read_manifest_splits() -> dict[str, list[str]]:
    """Reach split lists from the prep manifest (train/val/test patient IDs)."""
    mf = DATA_DIR / "manifest.json"
    if not mf.exists():
        return {"train": [], "val": [], "test": []}
    with open(mf) as f:
        m = json.load(f)
    return m["splits"]


def list_patients(split: Optional[str] = None) -> list[str]:
    """Return zero-padded patient IDs that exist on disk for all 4 modalities.

    split: "train" | "val" | "test" | None (= all).
    """
    splits = _read_manifest_splits()
    if split:
        candidates = splits.get(split, [])
    else:
        candidates = sum(splits.values(), [])
    out = []
    for pid in candidates:
        if all((DATA_DIR / k / f"case_{pid}.nii.gz").exists() for k in KINDS):
            out.append(pid)
    return out


def build_index(force: bool = False) -> list[dict]:
    """Build / cache a per-patient index with tissue-bearing slice counts."""
    if INDEX_CSV.exists() and not force:
        with open(INDEX_CSV, newline="") as f:
            return list(csv.DictReader(f))
    rows = []
    for pid in list_patients(split=None):
        t2 = _vol(pid, "t2")
        # Threshold at the in-body 25th percentile to get a "tissue" mask
        nz = t2[t2 > 0]
        thr = float(np.percentile(nz, 25)) if nz.size else 0.0
        n_with_tissue = sum(
            ((t2[:, :, z] > thr).sum() / t2[:, :, z].size) > MIN_TISSUE_FRACTION
            for z in range(t2.shape[2])
        )
        rows.append({
            "patient": pid,
            "n_slices": str(n_with_tissue),
            "z_total": str(t2.shape[2]),
        })
    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "n_slices", "z_total"])
        w.writeheader()
        w.writerows(rows)
    return rows


# ── Intensity normalization ───────────────────────────────────────────────

def _norm_t2(arr_vol: np.ndarray) -> tuple[float, float]:
    """Return (lo, hi) clamps for T2 → [-1, 1] from this volume's 99th pctile."""
    nz = arr_vol[arr_vol > 0]
    hi = float(np.percentile(nz, T2_PCTL)) if nz.size else 1.0
    return (0.0, max(hi, 1e-3))


def _norm_dwi(arr_vol: np.ndarray) -> tuple[float, float]:
    nz = arr_vol[arr_vol > 0]
    hi = float(np.percentile(nz, DWI_PCTL)) if nz.size else 1.0
    return (0.0, max(hi, 1e-3))


def _norm_adc(arr_vol: np.ndarray) -> tuple[float, float]:
    return ADC_CLAMP


def _normalize_with_clamp(s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = np.clip(s.astype(np.float32), lo, hi)
    s = (s - lo) / max(hi - lo, 1e-6)
    return s * 2.0 - 1.0


# ── Tissue-bearing slice indices ──────────────────────────────────────────

def _tissue_indices(t2_vol: np.ndarray) -> list[int]:
    nz = t2_vol[t2_vol > 0]
    if not nz.size:
        return list(range(t2_vol.shape[2]))
    thr = float(np.percentile(nz, 25))
    keep = []
    for z in range(t2_vol.shape[2]):
        frac = (t2_vol[:, :, z] > thr).sum() / t2_vol[:, :, z].size
        if frac > MIN_TISSUE_FRACTION:
            keep.append(z)
    return keep


# ── PyTorch Dataset ───────────────────────────────────────────────────────

def get_paired_2d_dataset(split: str = "train",
                          input_modality: str = "t2",
                          target_modality: str = "adc",
                          augment: bool = True,
                          include_mask: bool = False,
                          image_size: int = 256):
    """Yields paired (input, target, [mask]) 2-D axial slices in [-1, 1].

    input_modality / target_modality ∈ {"t2", "adc", "dwi"}.
    include_mask=True adds the multi-class t2_anatomy mask (long tensor) for
    SPADE-style training or mask-conditional diffusion.
    """
    import torch
    from torch.utils.data import Dataset
    from torch.nn import functional as F

    if input_modality not in {"t2", "adc", "dwi"}:
        raise ValueError(f"input_modality must be one of t2/adc/dwi")
    if target_modality not in {"t2", "adc", "dwi"}:
        raise ValueError(f"target_modality must be one of t2/adc/dwi")

    norm_fn = {
        "t2":  _norm_t2,
        "adc": _norm_adc,
        "dwi": _norm_dwi,
    }

    patients = list_patients(split=split)
    if not patients:
        raise RuntimeError(f"No patients in {split} split — did you run "
                           f"prepare_prostate158.py?")

    # Flatten: list of (pid, z) for tissue-bearing slices
    samples = []
    norm_ranges_in: dict[str, tuple[float, float]] = {}
    norm_ranges_tg: dict[str, tuple[float, float]] = {}
    for pid in patients:
        t2 = _vol(pid, "t2")
        zs = _tissue_indices(t2)
        for z in zs:
            samples.append((pid, z))
        norm_ranges_in[pid] = norm_fn[input_modality](_vol(pid, input_modality))
        norm_ranges_tg[pid] = norm_fn[target_modality](_vol(pid, target_modality))

    class _Prostate2D(Dataset):
        def __len__(self_): return len(samples)

        def __getitem__(self_, i):
            pid, z = samples[i]
            x_vol = _vol(pid, input_modality)
            y_vol = _vol(pid, target_modality)

            x = _normalize_with_clamp(x_vol[:, :, z], *norm_ranges_in[pid])
            y = _normalize_with_clamp(y_vol[:, :, z], *norm_ranges_tg[pid])

            # Random hflip augmentation (no rotation — anisotropic data)
            if augment and torch.rand(1).item() < 0.5:
                x = np.ascontiguousarray(x[:, ::-1])
                y = np.ascontiguousarray(y[:, ::-1])

            # (H, W) → (1, H, W); resize if needed
            xt = torch.from_numpy(x).float().unsqueeze(0)
            yt = torch.from_numpy(y).float().unsqueeze(0)
            if xt.shape[-1] != image_size:
                xt = F.interpolate(xt.unsqueeze(0), size=(image_size, image_size),
                                   mode="bilinear", align_corners=False).squeeze(0)
                yt = F.interpolate(yt.unsqueeze(0), size=(image_size, image_size),
                                   mode="bilinear", align_corners=False).squeeze(0)

            item = {"input": xt, "target": yt, "patient": pid, "z": z}
            if include_mask:
                m = _vol(pid, "t2_anatomy")[:, :, z].astype(np.int64)
                if augment and item.get("_hflip"):  # already applied above
                    pass
                mt = torch.from_numpy(m).long().unsqueeze(0)
                if mt.shape[-1] != image_size:
                    mt = F.interpolate(mt.float().unsqueeze(0),
                                       size=(image_size, image_size),
                                       mode="nearest").squeeze(0).long()
                item["mask"] = mt.squeeze(0)
            return item

    return _Prostate2D()
