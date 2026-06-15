"""TopBrain (MICCAI 2025) cerebrovascular dataset utilities.

Operates on the OUTPUT of prepare_topbrain.py: 256³ isotropic volumes for
each modality (mr, ct) and matching multi-class vessel masks.

Provides:
  - build_index(): subject inventory
  - subject_split(): subject-level train/val/test split (deterministic)
  - get_paired_2d_dataset(): PyTorch Dataset that yields paired (mr, ct, mask)
    2-D axial slices. Used for pix2pix MR→CT translation and (later) mask-
    conditional synthesis.

Slice extraction:
  - Axial slices (along the z axis) — anatomically natural for brain scans.
  - Skip empty slices (mostly air outside the head) via a tissue threshold
    on the MR volume.
  - Each volume contributes ~150-220 non-empty slices, so 25 patients ×
    ~180 slices ≈ 4 500 paired slices per modality direction.

Intensity normalization:
  - MR: per-volume scale to [0, 1] using the volume's 99th percentile
        (robust to scanner outliers).
  - CT: clamp to [-200, 1500] HU (vessel + bone window) → rescale to [0, 1].
        These limits are wide enough to keep both soft tissue and contrast.
  - Both then mapped to [-1, 1] for tanh-output generators.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "topbrain_preprocessed"
INDEX_CSV = ROOT / "cv_index.csv"

# Intensity clamps used to normalize each modality to [-1, 1].
MR_PCTL = 99.0
CT_CLAMP = (-200.0, 1500.0)

# Skip slices with this little MR tissue (mostly air) — keeps loaders fast.
MIN_TISSUE_FRACTION = 0.05


# ── Lazy nibabel + cache ──

_volume_cache: dict[tuple[str, str], np.ndarray] = {}


def _load(path: Path) -> np.ndarray:
    """Load a NIfTI as a numpy array; cache by (kind, pid)."""
    import nibabel as nib
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _vol(pid: str, kind: str) -> np.ndarray:
    """Cached load of one volume. kind ∈ {mr, ct, mr_mask, ct_mask}."""
    key = (pid, kind)
    if key in _volume_cache:
        return _volume_cache[key]
    p = DATA_DIR / kind / f"topcow_{pid}.nii.gz"
    arr = _load(p)
    _volume_cache[key] = arr
    return arr


def clear_cache():
    _volume_cache.clear()


# ── Subject index ──

def build_index(force: bool = False) -> list[dict]:
    """Scan preprocessed dir, write CSV listing subjects and slice counts."""
    if INDEX_CSV.exists() and not force:
        with open(INDEX_CSV, newline="") as f:
            return list(csv.DictReader(f))
    pids = sorted([p.name.split("_")[1].split(".")[0]
                   for p in (DATA_DIR / "mr").glob("topcow_*.nii.gz")])
    rows = []
    for pid in pids:
        mr = _vol(pid, "mr")
        # Per-slice tissue fraction along z
        thresh = np.percentile(mr[mr > 0], 25) if mr.max() > 0 else 0
        nonempty = sum(
            ((mr[:, :, z] > thresh).sum() / mr[:, :, z].size)
            > MIN_TISSUE_FRACTION
            for z in range(mr.shape[2])
        )
        rows.append({
            "patient": pid,
            "n_slices": str(nonempty),
            "first_z": str(0),
            "last_z": str(mr.shape[2] - 1),
        })
    with open(INDEX_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient", "n_slices",
                                          "first_z", "last_z"])
        w.writeheader()
        w.writerows(rows)
    return rows


def subject_split(seed: int = 42,
                  ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
                  ) -> dict[str, list[str]]:
    """Deterministic subject-level train/val/test split."""
    rows = build_index()
    pids = sorted(r["patient"] for r in rows)
    rng = np.random.default_rng(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_tr = max(1, int(round(n * ratios[0])))
    n_va = max(1, int(round(n * ratios[1])))
    return {
        "train": pids[:n_tr],
        "val":   pids[n_tr:n_tr + n_va],
        "test":  pids[n_tr + n_va:],
    }


# ── Intensity normalization ──

def _norm_mr_slice(s: np.ndarray) -> np.ndarray:
    """MR slice → [-1, 1] using a per-VOLUME 99th-percentile reference.

    Volume-level so all slices from one patient share a scale. The caller
    passes the percentile as part of the slice extractor below.
    """
    raise NotImplementedError("use _norm_slice_with_ref")


def _norm_slice_with_ref(s: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = np.clip(s.astype(np.float32), lo, hi)
    s = (s - lo) / max(hi - lo, 1e-6)
    return s * 2.0 - 1.0  # [-1, 1]


def _tissue_indices(mr_vol: np.ndarray) -> list[int]:
    """Indices of axial slices with > MIN_TISSUE_FRACTION tissue."""
    if mr_vol.max() <= 0:
        return list(range(mr_vol.shape[2]))
    thresh = np.percentile(mr_vol[mr_vol > 0], 25)
    keep = []
    for z in range(mr_vol.shape[2]):
        frac = (mr_vol[:, :, z] > thresh).sum() / mr_vol[:, :, z].size
        if frac > MIN_TISSUE_FRACTION:
            keep.append(z)
    return keep


# ── PyTorch Dataset ──

def get_paired_2d_dataset(split: str = "train", augment: bool = True,
                          include_masks: bool = False,
                          image_size: int = 256, seed: int = 42):
    """Yields paired (mr, ct, [mask]) 2-D axial slices.

    The volumes are already on a shared 256³ grid (see prepare_topbrain.py),
    so slice z in MR and slice z in CT are anatomically the same slice.

    Each __getitem__ returns:
        {"mr": tensor (1, H, W) in [-1, 1],
         "ct": tensor (1, H, W) in [-1, 1],
         "mask_mr": tensor (H, W) long  [optional],
         "mask_ct": tensor (H, W) long  [optional],
         "patient": str, "z": int}
    """
    from torchvision.transforms import functional as TF

    splits = subject_split(seed=seed)
    pids = splits[split]
    if not pids:
        raise ValueError(f"empty split {split}")
    rng = np.random.default_rng(seed)

    # Pre-compute per-volume references (the MR 99th percentile)
    # and the set of non-empty slices per patient.
    plan: list[tuple[str, int, float]] = []  # (pid, z, mr_p99)
    pid_mr_p99: dict[str, float] = {}
    for pid in pids:
        mr_vol = _vol(pid, "mr")
        p99 = float(np.percentile(mr_vol[mr_vol > 0], MR_PCTL)) if mr_vol.max() > 0 else 1.0
        pid_mr_p99[pid] = p99
        for z in _tissue_indices(mr_vol):
            plan.append((pid, z, p99))

    class _CV2D(Dataset):
        def __init__(self):
            self.items = plan
            self.augment = augment
            self.image_size = image_size
            self.include_masks = include_masks

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            pid, z, mr_p99 = self.items[i]
            mr_vol = _vol(pid, "mr")
            ct_vol = _vol(pid, "ct")
            mr_s = mr_vol[:, :, z]
            ct_s = ct_vol[:, :, z]

            mr_n = _norm_slice_with_ref(mr_s, 0.0, mr_p99)
            ct_n = _norm_slice_with_ref(ct_s, CT_CLAMP[0], CT_CLAMP[1])

            mr_pil = Image.fromarray(((mr_n + 1) * 127.5).clip(0, 255).astype(np.uint8), "L")
            ct_pil = Image.fromarray(((ct_n + 1) * 127.5).clip(0, 255).astype(np.uint8), "L")

            mask_mr_pil = mask_ct_pil = None
            if self.include_masks:
                mm = _vol(pid, "mr_mask")[:, :, z].astype(np.int16)
                mc = _vol(pid, "ct_mask")[:, :, z].astype(np.int16)
                mask_mr_pil = Image.fromarray(mm.astype(np.uint8), "L")
                mask_ct_pil = Image.fromarray(mc.astype(np.uint8), "L")

            # Resize to target image_size
            if mr_pil.size != (self.image_size, self.image_size):
                mr_pil = mr_pil.resize((self.image_size,) * 2, Image.BILINEAR)
                ct_pil = ct_pil.resize((self.image_size,) * 2, Image.BILINEAR)
                if mask_mr_pil is not None:
                    mask_mr_pil = mask_mr_pil.resize((self.image_size,) * 2, Image.NEAREST)
                    mask_ct_pil = mask_ct_pil.resize((self.image_size,) * 2, Image.NEAREST)

            if self.augment:
                if rng.random() < 0.5:
                    mr_pil = TF.hflip(mr_pil); ct_pil = TF.hflip(ct_pil)
                    if mask_mr_pil is not None:
                        mask_mr_pil = TF.hflip(mask_mr_pil)
                        mask_ct_pil = TF.hflip(mask_ct_pil)

            mr_t = TF.to_tensor(mr_pil) * 2.0 - 1.0   # (1, H, W) in [-1, 1]
            ct_t = TF.to_tensor(ct_pil) * 2.0 - 1.0
            out = {"mr": mr_t, "ct": ct_t, "patient": pid, "z": z}
            if self.include_masks and mask_mr_pil is not None:
                out["mask_mr"] = torch.from_numpy(np.array(mask_mr_pil)).long()
                out["mask_ct"] = torch.from_numpy(np.array(mask_ct_pil)).long()
            return out

    return _CV2D()


# ── CLI: build index + summary ──

if __name__ == "__main__":
    rows = build_index(force=True)
    print(f"TopBrain index: {len(rows)} patients")
    n_slices = sum(int(r["n_slices"]) for r in rows)
    print(f"  Total tissue slices (axial, MR-based): {n_slices}")
    print(f"  Avg per patient: {n_slices/max(len(rows),1):.0f}")
    splits = subject_split()
    for k, v in splits.items():
        print(f"  {k}: {len(v)} patients")
