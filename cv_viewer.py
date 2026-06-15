"""CTA / TOF-MRA Viewer backend (TopBrain dataset).

Serves 2D MPR slices (axial/sagittal/coronal) with multi-class vessel mask
overlay, per-slice mask-area profile, and NIfTI volume exports for NiiVue.

Mirrors the shape of mri_viewer.py but supports two modalities (ct, mr) on a
shared 256³ isotropic grid (volumes are co-registered upstream in
prepare_topbrain.py). Each modality has its own ITK-SNAP labelmap because
labels 35+ encode different vessel groups per modality.
"""

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import cv_data as cd


# ── Constants ──

AXES = ("axial", "sagittal", "coronal")
MODALITIES = ("ct", "mr")
SPACING = (0.5, 0.5, 0.5)  # mm, isotropic — set by prepare_topbrain.py

# Default windows useful for clinicians (CT only; MR has no Hounsfield scale).
CT_WINDOW_PRESETS = {
    "vessel": {"wl": 200,  "ww": 700},   # CT angiography
    "brain":  {"wl": 40,   "ww": 80},
    "bone":   {"wl": 600,  "ww": 1500},
    "wide":   {"wl": 50,   "ww": 1200},
}
DEFAULT_CT_WL = CT_WINDOW_PRESETS["vessel"]["wl"]
DEFAULT_CT_WW = CT_WINDOW_PRESETS["vessel"]["ww"]


# ── Labelmap loading (cached) ──

_ROOT = Path(__file__).parent
_LABELMAP_DIR = Path("/home/marija/Documents/TopBrain_Data_Release_Batches1n2_081425/itksnap_labelmap_txt")
_LABELMAP_FILES = {
    "ct": _LABELMAP_DIR / "labelmap_topbrain_ct.txt",
    "mr": _LABELMAP_DIR / "labelmap_topbrain_mr.txt",
}
_LABELMAP_CACHE: dict[str, dict[int, dict]] = {}


def _parse_labelmap(path: Path) -> dict[int, dict]:
    """Return {label_id: {color: (r,g,b), name: str}} from an ITK-SNAP labelmap.

    Skips background (id 0) — caller can render that as transparent.
    """
    out: dict[int, dict] = {}
    pat = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+\d+\s+\d+\s+\"([^\"]+)\"")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if not m:
            continue
        idx, r, g, b, name = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]
        if idx == 0:
            continue  # background
        out[idx] = {"color": (r, g, b), "name": name}
    return out


def get_labelmap(modality: str) -> dict[int, dict]:
    """Cached accessor for the official ITK-SNAP labelmap."""
    modality = modality.lower()
    if modality not in _LABELMAP_FILES:
        raise ValueError(f"modality must be one of {tuple(_LABELMAP_FILES)}")
    if modality not in _LABELMAP_CACHE:
        path = _LABELMAP_FILES[modality]
        if not path.exists():
            # Fallback if dataset path moved — build a synthetic palette
            _LABELMAP_CACHE[modality] = _synthetic_labelmap()
        else:
            _LABELMAP_CACHE[modality] = _parse_labelmap(path)
    return _LABELMAP_CACHE[modality]


def _synthetic_labelmap() -> dict[int, dict]:
    """Fallback colormap (tab20 + tab20b) if the dataset's official map missing."""
    import colorsys
    out = {}
    for i in range(1, 41):
        h = ((i * 0.618033988749895) % 1.0)  # golden-ratio hue cycling
        s = 0.7 if i % 2 else 0.9
        v = 0.95
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out[i] = {
            "color": (int(r * 255), int(g * 255), int(b * 255)),
            "name": f"label-{i}",
        }
    return out


# ── Subject listing ──

def list_subjects() -> list[dict]:
    """Return list of {patient, n_slices_with_tissue} for the sidebar."""
    rows = cd.build_index()
    return [{"patient": r["patient"], "n_slices": int(r["n_slices"])}
            for r in rows]


def axis_length(subject: str, axis: str) -> int:
    """How many slices along this axis (volumes are 256³ so always 256)."""
    vol = cd._vol(subject, "ct")
    H, W, D = vol.shape
    return {"axial": D, "sagittal": W, "coronal": H}[axis]


# ── Slice extraction ──

def _slice_arrays(image_vol: np.ndarray, mask_vol: np.ndarray,
                  axis: str, idx: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    if axis == "axial":
        return image_vol[:, :, idx], mask_vol[:, :, idx]
    if axis == "sagittal":
        return image_vol[:, idx, :], mask_vol[:, idx, :]
    if axis == "coronal":
        return image_vol[idx, :, :], mask_vol[idx, :, :]
    raise ValueError(f"unknown axis: {axis}")


def _apply_window(img2d: np.ndarray, wl: Optional[float],
                  ww: Optional[float]) -> np.ndarray:
    """CT/MR window-level → uint8. Passes through if both are None."""
    if wl is None or ww is None:
        # Auto-normalize: 1st/99th percentile
        lo, hi = np.percentile(img2d, (1.0, 99.0))
        if hi <= lo:
            lo, hi = float(img2d.min()), float(img2d.max() + 1e-6)
    else:
        lo, hi = wl - ww / 2.0, wl + ww / 2.0
    out = np.clip((img2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def _mask_outline(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Per-label outline. Returns same shape as mask with original label IDs
    only on boundary pixels, 0 elsewhere — preserves per-label coloring."""
    if not mask.any():
        return mask
    from PIL import ImageFilter
    # Outline of the *combined* foreground; we'll re-attach labels after.
    fg_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, "L")
    dil = fg_pil.filter(ImageFilter.MaxFilter(2 * thickness + 1))
    ero = fg_pil.filter(ImageFilter.MinFilter(2 * thickness + 1))
    boundary = (np.array(dil) > 0) & (np.array(ero) == 0)
    out = mask.copy()
    out[~boundary] = 0
    return out


def _colorize_mask(mask2d: np.ndarray, modality: str,
                   alpha: float, outline: bool) -> np.ndarray:
    """Multi-class colorization. Returns RGBA uint8 of same H×W."""
    if outline:
        mask2d = _mask_outline(mask2d)
    H, W = mask2d.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    if not mask2d.any():
        return rgba
    lmap = get_labelmap(modality)
    a = int(np.clip(alpha * 255, 0, 255))
    for lbl, info in lmap.items():
        m = (mask2d == lbl)
        if not m.any():
            continue
        r, g, b = info["color"]
        rgba[m] = (r, g, b, a)
    return rgba


# ── Rendering ──

def render_slice(modality: str, subject: str, axis: str, idx: int,
                 mask_alpha: float = 0.5,
                 wl: Optional[float] = None,
                 ww: Optional[float] = None,
                 outline: bool = False,
                 mode: str = "both") -> Image.Image:
    """Return a PIL RGB image.

    mode: "both"       — image + vessel-mask overlay (default)
          "image_only" — image, mask hidden
          "mask_only"  — black background, mask at full alpha (per-vessel colors)
    """
    modality = modality.lower()
    if modality not in MODALITIES:
        raise ValueError(f"modality must be one of {MODALITIES}")
    image_vol = cd._vol(subject, modality)
    mask_vol = cd._vol(subject, f"{modality}_mask")
    img2d, msk2d = _slice_arrays(image_vol, mask_vol.astype(np.int32), axis, idx)

    # Default to vessel window for CT if user didn't specify
    if modality == "ct" and (wl is None and ww is None):
        wl, ww = DEFAULT_CT_WL, DEFAULT_CT_WW
    img8 = _apply_window(img2d, wl, ww)

    if mode == "mask_only":
        img8 = np.zeros_like(img8)
        effective_alpha = 1.0
    elif mode == "image_only":
        effective_alpha = 0.0
    else:
        effective_alpha = mask_alpha

    base = np.stack([img8, img8, img8, np.full_like(img8, 255)], axis=-1)
    base_pil = Image.fromarray(base, "RGBA")

    if effective_alpha > 0 and msk2d.any():
        overlay = _colorize_mask(msk2d, modality, effective_alpha, outline)
        overlay_pil = Image.fromarray(overlay, "RGBA")
        base_pil = Image.alpha_composite(base_pil, overlay_pil)

    return base_pil.convert("RGB")


def mask_area_profile(modality: str, subject: str,
                      axis: str = "axial") -> list[int]:
    """Per-slice mask voxel count (any vessel class) along the requested axis."""
    modality = modality.lower()
    mask_vol = cd._vol(subject, f"{modality}_mask")
    bin_vol = (mask_vol > 0).astype(np.int32)
    if axis == "axial":
        return bin_vol.sum(axis=(0, 1)).tolist()
    if axis == "sagittal":
        return bin_vol.sum(axis=(0, 2)).tolist()
    if axis == "coronal":
        return bin_vol.sum(axis=(1, 2)).tolist()
    raise ValueError(f"unknown axis: {axis}")


def render_thumbnail(modality: str, subject: str,
                     max_dim: int = 128) -> Image.Image:
    """Mid-axial thumbnail with mask overlay for sidebar."""
    modality = modality.lower()
    vol = cd._vol(subject, modality)
    mid = vol.shape[2] // 2
    img = render_slice(modality, subject, "axial", mid, mask_alpha=0.5)
    img.thumbnail((max_dim, max_dim), Image.BILINEAR)
    return img


# ── NIfTI export for NiiVue ──

def to_nifti_bytes(modality: str, subject: str, kind: str = "image") -> bytes:
    """Serialize volume as gzipped NIfTI for the NiiVue 3D viewer.

    kind='image' returns the grayscale volume (CT or MR).
    kind='mask'  returns the multi-class vessel labels.
    """
    import gzip
    import nibabel as nib

    modality = modality.lower()
    if kind == "image":
        vol = cd._vol(subject, modality).astype(np.float32)
    elif kind == "mask":
        vol = cd._vol(subject, f"{modality}_mask").astype(np.int16)
    else:
        raise ValueError(f"kind must be image or mask, got {kind}")

    # Transpose to (X, Y, Z) and flip Y for radiological orientation
    arr = np.transpose(vol, (1, 0, 2))[:, ::-1, :].copy()
    sx, sy, sz = SPACING
    affine = np.diag([sx, sy, sz, 1.0])
    nii = nib.Nifti1Image(arr, affine)
    nii.header.set_xyzt_units("mm")
    return gzip.compress(nii.to_bytes())
