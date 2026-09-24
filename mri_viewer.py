"""MRI Viewer backend.

Serves 2D MPR slices (axial/sagittal/coronal) with optional GT mask overlay,
plus NIfTI exports of image and mask volumes for the NiiVue 3D viewer.

Pattern follows the lazy-cached approach used in hovernet.py / clam.py.
"""

import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

import pasd_data as pd


# ── Constants ──

AXES = ("axial", "sagittal", "coronal")

# Mask overlay color (RGBA tuple, 0-255). Magenta — distinct from grayscale MRI.
MASK_RGBA = (255, 0, 128, 255)


# ── Slice extraction ──

def _slice_arrays(image_vol: np.ndarray, mask_vol: np.ndarray,
                  axis: str, idx: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pull a 2D (H', W') slice from the (H, W, D) volumes.

    Axial    → img[:, :, idx]    shape (H, W)
    Sagittal → img[:, idx, :]    shape (H, D)
    Coronal  → img[idx, :, :]    shape (W, D)
    """
    if axis == "axial":
        return image_vol[:, :, idx], mask_vol[:, :, idx]
    if axis == "sagittal":
        return image_vol[:, idx, :], mask_vol[:, idx, :]
    if axis == "coronal":
        return image_vol[idx, :, :], mask_vol[idx, :, :]
    raise ValueError(f"unknown axis: {axis}")


def axis_length(modality: str, subject: str, axis: str) -> int:
    """How many slices exist along this axis for this volume."""
    image_vol, _ = pd.get_volume(modality, subject)
    H, W, D = image_vol.shape
    return {"axial": D, "sagittal": W, "coronal": H}[axis]


# ── Rendering ──

def _apply_window(img2d: np.ndarray, wl: Optional[float],
                  ww: Optional[float]) -> np.ndarray:
    """Apply window-level if requested; else passthrough."""
    if wl is None or ww is None:
        return img2d
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    out = np.clip((img2d.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def _scale_for_display(img2d: np.ndarray, mask2d: np.ndarray,
                       axis: str, spacing: tuple[float, float, float]
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Rescale sagittal/coronal so pixel aspect matches physical aspect.

    Volume axis order is (H_y, W_x, D_z) with spacing (sx, sy, sz).
    For axial we already have isotropic in-plane pixels; for sag/cor
    the D axis is much sparser, so we stretch it by sz/sx.
    """
    sx, sy, sz = spacing
    if axis == "axial":
        return img2d, mask2d
    h, w = img2d.shape
    if axis == "sagittal":   # shape (H, D), stretch W (D-axis) by sz/sy
        target_w = max(1, int(round(w * sz / sy)))
        target_h = h
    else:                    # coronal: shape (W, D), stretch W (D-axis) by sz/sx
        target_w = max(1, int(round(w * sz / sx)))
        target_h = h
    img_pil = Image.fromarray(img2d).resize((target_w, target_h), Image.BILINEAR)
    msk_pil = Image.fromarray(mask2d).resize((target_w, target_h), Image.NEAREST)
    return np.array(img_pil), np.array(msk_pil)


def _mask_outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """Binary outline of a 2D mask via dilation - erosion (no scipy dep).

    Uses a few cycles of max/min via PIL filters; cheap for 512x512.
    Returns uint8 0/1 array of same shape.
    """
    if not mask.any():
        return mask
    m_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255, "L")
    dil = m_pil.filter(ImageFilter.MaxFilter(2 * thickness + 1))
    ero = m_pil.filter(ImageFilter.MinFilter(2 * thickness + 1))
    out = (np.array(dil) > 0) & (np.array(ero) == 0)
    return out.astype(np.uint8)


def render_slice(modality: str, subject: str, axis: str, idx: int,
                 mask_alpha: float = 0.4,
                 wl: Optional[float] = None,
                 ww: Optional[float] = None,
                 outline: bool = False,
                 mode: str = "both") -> Image.Image:
    """Return a PIL RGB image of the requested slice.

    mode: "both"       — grayscale image + mask overlay (default)
          "image_only" — grayscale image, mask hidden
          "mask_only"  — black background, mask at full alpha (still respects outline)
    """
    image_vol, mask_vol = pd.get_volume(modality, subject)
    img2d, msk2d = _slice_arrays(image_vol, mask_vol, axis, idx)
    img2d = _apply_window(img2d, wl, ww)

    spacing = pd.get_spacing(modality)
    img2d, msk2d = _scale_for_display(img2d, msk2d, axis, spacing)

    if mode == "mask_only":
        img2d = np.zeros_like(img2d)
        effective_alpha = 1.0
    elif mode == "image_only":
        effective_alpha = 0.0
    else:
        effective_alpha = mask_alpha

    # Compose RGBA: grayscale base + mask overlay
    base = np.stack([img2d, img2d, img2d, np.full_like(img2d, 255)], axis=-1)
    base_pil = Image.fromarray(base, "RGBA")

    if effective_alpha > 0 and msk2d.any():
        overlay = np.zeros((*msk2d.shape, 4), dtype=np.uint8)
        if outline:
            m = _mask_outline(msk2d, thickness=2) > 0
            # Outline stays at full overlay alpha so it remains visible
            overlay[m] = MASK_RGBA
        else:
            m = msk2d > 0
            overlay[m] = MASK_RGBA
            overlay[..., 3] = (overlay[..., 3].astype(np.float32) * effective_alpha
                               ).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay, "RGBA")
        base_pil = Image.alpha_composite(base_pil, overlay_pil)

    return base_pil.convert("RGB")


def mask_area_profile(modality: str, subject: str,
                      axis: str = "axial") -> list[int]:
    """Per-slice mask voxel count along the requested axis.

    Returns a list of length axis_length. Used for the sidebar sparkline.
    """
    _, mask_vol = pd.get_volume(modality, subject)
    if axis == "axial":
        # sum over (H, W) → vector of length D
        return mask_vol.reshape(-1, mask_vol.shape[2]).sum(axis=0).astype(int).tolist()
    if axis == "sagittal":
        return mask_vol.sum(axis=(0, 2)).astype(int).tolist()
    if axis == "coronal":
        return mask_vol.sum(axis=(1, 2)).astype(int).tolist()
    raise ValueError(f"unknown axis: {axis}")


def render_thumbnail(modality: str, subject: str,
                     max_dim: int = 128) -> Image.Image:
    """Small mid-axial-slice thumbnail with mask overlay for sidebar."""
    image_vol, _ = pd.get_volume(modality, subject)
    mid = image_vol.shape[2] // 2
    img = render_slice(modality, subject, "axial", mid, mask_alpha=0.35)
    img.thumbnail((max_dim, max_dim), Image.BILINEAR)
    return img


# ── NIfTI export for NiiVue ──

# Level 1 for the grayscale volume: on quantized image data it lands within 2 %
# of level 9 while compressing ~3x faster. Masks are the opposite case — sparse
# label runs, where level 9 is 3x SMALLER (66 KB vs 214 KB) and costs
# milliseconds on a volume that size. Pick per kind, never one level for both.
_NIFTI_GZIP_LEVEL = 1
_NIFTI_MASK_GZIP_LEVEL = 9
_NIFTI_CACHE_MAX = 4
_NIFTI_CACHE: dict[tuple[str, str, str], bytes] = {}


def to_nifti_bytes(modality: str, subject: str, kind: str = "image") -> bytes:
    """Serialize a volume as NIfTI-1 .nii.gz bytes (for NiiVue).

    kind='image' returns the grayscale MRI volume,
    kind='mask'  returns the binary placenta mask.
    """
    import nibabel as nib

    key = (modality, subject, kind)
    cached = _NIFTI_CACHE.get(key)
    if cached is not None:
        return cached

    image_vol, mask_vol = pd.get_volume(modality, subject)
    vol = image_vol if kind == "image" else mask_vol
    # NIfTI canonical orientation: (X, Y, Z). Our arrays are (H_y, W_x, D_z),
    # so transpose to (W_x, H_y, D_z) and flip Y for radiological display.
    arr = np.transpose(vol, (1, 0, 2))[:, ::-1, :].copy()

    sx, sy, sz = pd.get_spacing(modality)
    affine = np.diag([sx, sy, sz, 1.0])
    nii = nib.Nifti1Image(arr, affine)
    nii.header.set_xyzt_units("mm")
    if kind == "image":
        # Scaled int16 on the wire — see the note in cv_viewer.to_nifti_bytes.
        # The mask is binary and stays exact.
        nii.set_data_dtype(np.int16)

    import gzip
    level = _NIFTI_GZIP_LEVEL if kind == "image" else _NIFTI_MASK_GZIP_LEVEL
    data = gzip.compress(nii.to_bytes(), level)
    _NIFTI_CACHE[key] = data
    while len(_NIFTI_CACHE) > _NIFTI_CACHE_MAX:
        _NIFTI_CACHE.pop(next(iter(_NIFTI_CACHE)))
    return data
