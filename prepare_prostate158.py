"""Preprocess prostate158 (Adams et al. 2022) to a common 256x256 axial grid.

The dataset has per-patient variable shapes (270x270, 442x442, 232x232, etc.)
and 3 distinct in-plane spacings (0.27, 0.40, 0.47 mm). Slice thickness is
uniformly 3 mm. All four modalities (T2, ADC, DWI, anatomy_reader1 mask) are
already on the same per-patient grid, so resampling one applies to all.

Output: `prostate158_preprocessed/{t2,adc,dwi,t2_anatomy}/case_XXX.nii.gz`
on a 256x256 in-plane grid at 0.5 mm pixel spacing, with native slice count.

Splits follow the dataset's own train.csv / valid.csv + the separate test set.
Patient IDs are zero-padded 3-digit strings (e.g. "024").
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk


SRC_TRAIN = Path("/home/marija/Desktop/ferit/ROBOMED/dataset/prostate158_train")
SRC_TEST  = Path("/home/marija/Desktop/ferit/ROBOMED/dataset/prostate158_test/prostate158_test")
OUT_DIR   = Path(__file__).parent / "prostate158_preprocessed"

# Target in-plane spacing + size. Keep native slice thickness (3 mm) since all
# tasks are 2D axial. Z axis spacing stays whatever the patient had natively.
TARGET_SPACING_XY = 0.5     # mm
TARGET_SIZE_XY    = 256

# Four modalities to materialize. Mask is integer; the rest are float intensity.
MODALITIES = {
    "t2":         {"src": "t2.nii.gz",                    "kind": "intensity"},
    "adc":        {"src": "adc.nii.gz",                   "kind": "intensity"},
    "dwi":        {"src": "dwi.nii.gz",                   "kind": "intensity"},
    "t2_anatomy": {"src": "t2_anatomy_reader1.nii.gz",    "kind": "mask"},
}


# ── Resampling ────────────────────────────────────────────────────────────

def _resample_inplane(itk_img: sitk.Image, is_mask: bool) -> sitk.Image:
    """Resample to TARGET_SPACING_XY in x/y, keep native z spacing.

    Then center-crop / pad to TARGET_SIZE_XY x TARGET_SIZE_XY.
    """
    orig_spacing = itk_img.GetSpacing()        # (sx, sy, sz)
    orig_size    = itk_img.GetSize()           # (X, Y, Z)
    sx_old, sy_old, sz_old = orig_spacing

    # New size to preserve physical extent at the new spacing
    new_sx = int(round(orig_size[0] * sx_old / TARGET_SPACING_XY))
    new_sy = int(round(orig_size[1] * sy_old / TARGET_SPACING_XY))
    new_sz = orig_size[2]

    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([new_sx, new_sy, new_sz])
    resampler.SetOutputSpacing([TARGET_SPACING_XY, TARGET_SPACING_XY, sz_old])
    resampler.SetOutputOrigin(itk_img.GetOrigin())
    resampler.SetOutputDirection(itk_img.GetDirection())
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(0)
    resampled = resampler.Execute(itk_img)

    # Center-crop / pad in x/y to TARGET_SIZE_XY
    arr = sitk.GetArrayFromImage(resampled)  # (Z, Y, X)
    Z, Y, X = arr.shape
    # X axis
    if X >= TARGET_SIZE_XY:
        x0 = (X - TARGET_SIZE_XY) // 2
        arr = arr[:, :, x0:x0 + TARGET_SIZE_XY]
    else:
        pad = TARGET_SIZE_XY - X
        l = pad // 2; r = pad - l
        arr = np.pad(arr, ((0, 0), (0, 0), (l, r)))
    # Y axis
    if Y >= TARGET_SIZE_XY:
        y0 = (Y - TARGET_SIZE_XY) // 2
        arr = arr[:, y0:y0 + TARGET_SIZE_XY, :]
    else:
        pad = TARGET_SIZE_XY - Y
        t = pad // 2; b = pad - t
        arr = np.pad(arr, ((0, 0), (t, b), (0, 0)))
    return arr.astype(np.float32) if not is_mask else arr.astype(np.int16)


def _save_nifti(arr_zyx: np.ndarray, path: Path, sz_mm: float):
    """Write (Z, Y, X) → NIfTI as (X, Y, Z) with the supplied z spacing."""
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    affine = np.diag([TARGET_SPACING_XY, TARGET_SPACING_XY, sz_mm, 1.0])
    nii = nib.Nifti1Image(arr_xyz, affine)
    nii.header.set_xyzt_units("mm")
    nib.save(nii, str(path))


# ── Per-patient pipeline ──────────────────────────────────────────────────

def _process_patient(patient_id: str, src_dir: Path) -> bool:
    """Resample + crop all 4 modalities for one patient. Returns True on success."""
    has_all = all((src_dir / cfg["src"]).exists() for cfg in MODALITIES.values())
    if not has_all:
        missing = [k for k, cfg in MODALITIES.items() if not (src_dir / cfg["src"]).exists()]
        print(f"  [{patient_id}] missing: {missing} — skipped")
        return False

    # Resample each modality. Use SimpleITK with the same interp logic per kind.
    out_paths = {}
    z_spacing = None
    for key, cfg in MODALITIES.items():
        src = src_dir / cfg["src"]
        itk = sitk.ReadImage(str(src))
        if z_spacing is None:
            z_spacing = itk.GetSpacing()[2]
        arr = _resample_inplane(itk, is_mask=(cfg["kind"] == "mask"))
        out_dir = OUT_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"case_{patient_id}.nii.gz"
        _save_nifti(arr, out_path, sz_mm=z_spacing)
        out_paths[key] = out_path

    return True


# ── Driver ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N patients (smoke testing).")
    ap.add_argument("--force", action="store_true",
                    help="Reprocess even if outputs exist.")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # Build patient list: train + valid (from train.csv & valid.csv) + test
    train_csv = SRC_TRAIN / "train.csv"
    valid_csv = SRC_TRAIN / "valid.csv"
    splits = {"train": [], "val": [], "test": []}

    def _id3(s: str) -> str:
        return str(int(s)).zfill(3)

    for csv_path, split in [(train_csv, "train"), (valid_csv, "val")]:
        with open(csv_path, newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                splits[split].append(_id3(row["ID"]))

    # Test set: scan the test folder
    test_dir = SRC_TEST / "test"
    if test_dir.exists():
        for d in sorted(test_dir.iterdir()):
            if d.is_dir():
                splits["test"].append(_id3(d.name))

    all_pairs = [(pid, "train", SRC_TRAIN / "train" / pid) for pid in splits["train"]] + \
                [(pid, "val",   SRC_TRAIN / "train" / pid) for pid in splits["val"]] + \
                [(pid, "test",  test_dir / pid) for pid in splits["test"]]

    if args.limit:
        all_pairs = all_pairs[:args.limit]

    print(f"[prep] {len(splits['train'])} train + {len(splits['val'])} val + "
          f"{len(splits['test'])} test = {len(all_pairs)} patients total")
    print(f"[prep] target grid: {TARGET_SIZE_XY}x{TARGET_SIZE_XY} @ "
          f"{TARGET_SPACING_XY} mm (in-plane); native z spacing preserved\n")

    summary = []
    n_ok = 0
    for i, (pid, split, src_dir) in enumerate(all_pairs):
        # Skip if already processed (one canonical output per patient is t2)
        out_t2 = OUT_DIR / "t2" / f"case_{pid}.nii.gz"
        if out_t2.exists() and not args.force:
            print(f"  [{i+1:3d}/{len(all_pairs)}] {pid} ({split}) — already done")
            n_ok += 1
            summary.append({"patient": pid, "split": split, "status": "cached"})
            continue

        try:
            ok = _process_patient(pid, src_dir)
            status = "ok" if ok else "skipped_missing"
        except Exception as e:
            status = f"failed: {type(e).__name__}: {e}"
            ok = False

        print(f"  [{i+1:3d}/{len(all_pairs)}] {pid} ({split}) — {status}")
        if ok:
            n_ok += 1
        summary.append({"patient": pid, "split": split, "status": status})

    # Write split manifest + summary
    manifest = {
        "splits": splits,
        "target_size_xy": TARGET_SIZE_XY,
        "target_spacing_xy_mm": TARGET_SPACING_XY,
        "summary": summary,
        "n_ok": n_ok,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[done] {n_ok}/{len(all_pairs)} patients processed → {OUT_DIR}")


if __name__ == "__main__":
    main()
