"""One-time preprocessing for the TopBrain (MICCAI 2025) dataset.

Per-patient pipeline:
  1. Load real-world CT + MR volumes (NIfTI, different grids).
  2. Rigid-register CT to MR using Mattes mutual information.
  3. Resample CT volume + CT mask to MR grid.
  4. Resample ALL volumes + masks to a common isotropic grid (0.5 mm³)
     of fixed shape (256, 256, 256) centered on the MR's brain volume.
  5. Save preprocessed NIfTIs to topbrain_preprocessed/{ct,mr,ct_mask,mr_mask}/.

Why this matters:
  - The raw release ships CT and MR at different spacings and orientations.
    Any paired training (pix2pix, conditional LDM) needs them in the same grid.
  - 0.5 mm isotropic = midway between CT's 0.5 mm in-plane / 0.625 mm slice
    and MR's 0.3 mm in-plane / 0.6 mm slice. Captures fine vessels without
    blowing up volume size.
  - Fixed shape simplifies downstream 2D slicing + batching.

Run:
    python prepare_topbrain.py
        [--src /home/marija/Documents/TopBrain_Data_Release_Batches1n2_081425]
        [--out topbrain_preprocessed]
        [--iso 0.5] [--shape 256]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _read(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def _save(img: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def _rigid_register(fixed: sitk.Image, moving: sitk.Image
                    ) -> sitk.Transform:
    """Multi-resolution rigid registration with Mattes MI."""
    fx = sitk.Cast(fixed, sitk.sitkFloat32)
    mv = sitk.Cast(moving, sitk.sitkFloat32)

    initial_tx = sitk.CenteredTransformInitializer(
        fx, mv, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.25, seed=0)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_tx, inPlace=False)
    return reg.Execute(fx, mv)


def _resample(moving: sitk.Image, fixed: sitk.Image, tx: sitk.Transform,
              nearest: bool = False, default_val: float = 0.0) -> sitk.Image:
    """Resample `moving` onto `fixed`'s grid using `tx`."""
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    out_type = moving.GetPixelID()
    return sitk.Resample(moving, fixed, tx, interp, default_val, out_type)


def _to_isotropic(img: sitk.Image, iso_mm: float, shape: int,
                  nearest: bool = False) -> sitk.Image:
    """Resample to a fixed isotropic grid (shape, shape, shape) centered on
    the image's bounding-box center, with `iso_mm` spacing."""
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    center = np.array(img.TransformContinuousIndexToPhysicalPoint(
        [(s - 1) / 2.0 for s in img.GetSize()]))
    new_size = (shape, shape, shape)
    new_spacing = (iso_mm, iso_mm, iso_mm)
    # Origin so that the center of the new volume = center of input
    half_extent = np.array(new_spacing) * (np.array(new_size) - 1) / 2.0
    new_origin = tuple(center - half_extent)
    return sitk.Resample(img,
                         size=new_size,
                         transform=sitk.Transform(),
                         interpolator=interp,
                         outputOrigin=new_origin,
                         outputSpacing=new_spacing,
                         outputDirection=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                         defaultPixelValue=0,
                         outputPixelType=img.GetPixelID())


def process_patient(pat_id: str, src_root: Path, out_root: Path,
                    iso_mm: float, shape: int):
    ct_img_p = src_root / "imagesTr_topbrain_ct" / f"topcow_ct_{pat_id}_0000.nii.gz"
    mr_img_p = src_root / "imagesTr_topbrain_mr" / f"topcow_mr_{pat_id}_0000.nii.gz"
    ct_msk_p = src_root / "labelsTr_topbrain_ct" / f"topcow_ct_{pat_id}.nii.gz"
    mr_msk_p = src_root / "labelsTr_topbrain_mr" / f"topcow_mr_{pat_id}.nii.gz"

    ct = _read(ct_img_p)
    mr = _read(mr_img_p)
    ct_msk = _read(ct_msk_p)
    mr_msk = _read(mr_msk_p)

    # 1) Rigid-register CT → MR space (Mattes MI on CT(=moving) → MR(=fixed))
    tx = _rigid_register(fixed=mr, moving=ct)

    # 2) Resample CT image+mask onto MR's grid
    ct_in_mr = _resample(ct,     mr, tx, nearest=False, default_val=-1024)
    ctm_in_mr = _resample(ct_msk, mr, tx, nearest=True,  default_val=0)

    # 3) Resample ALL four to a common isotropic grid centered on MR
    mr_iso  = _to_isotropic(mr,        iso_mm, shape, nearest=False)
    ct_iso  = _to_isotropic(ct_in_mr,  iso_mm, shape, nearest=False)
    mrm_iso = _to_isotropic(mr_msk,    iso_mm, shape, nearest=True)
    ctm_iso = _to_isotropic(ctm_in_mr, iso_mm, shape, nearest=True)

    # 4) Save
    _save(mr_iso,  out_root / "mr"      / f"topcow_{pat_id}.nii.gz")
    _save(ct_iso,  out_root / "ct"      / f"topcow_{pat_id}.nii.gz")
    _save(mrm_iso, out_root / "mr_mask" / f"topcow_{pat_id}.nii.gz")
    _save(ctm_iso, out_root / "ct_mask" / f"topcow_{pat_id}.nii.gz")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path,
                   default=Path("/home/marija/Documents/TopBrain_Data_Release_Batches1n2_081425"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/marija/Desktop/ferit/ROBOMED/godina1/wsi_app/topbrain_preprocessed"))
    p.add_argument("--iso", type=float, default=0.5,
                   help="Isotropic spacing in mm (default 0.5)")
    p.add_argument("--shape", type=int, default=256,
                   help="Cubic side length in voxels (default 256)")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated patient IDs to process (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.src.exists():
        raise FileNotFoundError(f"Source dataset not found: {args.src}")
    pats = sorted([p.name.split("_")[2] for p in
                   (args.src / "imagesTr_topbrain_ct").glob("topcow_ct_*_0000.nii.gz")])
    if args.only:
        wanted = set(args.only.split(","))
        pats = [p for p in pats if p in wanted]
    print(f"[prep] {len(pats)} patients · iso={args.iso}mm · "
          f"shape={args.shape}³ → {args.out}")
    t0 = time.time()
    for i, pid in enumerate(pats, 1):
        t1 = time.time()
        try:
            process_patient(pid, args.src, args.out, args.iso, args.shape)
        except Exception as e:
            print(f"  [{i:2d}/{len(pats)}] sub{pid}: FAILED: {e}", flush=True)
            continue
        print(f"  [{i:2d}/{len(pats)}] sub{pid}: {time.time()-t1:.1f}s",
              flush=True)
    print(f"[prep] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
