"""Flask app — PASD (placental MRI synthesis) + TopBrain (cerebrovascular).

Originally extended from a PANDA WSI viewer; WSI parts removed 2026-06-07.
"""

import io
import json
import math
import zipfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image
import numpy as np
import torch


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════════════════
# MRI Viewer: Placenta Accreta Spectrum Disorders (PASD) dataset
# ══════════════════════════════════════════════════════════════════════════════

import mri_viewer as mri_mod
import pasd_data as pasd_data_mod


@app.route("/api/mri/subjects")
def api_mri_subjects():
    """List subjects for a modality. Query: modality=BTFE|TSE (default BTFE)."""
    modality = request.args.get("modality", "BTFE").upper()
    if modality not in pasd_data_mod.MODALITIES:
        return jsonify({"error": f"unknown modality {modality}"}), 400
    return jsonify({
        "modality": modality,
        "subjects": pasd_data_mod.list_subjects(modality),
    })


@app.route("/api/mri/<modality>/<subject>/info")
def api_mri_info(modality, subject):
    modality = modality.upper()
    try:
        img_vol, msk_vol = pasd_data_mod.get_volume(modality, subject)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    H, W, D = img_vol.shape
    return jsonify({
        "modality": modality,
        "subject": subject,
        "shape": [H, W, D],
        "axis_lengths": {"axial": D, "sagittal": W, "coronal": H},
        "spacing_mm": list(pasd_data_mod.get_spacing(modality)),
        "has_mask": bool(msk_vol.any()),
        "mask_voxel_count": int((msk_vol > 0).sum()),
    })


@app.route("/api/mri/<modality>/<subject>/slice/<axis>/<int:idx>")
def api_mri_slice(modality, subject, axis, idx):
    """Return a 2D slice as JPEG (default) or PNG, with optional mask overlay.

    Query: mask_alpha (0-1, default 0.4), wl, ww, outline (0|1),
           format (jpeg|png, default jpeg), download (0|1).
    """
    modality = modality.upper()
    if axis not in mri_mod.AXES:
        return jsonify({"error": f"axis must be one of {mri_mod.AXES}"}), 400
    try:
        n = mri_mod.axis_length(modality, subject, axis)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    if idx < 0 or idx >= n:
        return jsonify({"error": f"idx out of range [0, {n-1}]"}), 400

    mask_alpha = float(request.args.get("mask_alpha", 0.4))
    wl = request.args.get("wl")
    ww = request.args.get("ww")
    wl = float(wl) if wl is not None else None
    ww = float(ww) if ww is not None else None
    outline = request.args.get("outline", "0") == "1"
    mode = request.args.get("mode", "both")
    if mode not in ("both", "image_only", "mask_only"):
        mode = "both"
    fmt = request.args.get("format", "jpeg").lower()
    download = request.args.get("download", "0") == "1"

    img = mri_mod.render_slice(modality, subject, axis, idx,
                               mask_alpha=mask_alpha, wl=wl, ww=ww,
                               outline=outline, mode=mode)
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=88)
        mime = "image/jpeg"
    buf.seek(0)
    return send_file(buf, mimetype=mime, as_attachment=download,
                     download_name=f"{subject}_{axis}_{idx:03d}.{fmt}")


@app.route("/api/mri/<modality>/<subject>/mask_areas")
def api_mri_mask_areas(modality, subject):
    """Per-slice mask voxel counts along the requested axis (default axial)."""
    modality = modality.upper()
    axis = request.args.get("axis", "axial")
    if axis not in mri_mod.AXES:
        return jsonify({"error": f"axis must be one of {mri_mod.AXES}"}), 400
    try:
        areas = mri_mod.mask_area_profile(modality, subject, axis)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"axis": axis, "areas": areas})


@app.route("/api/mri/<modality>/<subject>/thumb")
def api_mri_thumb(modality, subject):
    modality = modality.upper()
    try:
        img = mri_mod.render_thumbnail(modality, subject)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/mri/<modality>/<subject>/<kind>.nii.gz")
def api_mri_nifti(modality, subject, kind):
    """Stream the image or mask volume as gzipped NIfTI for NiiVue."""
    modality = modality.upper()
    if kind not in ("image", "mask"):
        return jsonify({"error": "kind must be image or mask"}), 400
    try:
        data = mri_mod.to_nifti_bytes(modality, subject, kind)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    return send_file(
        io.BytesIO(data),
        mimetype="application/gzip",
        download_name=f"{subject}_{kind}.nii.gz",
    )


# ══════════════════════════════════════════════════════════════════════════════
# CTA / TOF-MRA Viewer (TopBrain dataset)
# ══════════════════════════════════════════════════════════════════════════════

import cv_viewer as cv_view_mod


@app.route("/api/cv_view/subjects")
def api_cv_view_subjects():
    return jsonify({"subjects": cv_view_mod.list_subjects()})


@app.route("/api/cv_view/labels/<modality>")
def api_cv_view_labels(modality):
    """Labelmap as JSON: {id: int, name: str, color: [r,g,b]} list."""
    modality = modality.lower()
    try:
        lmap = cv_view_mod.get_labelmap(modality)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "modality": modality,
        "labels": [{"id": k, "name": v["name"], "color": list(v["color"])}
                   for k, v in sorted(lmap.items())],
    })


@app.route("/api/cv_view/windows")
def api_cv_view_windows():
    return jsonify({
        "presets": cv_view_mod.CT_WINDOW_PRESETS,
        "default": {"wl": cv_view_mod.DEFAULT_CT_WL,
                    "ww": cv_view_mod.DEFAULT_CT_WW},
    })


@app.route("/api/cv_view/<modality>/<subject>/info")
def api_cv_view_info(modality, subject):
    modality = modality.lower()
    if modality not in cv_view_mod.MODALITIES:
        return jsonify({"error": f"unknown modality {modality}"}), 400
    try:
        import cv_data as cd
        vol = cd._vol(subject, modality)
        mask = cd._vol(subject, f"{modality}_mask")
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    H, W, D = vol.shape
    import numpy as np
    nz_count = int((mask > 0).sum())
    return jsonify({
        "modality": modality,
        "subject": subject,
        "shape": [H, W, D],
        "axis_lengths": {"axial": D, "sagittal": W, "coronal": H},
        "spacing_mm": list(cv_view_mod.SPACING),
        "mask_voxel_count": nz_count,
        "unique_labels": sorted(set(int(x) for x in np.unique(mask)) - {0}),
    })


@app.route("/api/cv_view/<modality>/<subject>/slice/<axis>/<int:idx>")
def api_cv_view_slice(modality, subject, axis, idx):
    modality = modality.lower()
    if axis not in cv_view_mod.AXES:
        return jsonify({"error": f"axis must be one of {cv_view_mod.AXES}"}), 400
    try:
        n = cv_view_mod.axis_length(subject, axis)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404
    if idx < 0 or idx >= n:
        return jsonify({"error": f"idx out of range [0, {n-1}]"}), 400

    mask_alpha = float(request.args.get("mask_alpha", 0.5))
    wl = request.args.get("wl")
    ww = request.args.get("ww")
    wl = float(wl) if wl not in (None, "", "auto") else None
    ww = float(ww) if ww not in (None, "", "auto") else None
    outline = request.args.get("outline", "0") == "1"
    mode = request.args.get("mode", "both")
    if mode not in ("both", "image_only", "mask_only"):
        mode = "both"
    fmt = request.args.get("format", "jpeg").lower()
    download = request.args.get("download", "0") == "1"

    img = cv_view_mod.render_slice(modality, subject, axis, idx,
                                    mask_alpha=mask_alpha, wl=wl, ww=ww,
                                    outline=outline, mode=mode)
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    buf.seek(0)
    return send_file(buf, mimetype=mime, as_attachment=download,
                     download_name=f"{subject}_{modality}_{axis}_{idx:03d}.{fmt}")


@app.route("/api/cv_view/<modality>/<subject>/mask_areas")
def api_cv_view_mask_areas(modality, subject):
    modality = modality.lower()
    axis = request.args.get("axis", "axial")
    if axis not in cv_view_mod.AXES:
        return jsonify({"error": f"axis must be one of {cv_view_mod.AXES}"}), 400
    try:
        areas = cv_view_mod.mask_area_profile(modality, subject, axis)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404
    return jsonify({"axis": axis, "modality": modality, "areas": areas})


@app.route("/api/cv_view/<modality>/<subject>/thumb")
def api_cv_view_thumb(modality, subject):
    modality = modality.lower()
    try:
        img = cv_view_mod.render_thumbnail(modality, subject)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/cv_view/<modality>/<subject>/<kind>.nii.gz")
def api_cv_view_nifti(modality, subject, kind):
    modality = modality.lower()
    if kind not in ("image", "mask"):
        return jsonify({"error": "kind must be image or mask"}), 400
    try:
        data = cv_view_mod.to_nifti_bytes(modality, subject, kind)
    except (FileNotFoundError, ValueError) as e:
        return jsonify({"error": str(e)}), 404
    return send_file(
        io.BytesIO(data),
        mimetype="application/gzip",
        download_name=f"{subject}_{modality}_{kind}.nii.gz",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PASD MRI Generation (Modules tab)
# ══════════════════════════════════════════════════════════════════════════════

import generation as pasd_gen

# Per-request session cache: holds last-generated samples so the UI can fetch
# each by index. Keyed by short opaque id, capped to ~32 sessions.
_pasd_gen_sessions: dict[str, dict] = {}
_PASD_GEN_MAX_SESSIONS = 32


def _pasd_gen_evict():
    while len(_pasd_gen_sessions) > _PASD_GEN_MAX_SESSIONS:
        _pasd_gen_sessions.pop(next(iter(_pasd_gen_sessions)))


@app.route("/api/pasd_gen/status")
def api_pasd_gen_status():
    """Which (modality, model) combos have trained weights available."""
    return jsonify(pasd_gen.available_checkpoints())


@app.route("/api/pasd_gen/sample", methods=["POST"])
def api_pasd_gen_sample():
    """Generate synthetic MRI slice(s) from a mask.

    JSON body:
      modality: "BTFE" | "TSE"
      model:    "pix2pix"
      num_samples: int (default 1)
      seed: int | null
      mask_source: "dataset" | "upload"
      # if dataset: subject, slice_idx (the real GT mask is used)
      # if upload : mask_b64 (base64 PNG)
    Returns: {session_id, samples: [{id, modality, model}], width, height}
    """
    import base64, secrets

    data = request.get_json(force=True) or {}
    modality = data.get("modality", "BTFE").upper()
    model = data.get("model", "pix2pix")
    num_samples = max(1, min(int(data.get("num_samples", 1)), 16))
    seed = data.get("seed")
    seed = int(seed) if seed is not None and seed != "" else None
    source = data.get("mask_source", "dataset")

    # Resolve mask
    if source == "dataset":
        subject = data.get("subject")
        slice_idx = int(data.get("slice_idx", 0))
        if not subject:
            return jsonify({"error": "subject required for mask_source=dataset"}), 400
        rows = [r for r in pasd_data_mod.build_index()
                if r["modality"] == modality and r["subject"] == subject
                and r["slice_idx"] == slice_idx]
        if not rows or not rows[0]["mask"]:
            return jsonify({"error": "mask not found"}), 404
        mask = Image.open(pasd_data_mod.ROOT / rows[0]["mask"])
    elif source == "upload":
        b64 = data.get("mask_b64", "")
        if not b64:
            return jsonify({"error": "mask_b64 required for upload"}), 400
        # Strip data URL prefix if present
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            mask = Image.open(io.BytesIO(base64.b64decode(b64)))
        except Exception as e:
            return jsonify({"error": f"failed to decode mask: {e}"}), 400
    else:
        return jsonify({"error": f"unknown mask_source: {source}"}), 400

    perturb = bool(data.get("perturb", False))
    # LDM-specific (ignored for pix2pix)
    num_inference_steps = int(data.get("num_inference_steps", 25))
    guidance_scale = float(data.get("guidance_scale", 1.5))

    try:
        if perturb:
            pairs = pasd_gen.sample_paired(
                mask, modality=modality, model=model,
                num_samples=num_samples, seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale)
            sample_imgs = [p[1] for p in pairs]
            sample_masks = [p[0] for p in pairs]
        else:
            sample_imgs = pasd_gen.sample(
                mask, modality=modality, model=model,
                num_samples=num_samples, seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale)
            sample_masks = None
    except FileNotFoundError as e:
        return jsonify({"error": f"no trained model: {e}"}), 503

    sid = secrets.token_hex(8)
    _pasd_gen_sessions[sid] = {
        "samples": sample_imgs,
        "masks":   sample_masks,        # None when perturb=False (shared mask)
        "shared_mask": mask if sample_masks is None else None,
        "modality": modality,
        "model": model,
        "source": source,
        "subject": data.get("subject"),
        "slice_idx": data.get("slice_idx"),
        # populated below
        "real_image": None,
        "metrics": None,
    }
    _pasd_gen_evict()

    # Display URLs depend on the source + perturb mode.
    mask_url = None     # used only when perturb=False (single shared mask)
    real_url = None
    sample_mask_urls = None
    if perturb:
        sample_mask_urls = [f"/api/pasd_gen/sample_mask/{sid}/{i}"
                            for i in range(len(sample_imgs))]
    elif source == "dataset":
        mask_url = (f"/api/pasd_gen/mask_preview?modality={modality}"
                    f"&subject={subject}&slice_idx={slice_idx}")
        sub_rows = sorted(
            (r for r in pasd_data_mod.build_index()
             if r["modality"] == modality and r["subject"] == subject),
            key=lambda r: r["slice_idx"],
        )
        positions = {r["slice_idx"]: i for i, r in enumerate(sub_rows)}
        pos = positions.get(slice_idx)
        if pos is not None:
            real_url = (f"/api/mri/{modality}/{subject}"
                        f"/slice/axial/{pos}?mask_alpha=0")

    # ── Per-sample metrics (mask Dice, plus SSIM/PSNR if real available) ──
    real_for_metrics: Image.Image | None = None
    if source == "dataset":
        # Pull the real slice image from the assembled volume to match the synth.
        sub_rows_for_real = sorted(
            (r for r in pasd_data_mod.build_index()
             if r["modality"] == modality and r["subject"] == subject),
            key=lambda r: r["slice_idx"],
        )
        positions_for_real = {r["slice_idx"]: i for i, r in enumerate(sub_rows_for_real)}
        pos_for_real = positions_for_real.get(slice_idx)
        if pos_for_real is not None:
            real_for_metrics = mri_mod.render_slice(
                modality, subject, "axial", pos_for_real, mask_alpha=0.0)
    metrics = []
    for i, sim in enumerate(sample_imgs):
        msk_for_metric = sample_masks[i] if sample_masks is not None else mask
        try:
            m = pasd_gen.compute_sample_metrics(
                sim, msk_for_metric, real_pil=real_for_metrics,
                modality=modality)
        except Exception as e:
            m = {"error": str(e)}
        metrics.append(m)

    # Persist real + metrics into the session so the zip endpoint can find them
    _pasd_gen_sessions[sid]["real_image"] = real_for_metrics
    _pasd_gen_sessions[sid]["metrics"] = metrics

    return jsonify({
        "session_id": sid,
        "modality": modality,
        "model": model,
        "mask_source": source,
        "perturb": perturb,
        "subject": data.get("subject"),
        "slice_idx": data.get("slice_idx"),
        "mask_url": mask_url,
        "real_url": real_url,
        "sample_mask_urls": sample_mask_urls,
        "num_samples": len(sample_imgs),
        "width": sample_imgs[0].width if sample_imgs else 0,
        "height": sample_imgs[0].height if sample_imgs else 0,
        "samples": [{"id": i, "metrics": metrics[i]} for i in range(len(sample_imgs))],
    })


@app.route("/api/pasd_gen/sample_image/<session_id>/<int:idx>")
def api_pasd_gen_sample_image(session_id, idx):
    """Stream a previously-generated sample as JPEG (or PNG with ?format=png)."""
    s = _pasd_gen_sessions.get(session_id)
    if not s or idx < 0 or idx >= len(s["samples"]):
        return jsonify({"error": "session/sample not found"}), 404
    fmt = request.args.get("format", "jpeg").lower()
    download = request.args.get("download", "0") == "1"
    buf = io.BytesIO()
    if fmt == "png":
        s["samples"][idx].save(buf, format="PNG")
        mime = "image/png"; ext = "png"
    else:
        s["samples"][idx].save(buf, format="JPEG", quality=92)
        mime = "image/jpeg"; ext = "jpg"
    buf.seek(0)
    return send_file(buf, mimetype=mime, as_attachment=download,
                     download_name=f"synthetic_{idx:02d}.{ext}")


# ── Cerebrovascular: TopBrain MRA → CTA (pix2pix) ──

import cv_data as cv_data_mod

_cv_sessions: dict[str, dict] = {}


@app.route("/api/cv/patients")
def api_cv_patients():
    """Return list of TopBrain patients with per-slice counts."""
    rows = cv_data_mod.build_index()
    out = []
    for r in rows:
        n = int(r["n_slices"])
        out.append({
            "patient": r["patient"],
            "n_slices": n,
            "first_z": int(r["first_z"]),
            "last_z": int(r["last_z"]),
        })
    return jsonify({"patients": out})


@app.route("/api/cv/<patient>/slice/<kind>/<int:z>")
def api_cv_slice(patient, kind, z):
    """Return a 2D axial slice as JPEG. kind ∈ {mr, ct, mr_mask, ct_mask}."""
    if kind not in ("mr", "ct", "mr_mask", "ct_mask"):
        return jsonify({"error": "kind must be mr|ct|mr_mask|ct_mask"}), 400
    try:
        vol = cv_data_mod._vol(patient, kind)
    except Exception as e:
        return jsonify({"error": f"slice not found: {e}"}), 404
    if z < 0 or z >= vol.shape[2]:
        return jsonify({"error": f"z out of range [0, {vol.shape[2]-1}]"}), 400
    arr = vol[:, :, z]
    if kind in ("mr_mask", "ct_mask"):
        # Map labels to a uint8 image (just the raw label values, capped)
        arr = arr.clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr, "L")
    elif kind == "mr":
        # Per-slice normalization using volume's 99th percentile
        p99 = float(np.percentile(vol[vol > 0], 99)) if vol.max() > 0 else 1.0
        norm = np.clip(arr / max(p99, 1e-6), 0, 1) * 255
        pil = Image.fromarray(norm.astype(np.uint8), "L")
    else:  # ct
        norm = np.clip((arr - cv_data_mod.CT_CLAMP[0]) /
                       (cv_data_mod.CT_CLAMP[1] - cv_data_mod.CT_CLAMP[0]),
                       0, 1) * 255
        pil = Image.fromarray(norm.astype(np.uint8), "L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/cv/translate", methods=["POST"])
def api_cv_translate():
    """Run MRA→CTA translation on a slice.

    JSON body: {patient, z}   OR   {image_b64}
    """
    import base64, secrets
    data = request.get_json(force=True) or {}
    if data.get("image_b64"):
        b64 = data["image_b64"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            input_pil = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
        except Exception as e:
            return jsonify({"error": f"decode failed: {e}"}), 400
        pat = z = None
    else:
        pat = data.get("patient")
        z = data.get("z")
        if pat is None or z is None:
            return jsonify({"error": "patient + z required"}), 400
        try:
            vol = cv_data_mod._vol(pat, "mr")
        except Exception as e:
            return jsonify({"error": str(e)}), 404
        z = int(z)
        if z < 0 or z >= vol.shape[2]:
            return jsonify({"error": "z out of range"}), 400
        arr = vol[:, :, z]
        p99 = float(np.percentile(vol[vol > 0], 99)) if vol.max() > 0 else 1.0
        norm = np.clip(arr / max(p99, 1e-6), 0, 1) * 255
        input_pil = Image.fromarray(norm.astype(np.uint8), "L")

    model = data.get("model", "pix2pix")
    try:
        if model == "ldm":
            steps = int(data.get("num_inference_steps", 25))
            cfg   = float(data.get("guidance_scale", 1.5))
            seed  = data.get("seed")
            outs = pasd_gen.cv_translate_mr2ct_ldm(
                input_pil, num_samples=1,
                num_inference_steps=steps, guidance_scale=cfg, seed=seed)
            out_pil = outs[0]
        elif model == "ldm_combined":
            # Needs a CT vessel mask. From dataset only — uses the corresponding CT mask.
            if pat is None or z is None:
                return jsonify({"error": "ldm_combined requires patient+z (needs CT mask)"}), 400
            mask_vol = cv_data_mod._vol(pat, "ct_mask")
            mask_arr = mask_vol[:, :, z].astype(np.int16)
            mask_t = torch.from_numpy(mask_arr).long()
            steps = int(data.get("num_inference_steps", 25))
            cfg   = float(data.get("guidance_scale", 1.5))
            seed  = data.get("seed")
            out_pil = pasd_gen.cv_translate_mr_mask_to_ct_ldm(
                input_pil, mask_t,
                num_inference_steps=steps, guidance_scale=cfg, seed=seed)
        elif model == "cyclegan":
            out_pil = pasd_gen.cv_translate_cyclegan(input_pil, direction="MR_to_CT")
        else:
            out_pil = pasd_gen.cv_translate_mr2ct(input_pil)
    except FileNotFoundError as e:
        return jsonify({"error": f"no trained model: {e}"}), 503

    sid = secrets.token_hex(8)
    _cv_sessions[sid] = {
        "input": input_pil.convert("RGB").resize((256, 256)),
        "output": out_pil,
        "patient": pat, "z": z,
        "model": model,
    }
    while len(_cv_sessions) > 32:
        _cv_sessions.pop(next(iter(_cv_sessions)))

    return jsonify({
        "session_id": sid,
        "patient": pat,
        "z": z,
        "input_url":  f"/api/cv/session/{sid}/input",
        "output_url": f"/api/cv/session/{sid}/output",
    })


@app.route("/api/cv/spade_generate", methods=["POST"])
def api_cv_spade_generate():
    """Generate a CTA slice from a multi-class vessel mask via SPADE.

    JSON body: {patient, z, seed?}
    """
    import secrets, torch
    data = request.get_json(force=True) or {}
    pat = data.get("patient"); z = data.get("z")
    if pat is None or z is None:
        return jsonify({"error": "patient + z required"}), 400
    try:
        mask_vol = cv_data_mod._vol(pat, "ct_mask")
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    z = int(z)
    if z < 0 or z >= mask_vol.shape[2]:
        return jsonify({"error": "z out of range"}), 400
    mask = mask_vol[:, :, z].astype(np.int16)
    mask_t = torch.from_numpy(mask).long()
    seed = data.get("seed")
    try:
        out_pil = pasd_gen.cv_sample_mask2ct(mask_t, seed=seed)
    except FileNotFoundError as e:
        return jsonify({"error": f"no trained model: {e}"}), 503

    sid = secrets.token_hex(8)
    # Render the mask as a color preview (label IDs → grayscale)
    mask_vis_arr = np.clip(mask, 0, 40).astype(np.float32) / 40 * 255
    mask_vis_pil = Image.fromarray(mask_vis_arr.astype(np.uint8), "L")
    _cv_sessions[sid] = {
        "input": mask_vis_pil.convert("RGB").resize((256, 256)),
        "output": out_pil,
        "patient": pat, "z": z,
    }
    while len(_cv_sessions) > 32:
        _cv_sessions.pop(next(iter(_cv_sessions)))

    return jsonify({
        "session_id": sid, "patient": pat, "z": z,
        "input_url":  f"/api/cv/session/{sid}/input",
        "output_url": f"/api/cv/session/{sid}/output",
        "real_ct_url": f"/api/cv/{pat}/slice/ct/{z}",
    })


@app.route("/api/cv/session/<sid>/<which>")
def api_cv_session_image(sid, which):
    s = _cv_sessions.get(sid)
    if not s or which not in ("input", "output"):
        return jsonify({"error": "not found"}), 404
    buf = io.BytesIO()
    s[which].save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg",
                     download_name=f"cv_{which}_{sid}.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# Prostate158 — T2 → ADC (pix2pix) and (later) Anatomy Mask → T2 (SPADE)
# ══════════════════════════════════════════════════════════════════════════════

import prostate_data as prostate_data_mod

_prostate_sessions: dict[str, dict] = {}


@app.route("/api/prostate/patients")
def api_prostate_patients():
    """List patients (val + test, deduped) with per-slice counts."""
    rows = prostate_data_mod.build_index()
    val_set = set(prostate_data_mod.list_patients("val"))
    test_set = set(prostate_data_mod.list_patients("test"))
    out = []
    for r in rows:
        pid = r["patient"]
        if pid in val_set or pid in test_set:
            split = "test" if pid in test_set else "val"
            out.append({
                "patient": pid,
                "n_slices": int(r["n_slices"]),
                "z_total": int(r["z_total"]),
                "split": split,
            })
    return jsonify({"patients": out})


@app.route("/api/prostate/<patient>/info")
def api_prostate_info(patient):
    try:
        t2 = prostate_data_mod._vol(patient, "t2")
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    H, W, D = t2.shape
    return jsonify({
        "patient": patient,
        "shape": [int(H), int(W), int(D)],
        "z_total": int(D),
    })


@app.route("/api/prostate/<patient>/slice/<kind>/<int:z>")
def api_prostate_slice(patient, kind, z):
    """Render a 2D axial slice as JPEG. kind ∈ {t2, adc, dwi, t2_anatomy}."""
    if kind not in ("t2", "adc", "dwi", "t2_anatomy"):
        return jsonify({"error": "kind must be t2|adc|dwi|t2_anatomy"}), 400
    try:
        vol = prostate_data_mod._vol(patient, kind)
    except Exception as e:
        return jsonify({"error": f"slice not found: {e}"}), 404
    if z < 0 or z >= vol.shape[2]:
        return jsonify({"error": f"z out of range [0, {vol.shape[2]-1}]"}), 400
    arr = vol[:, :, z]
    if kind == "t2_anatomy":
        # 0/1/2 → 0/128/255 for visualization
        arr = (np.clip(arr, 0, 2) * 127).astype(np.uint8)
        pil = Image.fromarray(arr, "L")
    elif kind == "t2":
        p99 = float(np.percentile(vol[vol > 0], 99)) if vol.max() > 0 else 1.0
        norm = np.clip(arr / max(p99, 1e-6), 0, 1) * 255
        pil = Image.fromarray(norm.astype(np.uint8), "L")
    elif kind == "adc":
        lo, hi = prostate_data_mod.ADC_CLAMP
        norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        pil = Image.fromarray(norm.astype(np.uint8), "L")
    else:  # dwi
        p99 = float(np.percentile(vol[vol > 0], 99)) if vol.max() > 0 else 1.0
        norm = np.clip(arr / max(p99, 1e-6), 0, 1) * 255
        pil = Image.fromarray(norm.astype(np.uint8), "L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/api/prostate/translate", methods=["POST"])
def api_prostate_translate():
    """Run T2 → ADC translation on a slice.

    JSON body: {patient, z, model?}
    """
    import secrets
    data = request.get_json(force=True) or {}
    pat = data.get("patient")
    z   = data.get("z")
    if pat is None or z is None:
        return jsonify({"error": "patient + z required"}), 400
    try:
        vol = prostate_data_mod._vol(pat, "t2")
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    z = int(z)
    if z < 0 or z >= vol.shape[2]:
        return jsonify({"error": "z out of range"}), 400

    arr = vol[:, :, z]
    p99 = float(np.percentile(vol[vol > 0], 99)) if vol.max() > 0 else 1.0
    norm = np.clip(arr / max(p99, 1e-6), 0, 1) * 255
    t2_pil = Image.fromarray(norm.astype(np.uint8), "L")

    model = data.get("model", "pix2pix")
    try:
        if model == "pix2pix":
            out_pil = pasd_gen.prostate_translate_t2_to_adc(t2_pil)
        elif model == "ldm":
            steps = int(data.get("num_inference_steps", 25))
            cfg   = float(data.get("guidance_scale", 1.0))  # 1.0 Pareto-beats 1.5 on prostate
            seed  = data.get("seed")
            n_ens = int(data.get("n_ensemble", 1))
            out_pil = pasd_gen.prostate_translate_t2_to_adc_ldm(
                t2_pil, num_inference_steps=steps,
                guidance_scale=cfg, seed=seed, n_ensemble=n_ens)
        elif model == "cyclegan":
            out_pil = pasd_gen.prostate_translate_cyclegan(
                t2_pil, direction="T2_to_ADC")
        else:
            return jsonify({"error": f"model {model} not yet trained"}), 503
    except FileNotFoundError as e:
        return jsonify({"error": f"no trained model: {e}"}), 503

    sid = secrets.token_hex(8)
    _prostate_sessions[sid] = {
        "input":  t2_pil.convert("RGB").resize((256, 256)),
        "output": out_pil,
        "patient": pat, "z": z, "model": model,
    }
    while len(_prostate_sessions) > 32:
        _prostate_sessions.pop(next(iter(_prostate_sessions)))

    return jsonify({
        "session_id": sid,
        "patient": pat, "z": z, "model": model,
        "input_url":   f"/api/prostate/session/{sid}/input",
        "output_url":  f"/api/prostate/session/{sid}/output",
        "real_adc_url": f"/api/prostate/{pat}/slice/adc/{z}",
        "real_t2_url":  f"/api/prostate/{pat}/slice/t2/{z}",
    })


@app.route("/api/prostate/spade_generate", methods=["POST"])
def api_prostate_spade_generate():
    """Generate a T2 slice from a 3-class anatomy mask via SPADE.

    JSON body: {patient, z, seed?}
    """
    import secrets
    data = request.get_json(force=True) or {}
    pat = data.get("patient"); z = data.get("z")
    if pat is None or z is None:
        return jsonify({"error": "patient + z required"}), 400
    try:
        mask_vol = prostate_data_mod._vol(pat, "t2_anatomy")
    except Exception as e:
        return jsonify({"error": str(e)}), 404
    z = int(z)
    if z < 0 or z >= mask_vol.shape[2]:
        return jsonify({"error": "z out of range"}), 400
    mask = mask_vol[:, :, z].astype(np.int16)
    mask_t = torch.from_numpy(mask).long()
    seed = data.get("seed")
    try:
        out_pil = pasd_gen.prostate_sample_mask_to_t2(mask_t, seed=seed)
    except FileNotFoundError as e:
        return jsonify({"error": f"no trained model: {e}"}), 503

    sid = secrets.token_hex(8)
    mask_vis_arr = np.clip(mask, 0, 2).astype(np.float32) / 2 * 255
    mask_vis_pil = Image.fromarray(mask_vis_arr.astype(np.uint8), "L")
    _prostate_sessions[sid] = {
        "input":  mask_vis_pil.convert("RGB").resize((256, 256)),
        "output": out_pil,
        "patient": pat, "z": z,
    }
    while len(_prostate_sessions) > 32:
        _prostate_sessions.pop(next(iter(_prostate_sessions)))

    return jsonify({
        "session_id": sid, "patient": pat, "z": z,
        "input_url":   f"/api/prostate/session/{sid}/input",
        "output_url":  f"/api/prostate/session/{sid}/output",
        "real_t2_url": f"/api/prostate/{pat}/slice/t2/{z}",
        "real_mask_url": f"/api/prostate/{pat}/slice/t2_anatomy/{z}",
    })


@app.route("/api/prostate/session/<sid>/<which>")
def api_prostate_session_image(sid, which):
    s = _prostate_sessions.get(sid)
    if not s or which not in ("input", "output"):
        return jsonify({"error": "not found"}), 404
    buf = io.BytesIO()
    s[which].save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg",
                     download_name=f"prostate_{which}_{sid}.jpg")


# ══════════════════════════════════════════════════════════════════════════════
# Segmentation (TSTR): visualize segmenter predictions on real test slices
# ══════════════════════════════════════════════════════════════════════════════

import pasd_seg as seg_mod

_TSTR_DATA = Path(__file__).parent / "tstr_synth_data"
_TSTR_MODELS = Path(__file__).parent / "tstr_synth_models"
_seg_cache: dict = {}          # (dataset, mode) -> (model, mtime)
_tstr_seg_sessions: dict[str, dict] = {}


def _tstr_conditions(dataset):
    """Available trained segmenter conditions for a dataset (seed-0 dirs)."""
    out = []
    for d in sorted(_TSTR_MODELS.glob(f"{dataset}_*")):
        name = d.name[len(dataset) + 1:]
        if "_s" in name or name.endswith("_resunet"):
            continue  # skip multi-seed / alt-arch dirs in the default UI
        if (d / "best.pt").exists():
            out.append(name)
    return out


def _get_tstr_segmenter(dataset, mode, device="cuda"):
    ck = _TSTR_MODELS / f"{dataset}_{mode}" / "best.pt"
    if not ck.exists():
        raise FileNotFoundError(f"no segmenter at {ck}")
    mtime = ck.stat().st_mtime
    key = (dataset, mode)
    if key in _seg_cache and _seg_cache[key][1] == mtime:
        return _seg_cache[key][0]
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    base = int(sd.get("args", {}).get("base", 32))
    model = seg_mod.SmallUNet(in_ch=1, out_ch=1, base=base)
    model.load_state_dict(sd["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device)
    _seg_cache[key] = (model, mtime)
    return model


def _overlay(gray_arr, mask_bool, color, alpha=0.45):
    """gray uint8 (H,W) + boolean mask -> RGB PIL with colored overlay."""
    rgb = np.stack([gray_arr] * 3, axis=-1).astype(np.float32)
    col = np.array(color, dtype=np.float32)
    rgb[mask_bool] = (1 - alpha) * rgb[mask_bool] + alpha * col
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8), "RGB")


@app.route("/api/tstr/<dataset>/info")
def api_tstr_info(dataset):
    test_dir = _TSTR_DATA / f"{dataset}_real_test" / "images"
    if not test_dir.exists():
        return jsonify({"error": f"no test data for {dataset}"}), 404
    n = len(list(test_dir.glob("*.jpg")))
    return jsonify({"dataset": dataset, "n_slices": n,
                    "conditions": _tstr_conditions(dataset)})


@app.route("/api/tstr/<dataset>/segment/<int:idx>", methods=["GET", "POST"])
def api_tstr_segment(dataset, idx):
    """Run selected segmenters on real test slice idx; return overlays + Dice.

    Query/body: conditions=comma-separated (default: all available).
    """
    import secrets
    test_img = _TSTR_DATA / f"{dataset}_real_test" / "images" / f"idx{idx:05d}.jpg"
    test_msk = _TSTR_DATA / f"{dataset}_real_test" / "masks" / f"idx{idx:05d}.png"
    if not test_img.exists():
        return jsonify({"error": f"slice {idx} not found"}), 404

    gray = np.asarray(Image.open(test_img).convert("L"), dtype=np.uint8)
    gt = np.asarray(Image.open(test_msk).convert("L"), dtype=np.float32) > 127

    conds_param = (request.args.get("conditions")
                   or (request.get_json(silent=True) or {}).get("conditions"))
    conds = ([c for c in conds_param.split(",") if c] if conds_param
             else _tstr_conditions(dataset))

    x = torch.from_numpy(gray.astype(np.float32) / 255.0 * 2 - 1)[None, None]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(dev)

    sid = secrets.token_hex(8)
    panels = {"input": Image.fromarray(gray, "L").convert("RGB"),
              "gt": _overlay(gray, gt, (60, 200, 60))}
    results = [{"key": "input", "label": "Input"},
               {"key": "gt", "label": "Ground truth", "color": "#3cc83c"}]
    for mode in conds:
        try:
            model = _get_tstr_segmenter(dataset, mode, device=dev)
        except FileNotFoundError:
            continue
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)[0, 0]) > 0.5).cpu().numpy()
        inter = (pred & gt).sum()
        dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-6)
        panels[mode] = _overlay(gray, pred, (220, 70, 70))
        label = "Real-trained" if mode == "real" else mode.replace("synth_", "").capitalize()
        results.append({"key": mode, "label": label, "dice": round(float(dice), 3),
                        "color": "#dc4646"})

    _tstr_seg_sessions[sid] = panels
    while len(_tstr_seg_sessions) > 16:
        _tstr_seg_sessions.pop(next(iter(_tstr_seg_sessions)))
    return jsonify({"session_id": sid, "dataset": dataset, "idx": idx,
                    "panels": [{**r, "url": f"/api/tstr/session/{sid}/{r['key']}"}
                               for r in results]})


@app.route("/api/tstr/session/<sid>/<key>")
def api_tstr_session_image(sid, key):
    s = _tstr_seg_sessions.get(sid)
    if not s or key not in s:
        return jsonify({"error": "not found"}), 404
    buf = io.BytesIO()
    s[key].save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


# ── Cross-modality (CycleGAN BTFE ↔ TSE) ──

_pasd_xmod_sessions: dict[str, dict] = {}


@app.route("/api/pasd_xmod/translate", methods=["POST"])
def api_pasd_xmod_translate():
    """Translate a slice between modalities via CycleGAN.

    JSON body:
      direction: "BTFE_to_TSE" | "TSE_to_BTFE"
      source: "dataset" | "upload"
      # dataset: subject, slice_idx (the input modality is implied by direction)
      # upload : image_b64 (base64 image)
    Returns: { session_id, input_url, output_url }
    """
    import base64, secrets
    data = request.get_json(force=True) or {}
    direction = data.get("direction", "BTFE_to_TSE")
    if direction not in ("BTFE_to_TSE", "TSE_to_BTFE"):
        return jsonify({"error": "invalid direction"}), 400
    src_modality = "BTFE" if direction == "BTFE_to_TSE" else "TSE"
    source = data.get("source", "dataset")

    if source == "dataset":
        subject = data.get("subject")
        slice_idx = data.get("slice_idx")
        if subject is None or slice_idx is None:
            return jsonify({"error": "subject + slice_idx required"}), 400
        slice_idx = int(slice_idx)
        rows = [r for r in pasd_data_mod.build_index()
                if r["modality"] == src_modality
                and r["subject"] == subject
                and r["slice_idx"] == slice_idx]
        if not rows:
            return jsonify({"error": "slice not found"}), 404
        input_pil = Image.open(pasd_data_mod.ROOT / rows[0]["image"])
    elif source == "upload":
        b64 = data.get("image_b64", "")
        if not b64:
            return jsonify({"error": "image_b64 required"}), 400
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            input_pil = Image.open(io.BytesIO(base64.b64decode(b64)))
        except Exception as e:
            return jsonify({"error": f"decode failed: {e}"}), 400
    else:
        return jsonify({"error": "invalid source"}), 400

    try:
        translated = pasd_gen.translate(input_pil, direction=direction)
    except FileNotFoundError as e:
        return jsonify({"error": f"no trained CycleGAN: {e}"}), 503

    sid = secrets.token_hex(8)
    _pasd_xmod_sessions[sid] = {
        "input": input_pil.convert("RGB").resize((256, 256)),
        "output": translated,
        "direction": direction,
        "source": source,
        "subject": data.get("subject"),
        "slice_idx": data.get("slice_idx"),
    }
    while len(_pasd_xmod_sessions) > 32:
        _pasd_xmod_sessions.pop(next(iter(_pasd_xmod_sessions)))

    return jsonify({
        "session_id": sid,
        "direction": direction,
        "source_modality": src_modality,
        "target_modality": "TSE" if direction == "BTFE_to_TSE" else "BTFE",
        "input_url":  f"/api/pasd_xmod/image/{sid}/input",
        "output_url": f"/api/pasd_xmod/image/{sid}/output",
    })


@app.route("/api/pasd_xmod/image/<session_id>/<which>")
def api_pasd_xmod_image(session_id, which):
    """Stream the input or output image from a translation session."""
    s = _pasd_xmod_sessions.get(session_id)
    if not s or which not in ("input", "output"):
        return jsonify({"error": "not found"}), 404
    buf = io.BytesIO()
    s[which].save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg",
                     download_name=f"{which}_{session_id}.jpg")


@app.route("/api/pasd_gen/download_session/<session_id>")
def api_pasd_gen_download_session(session_id):
    """Bundle the entire session as a ZIP: synthetic images, masks, real MRI
    (if dataset source), metrics, and a README explaining the contents."""
    import zipfile, json as _json
    s = _pasd_gen_sessions.get(session_id)
    if not s:
        return jsonify({"error": "session not found"}), 404

    def _png_bytes(img):
        b = io.BytesIO(); img.save(b, format="PNG"); return b.getvalue()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        n = len(s["samples"])
        per_sample_masks = s.get("masks") is not None

        for i in range(n):
            sub = f"sample_{i:02d}"
            zf.writestr(f"{sub}/synthetic.png", _png_bytes(s["samples"][i]))
            mask = s["masks"][i] if per_sample_masks else s.get("shared_mask")
            if mask is not None:
                zf.writestr(f"{sub}/mask.png", _png_bytes(mask))
            if s.get("real_image") is not None:
                zf.writestr(f"{sub}/real_mri.png", _png_bytes(s["real_image"]))

        meta = {
            "session_id": session_id,
            "modality": s.get("modality"),
            "model": s.get("model"),
            "source": s.get("source"),
            "subject": s.get("subject"),
            "slice_idx": s.get("slice_idx"),
            "per_sample_masks": per_sample_masks,
            "num_samples": n,
            "metrics": s.get("metrics"),
        }
        zf.writestr("metrics.json", _json.dumps(meta, indent=2))

        readme = (
            "PASD MRI generation — exported session\n"
            "======================================\n"
            f"Session    : {session_id}\n"
            f"Model      : {s.get('model')}\n"
            f"Modality   : {s.get('modality')}\n"
            f"Source     : {s.get('source')}\n"
            f"Subject    : {s.get('subject') or '(uploaded mask)'}\n"
            f"Slice idx  : {s.get('slice_idx')}\n"
            f"# samples  : {n}\n"
            f"Per-sample masks (perturb mode): {per_sample_masks}\n\n"
            "Layout\n------\n"
            "Each sample_NN/ contains:\n"
            "  synthetic.png  — the model's output\n"
            "  mask.png       — the input mask (per-sample if perturb mode, else shared)\n"
            "  real_mri.png   — paired real MRI slice (present only if source=dataset)\n"
            "metrics.json     — per-sample mask Dice / PSNR / SSIM + session metadata\n\n"
            "SYNTHETIC IMAGES ARE FOR RESEARCH ONLY. NOT FOR CLINICAL USE.\n"
        )
        zf.writestr("README.txt", readme)

    buf.seek(0)
    return send_file(
        buf, mimetype="application/zip",
        download_name=f"pasd_gen_{s.get('model','x')}_{session_id}.zip",
        as_attachment=True,
    )


@app.route("/api/pasd_gen/sample_mask/<session_id>/<int:idx>")
def api_pasd_gen_sample_mask(session_id, idx):
    """Return the perturbed mask used for a given generated sample."""
    s = _pasd_gen_sessions.get(session_id)
    if not s or s.get("masks") is None or idx < 0 or idx >= len(s["masks"]):
        return jsonify({"error": "session/sample mask not found"}), 404
    m = s["masks"][idx]
    # Render mask as a colored RGBA preview for consistency with mask_preview
    arr = np.array(m.convert("L"))
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[arr > 0] = (255, 0, 128, 220)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/pasd_gen/mask_preview")
def api_pasd_gen_mask_preview():
    """Return a colored preview of a PASD GT mask (for the dataset picker UI)."""
    modality = request.args.get("modality", "BTFE").upper()
    subject = request.args.get("subject")
    slice_idx = int(request.args.get("slice_idx", 0))
    rows = [r for r in pasd_data_mod.build_index()
            if r["modality"] == modality and r["subject"] == subject
            and r["slice_idx"] == slice_idx]
    if not rows or not rows[0]["mask"]:
        return jsonify({"error": "mask not found"}), 404
    mask = Image.open(pasd_data_mod.ROOT / rows[0]["mask"]).convert("L")
    arr = np.array(mask)
    rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)
    rgba[arr > 0] = (255, 0, 128, 220)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ── Results tab: serve precomputed eval plot images ──

_APP_ROOT = Path(__file__).parent
_RESULTS_PLOTS = {
    "lowdata_sweep_both_modalities.png":
        _APP_ROOT / "pasd_models" / "tstr" / "lowdata_sweep_both_modalities.png",
    "lowdata_sweep_4way_BTFE.png":
        _APP_ROOT / "pasd_models" / "tstr" / "lowdata_sweep_4way_BTFE.png",
    "lowdata_sweep_BTFE.png":
        _APP_ROOT / "pasd_models" / "tstr" / "lowdata_sweep_BTFE.png",
    "cv_samples_ldm":      _APP_ROOT / "cv_eval" / "samples_ldm.png",
    "cv_samples_pix2pix":  _APP_ROOT / "cv_eval" / "samples_pix2pix.png",
    "cv_samples_cyclegan": _APP_ROOT / "cv_eval" / "samples_cyclegan.png",
}


@app.route("/api/results/plot/<name>")
def api_results_plot(name):
    """Serve a precomputed eval plot image referenced from the Results tab."""
    path = _RESULTS_PLOTS.get(name)
    if path is None or not path.exists():
        return jsonify({"error": f"plot '{name}' not available "
                        f"(missing or not yet generated)"}), 404
    return send_file(str(path), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
