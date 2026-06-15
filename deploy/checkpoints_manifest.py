"""Single source of truth: the production checkpoints SynthMed loads at runtime.

These are the ONLY checkpoints the running app needs (resolved from the path
constants in generation.py and from the TSTR segmentation viewer in app.py).
Everything else under *_models/ — per-step training snapshots, the multi-seed
TSTR sweep, alt-architecture runs, failed-SOTA experiments — is training-time
only and must NOT be shipped. The full tree is ~109 GB; this set is ~6 GB.

Both deploy/upload_checkpoints.py and deploy/download_checkpoints.py import this
list so the uploaded HF repo and the runtime download stay in lockstep.
"""
from pathlib import Path

# ── Generator / VAE checkpoints (paths relative to the app root) ──
GENERATOR_CKPTS = [
    # PASD — placental MRI. BTFE branch only (TSE LDM/VAE were not trained).
    "pasd_models/ldm_btfe/latest.pt",
    "pasd_models/vae_btfe/latest.pt",
    "pasd_models/cyclegan_btfe_tse/latest_G.pt",
    "pasd_models/tstr/real_BTFE/best.pt",
    "pasd_models/tstr/real_TSE/best.pt",
    # TopBrain — cerebrovascular CT/MRA.
    "cv_models/pix2pix_mr2ct/latest_G.pt",
    "cv_models/spade_mask2ct/latest_G.pt",
    "cv_models/ldm_mr2ct/latest.pt",
    "cv_models/ldm_combined/latest.pt",
    "cv_models/vae/latest.pt",
    "cv_models/cyclegan_mr_ct/best_G.pt",
    # Prostate158 — T2/ADC.
    "prostate_models/pix2pix_t2_adc/latest_G.pt",
    "prostate_models/vae/latest.pt",
    "prostate_models/ldm_t2_adc/latest.pt",
    "prostate_models/ldm_combined_t2_adc/latest.pt",
    "prostate_models/cyclegan_t2_adc/best_G.pt",
    "prostate_models/spade_mask_t2/best_G.pt",
]

# ── TSTR segmenters for the "Segmentation: Realism vs Relevance" viewer ──
# Only the non-seeded, default-architecture (UNet) best.pt per (dataset, mode).
TSTR_SEG_CONDITIONS = {
    "prostate": ["real", "synth_combined", "synth_cyclegan", "synth_ldm", "synth_pix2pix"],
    "topbrain": ["real", "synth_combined", "synth_cyclegan", "synth_ldm", "synth_pix2pix"],
    "pasd":     ["real", "synth_ldm", "synth_pix2pix", "synth_spade"],
}
TSTR_SEG_CKPTS = [
    f"tstr_synth_models/{ds}_{mode}/best.pt"
    for ds, modes in TSTR_SEG_CONDITIONS.items() for mode in modes
]

PRODUCTION_CKPTS = GENERATOR_CKPTS + TSTR_SEG_CKPTS


def existing(root: Path):
    """Subset of PRODUCTION_CKPTS that actually exists under `root`."""
    return [p for p in PRODUCTION_CKPTS if (Path(root) / p).is_file()]


def missing(root: Path):
    return [p for p in PRODUCTION_CKPTS if not (Path(root) / p).is_file()]
