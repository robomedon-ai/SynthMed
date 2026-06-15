"""Inference module for the PASD MRI synthesis Modules-tab UI.

Currently wraps the pix2pix baseline; the LDM will plug into the same
sample() API later.

Pattern follows hovernet/clam: a lazy module-level cache, one function
per supported model, returns PIL images ready for HTTP.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import pasd_pix2pix as p2p


# ── Paths & cache ──

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "pasd_models"
PIX2PIX_CKPTS = {
    "BTFE": MODELS_DIR / "pix2pix_btfe" / "latest_G.pt",
    "TSE":  MODELS_DIR / "pix2pix_tse"  / "latest_G.pt",
}
SPADE_CKPTS = {
    # v2 has L1=0 (matches Park 2019 recipe) + 200 epochs (vs v1's 80).
    # If you want to compare to v1, point this at spade_btfe/latest_G.pt.
    "BTFE": MODELS_DIR / "spade_btfe_v2" / "latest_G.pt",
    "TSE":  MODELS_DIR / "spade_tse_v2"  / "latest_G.pt",
}
LDM_CKPTS = {
    "BTFE": MODELS_DIR / "ldm_btfe" / "latest.pt",
    "TSE":  MODELS_DIR / "ldm_tse"  / "latest.pt",
}
# CycleGAN is bidirectional — one checkpoint contains both G_AB and G_BA.
CYCLEGAN_CKPT = MODELS_DIR / "cyclegan_btfe_tse" / "latest_G.pt"

# Cerebrovascular (TopBrain) generators live in their own dir.
CV_MODELS_DIR = ROOT / "cv_models"
CV_PIX2PIX_MR2CT_CKPT = CV_MODELS_DIR / "pix2pix_mr2ct" / "latest_G.pt"
CV_SPADE_MASK2CT_CKPT = CV_MODELS_DIR / "spade_mask2ct" / "latest_G.pt"
CV_LDM_MR2CT_CKPT     = CV_MODELS_DIR / "ldm_mr2ct"     / "latest.pt"
CV_LDM_COMBINED_CKPT  = CV_MODELS_DIR / "ldm_combined"  / "latest.pt"
CV_VAE_CKPT           = CV_MODELS_DIR / "vae"           / "latest.pt"
# CycleGAN is bidirectional — one checkpoint contains both G_AB (MR→CT)
# and G_BA (CT→MR). best_G.pt is the balance-guarded checkpoint; latest_G.pt
# is the final epoch (may have weaker D).
CV_CYCLEGAN_CKPT      = CV_MODELS_DIR / "cyclegan_mr_ct" / "best_G.pt"

# Prostate158 generators
PROSTATE_MODELS_DIR             = ROOT / "prostate_models"
PROSTATE_PIX2PIX_T2_ADC_CKPT    = PROSTATE_MODELS_DIR / "pix2pix_t2_adc" / "latest_G.pt"
PROSTATE_VAE_CKPT               = PROSTATE_MODELS_DIR / "vae"             / "latest.pt"
PROSTATE_LDM_T2_ADC_CKPT        = PROSTATE_MODELS_DIR / "ldm_t2_adc"      / "latest.pt"
PROSTATE_LDM_COMBINED_CKPT      = PROSTATE_MODELS_DIR / "ldm_combined_t2_adc" / "latest.pt"
# best_G = balance-guarded checkpoint (avoids late-stage D-collapse trap)
PROSTATE_CYCLEGAN_CKPT          = PROSTATE_MODELS_DIR / "cyclegan_t2_adc" / "best_G.pt"
# best_G = pre-collapse ep70 with EMA applied (D collapsed in late epochs)
PROSTATE_SPADE_CKPT             = PROSTATE_MODELS_DIR / "spade_mask_t2"   / "best_G.pt"
VAE_CKPTS = {
    "BTFE": MODELS_DIR / "vae_btfe" / "latest.pt",
    "TSE":  MODELS_DIR / "vae_tse"  / "latest.pt",
}

# Real-only TSTR segmenter used as a reference for mask-faithfulness scoring.
SEGMENTER_CKPTS = {
    "BTFE": MODELS_DIR / "tstr" / "real_BTFE" / "best.pt",
    "TSE":  MODELS_DIR / "tstr" / "real_TSE"  / "best.pt",
}

# {(modality, model_kind): (object, ckpt_mtime)} — mtime triggers reload
# For "pix2pix" the object is an nn.Module; for "ldm" it's (vae, unet).
_model_cache: dict[tuple[str, str], tuple[object, float]] = {}


def _ckpt_info(path: Path) -> dict:
    return {
        "available": path.exists(),
        "path": str(path.relative_to(ROOT)) if path.exists() else None,
        "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else None,
    }


def available_checkpoints() -> dict[str, dict]:
    """Report which modality/model combos have trained weights on disk."""
    out = {}
    for mod in PIX2PIX_CKPTS:
        out[mod] = {
            "pix2pix": _ckpt_info(PIX2PIX_CKPTS[mod]),
            "spade":   _ckpt_info(SPADE_CKPTS[mod]),
            "ldm":     _ckpt_info(LDM_CKPTS[mod]),
        }
    out["cyclegan"] = _ckpt_info(CYCLEGAN_CKPT)
    out["cv"] = {
        "pix2pix_mr2ct":   _ckpt_info(CV_PIX2PIX_MR2CT_CKPT),
        "spade_mask2ct":   _ckpt_info(CV_SPADE_MASK2CT_CKPT),
        "ldm_mr2ct":       _ckpt_info(CV_LDM_MR2CT_CKPT),
        "ldm_combined":    _ckpt_info(CV_LDM_COMBINED_CKPT),
        "cyclegan_mr_ct":  _ckpt_info(CV_CYCLEGAN_CKPT),
    }
    out["prostate"] = {
        "pix2pix_t2_adc":  _ckpt_info(PROSTATE_PIX2PIX_T2_ADC_CKPT),
        "ldm_t2_adc":      _ckpt_info(PROSTATE_LDM_T2_ADC_CKPT),
        "cyclegan_t2_adc": _ckpt_info(PROSTATE_CYCLEGAN_CKPT),
        "spade_mask_t2":   _ckpt_info(PROSTATE_SPADE_CKPT),
    }
    return out


# ── Cerebrovascular (TopBrain) inference ──

def _get_cv_pix2pix_mr2ct(device: str = "cuda"):
    """Lazy-load the MRA→CTA pix2pix generator."""
    key = ("cv", "pix2pix_mr2ct")
    ck = CV_PIX2PIX_MR2CT_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No CV pix2pix checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]
    import pasd_pix2pix as p2p
    G = p2p.UnetGenerator(in_ch=1, out_ch=1, ngf=64)
    sd = torch.load(str(ck), map_location="cpu", weights_only=True)
    G.load_state_dict(sd["G"] if "G" in sd else sd)
    G.eval()
    for p in G.parameters():
        p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G = G.to(device)
    _model_cache[key] = (G, mtime)
    return G


def _get_cv_spade_mask2ct(device: str = "cuda"):
    """Lazy-load the multi-class SPADE (vessel-mask → CTA) generator.

    Reads EMA weights from the latest checkpoint when present."""
    key = ("cv", "spade_mask2ct")
    ck = CV_SPADE_MASK2CT_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No CV SPADE checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import cv_spade as sp
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    args = sd.get("args", {})
    ngf = int(args.get("ngf", 32))
    G = sp.CvSPADEGenerator(mask_nc=sp.CV_LABEL_NC, image_ch=1,
                            ngf=ngf, z_dim=256)
    G.load_state_dict(sd["G"])
    G.eval()
    for p in G.parameters():
        p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G = G.to(device)
    _model_cache[key] = (G, mtime)
    return G


@torch.no_grad()
def cv_sample_mask2ct(mask_long: torch.Tensor,
                      seed: Optional[int] = None,
                      device: str = "cuda",
                      watermark: bool = True) -> Image.Image:
    """Run SPADE on a (H, W) integer mask. Returns a PIL grayscale CTA image."""
    import cv_spade as sp
    G = _get_cv_spade_mask2ct(device=device)
    if mask_long.ndim == 2:
        mask_long = mask_long.unsqueeze(0)            # (1, H, W)
    mask_oh = sp.one_hot_mask(mask_long)              # (1, 41, H, W)
    if device.startswith("cuda") and torch.cuda.is_available():
        mask_oh = mask_oh.to(device)
    if seed is not None:
        torch.manual_seed(int(seed))
    z = torch.randn(mask_oh.size(0), G.z_dim, device=mask_oh.device,
                    dtype=mask_oh.dtype)
    fake = G(mask_oh, z=z)
    arr = fake.clamp(-1, 1).add(1).div(2).mul(255).round().byte().squeeze().cpu().numpy()
    pil = Image.fromarray(arr, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"),
                         text="SYNTHETIC CTA (from mask) · research only")
    return pil


def _get_cv_cyclegan(device: str = "cuda"):
    """Lazy-load (G_AB, G_BA) for CV CycleGAN. A=MR, B=CT — same convention
    as train_cv_cyclegan.py. Returns the balance-guarded best_G.pt by default.
    """
    key = ("cv", "cyclegan")
    ck = CV_CYCLEGAN_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No CV CycleGAN checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import pasd_cyclegan as cg
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    ngf = int(sd.get("args", {}).get("ngf", 64))
    G_AB = cg.ResnetGenerator(1, 1, ngf=ngf, n_blocks=9)   # MR → CT
    G_BA = cg.ResnetGenerator(1, 1, ngf=ngf, n_blocks=9)   # CT → MR
    G_AB.load_state_dict(sd["G_AB"]); G_BA.load_state_dict(sd["G_BA"])
    for g in (G_AB, G_BA):
        g.eval()
        for p in g.parameters():
            p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G_AB = G_AB.to(device); G_BA = G_BA.to(device)
    _model_cache[key] = ((G_AB, G_BA), mtime)
    return G_AB, G_BA


@torch.no_grad()
def cv_translate_cyclegan(image: Image.Image, direction: str = "MR_to_CT",
                          device: str = "cuda",
                          watermark: bool = True) -> Image.Image:
    """Cross-modality CycleGAN translation.

    direction ∈ {"MR_to_CT", "CT_to_MR"} — uses G_AB for the former, G_BA for the latter.
    """
    G_AB, G_BA = _get_cv_cyclegan(device=device)
    if direction == "MR_to_CT":
        G = G_AB; tag = "CT (CycleGAN from MR)"
    elif direction == "CT_to_MR":
        G = G_BA; tag = "MR (CycleGAN from CT)"
    else:
        raise ValueError("direction must be MR_to_CT or CT_to_MR")

    img = image.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1   # (1,1,256,256) in [-1,1]
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)
    out_t = G(t)
    out = out_t.clamp(-1, 1).add(1).div(2).mul(255).round().byte().squeeze().cpu().numpy()
    pil = Image.fromarray(out, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"),
                         text=f"SYNTHETIC {tag} · research only")
    return pil


def _get_cv_ldm(device: str = "cuda"):
    """Lazy-load (VAE, UNet) pair for CV LDM inference.

    Uses the CV-fine-tuned decoder if present, else pretrained SD-VAE.
    EMA weights applied for inference quality.
    """
    key = ("cv", "ldm")
    ck = CV_LDM_MR2CT_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No CV LDM checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import cv_vae as vae_mod
    import cv_ldm as ldm

    vae = vae_mod.load_vae(freeze_encoder=True)
    if CV_VAE_CKPT.exists():
        sd = torch.load(str(CV_VAE_CKPT), map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    unet = ldm.build_unet()
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["unet"])
    if "ema" in sd:
        ema = ldm.EMAModel(unet)
        ema.load_state_dict(sd["ema"], device="cpu")
        ema.apply_to(unet)
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False

    if device.startswith("cuda") and torch.cuda.is_available():
        vae = vae.to(device); unet = unet.to(device)

    _model_cache[key] = ((vae, unet), mtime)
    return (vae, unet)


def _get_cv_ldm_combined(device: str = "cuda"):
    """Lazy-load (VAE, UNet) pair for the COMBINED LDM (MR + mask → CT)."""
    key = ("cv", "ldm_combined")
    ck = CV_LDM_COMBINED_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No combined CV LDM checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import cv_vae as vae_mod
    import cv_ldm_combined as ldm_c

    vae = vae_mod.load_vae(freeze_encoder=True)
    if CV_VAE_CKPT.exists():
        sd = torch.load(str(CV_VAE_CKPT), map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    unet = ldm_c.build_unet()
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["unet"])
    if "ema" in sd:
        ema = ldm_c.EMAModel(unet)
        ema.load_state_dict(sd["ema"], device="cpu")
        ema.apply_to(unet)
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False

    if device.startswith("cuda") and torch.cuda.is_available():
        vae = vae.to(device); unet = unet.to(device)

    _model_cache[key] = ((vae, unet), mtime)
    return (vae, unet)


@torch.no_grad()
def cv_translate_mr_mask_to_ct_ldm(mr_pil: Image.Image,
                                    mask_long: torch.Tensor,
                                    num_inference_steps: int = 25,
                                    guidance_scale: float = 1.5,
                                    seed: Optional[int] = None,
                                    device: str = "cuda",
                                    watermark: bool = True) -> Image.Image:
    """Combined-LDM inference: MR slice + CT vessel mask → synthetic CT slice.

    mr_pil: grayscale PIL of the MR slice.
    mask_long: torch tensor (H, W) long, values 0..40, the CT vessel mask.
    """
    import cv_ldm_combined as ldm_c
    vae, unet = _get_cv_ldm_combined(device=device)

    # MR → (1, 3, 256, 256) in [-1, 1]
    img = mr_pil.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1
    t = t.repeat(1, 3, 1, 1)
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)

    # Mask → (1, H, W) long. Caller may pass any size; ldm.sample handles resize.
    if mask_long.ndim == 2:
        mask_long = mask_long.unsqueeze(0)
    if device.startswith("cuda") and torch.cuda.is_available():
        mask_long = mask_long.to(device)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    image = ldm_c.sample(unet, vae, t, mask_long,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         generator=gen, device=device)
    arr = image[0].clamp(-1, 1).add(1).div(2).mul(255).round().byte().cpu().numpy()
    gray = arr.mean(axis=0).astype(np.uint8)
    pil = Image.fromarray(gray, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"),
                         text="SYNTHETIC CTA (LDM, MR+mask) · research only")
    return pil


@torch.no_grad()
def cv_translate_mr2ct_ldm(mr_pil: Image.Image,
                            num_samples: int = 1,
                            num_inference_steps: int = 25,
                            guidance_scale: float = 1.5,
                            seed: Optional[int] = None,
                            device: str = "cuda",
                            watermark: bool = True) -> list[Image.Image]:
    """MRA slice → list of stochastic synthetic CTA slices via LDM."""
    import cv_ldm as ldm
    vae, unet = _get_cv_ldm(device=device)

    # Preprocess MR to (1, 3, 256, 256) in [-1, 1]
    img = mr_pil.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1   # (1, 1, 256, 256)
    t = t.repeat(1, 3, 1, 1)                                      # (1, 3, 256, 256)
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)
    # Tile across num_samples (each replica gets a different random init latent)
    t = t.repeat(num_samples, 1, 1, 1)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    images = ldm.sample(unet, vae, t,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=gen, device=device)
    # Convert each (3, H, W) → grayscale PIL (average the 3 channels)
    outs = []
    for i in range(num_samples):
        arr = images[i].clamp(-1, 1).add(1).div(2).mul(255).round().byte().cpu().numpy()
        gray = arr.mean(axis=0).astype(np.uint8)
        pil = Image.fromarray(gray, "L")
        if watermark:
            pil = _watermark(pil.convert("RGB"),
                             text="SYNTHETIC CTA (LDM, from MRA) · research only")
        outs.append(pil)
    return outs


@torch.no_grad()
def cv_translate_mr2ct(mr_pil: Image.Image,
                       device: str = "cuda",
                       watermark: bool = True) -> Image.Image:
    """MRA slice (grayscale PIL) → synthetic CTA slice (grayscale PIL, [0,255])."""
    G = _get_cv_pix2pix_mr2ct(device=device)
    img = mr_pil.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1  # (1,1,256,256) in [-1,1]
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)
    fake = G(t)  # (1, 1, 256, 256) in [-1, 1]
    out = fake.clamp(-1, 1).add(1).div(2).mul(255).round().byte().squeeze().cpu().numpy()
    pil = Image.fromarray(out, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"),
                         text="SYNTHETIC CTA (from MRA) · research only")
    return pil


# ── Prostate158 inference ────────────────────────────────────────────────

def _get_prostate_pix2pix_t2_adc(device: str = "cuda"):
    """Lazy-load the T2 → ADC pix2pix generator (prostate158)."""
    key = ("prostate", "pix2pix_t2_adc")
    ck = PROSTATE_PIX2PIX_T2_ADC_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No prostate pix2pix checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]
    import pasd_pix2pix as p2p
    G = p2p.UnetGenerator(in_ch=1, out_ch=1, ngf=64)
    sd = torch.load(str(ck), map_location="cpu", weights_only=True)
    G.load_state_dict(sd["G"] if "G" in sd else sd)
    G.eval()
    for p in G.parameters():
        p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G = G.to(device)
    _model_cache[key] = (G, mtime)
    return G


@torch.no_grad()
def prostate_translate_t2_to_adc(t2_pil: Image.Image,
                                  device: str = "cuda",
                                  watermark: bool = True) -> Image.Image:
    """T2 slice (grayscale PIL) → synthetic ADC slice (grayscale PIL, [0,255])."""
    G = _get_prostate_pix2pix_t2_adc(device=device)
    img = t2_pil.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)
    fake = G(t)
    out = fake.clamp(-1, 1).add(1).div(2).mul(255).round().byte().squeeze().cpu().numpy()
    pil = Image.fromarray(out, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"),
                         text="SYNTHETIC ADC (from T2) · research only")
    return pil


def _get_prostate_ldm(device: str = "cuda"):
    """Lazy-load (VAE, UNet) pair for the prostate158 LDM (T2 → ADC)."""
    key = ("prostate", "ldm_t2_adc")
    ck = PROSTATE_LDM_T2_ADC_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No prostate LDM checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import prostate_vae as vae_mod
    import prostate_ldm as ldm

    vae = vae_mod.load_vae(freeze_encoder=True)
    if PROSTATE_VAE_CKPT.exists():
        sd = torch.load(str(PROSTATE_VAE_CKPT), map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    unet = ldm.build_unet()
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["unet"])
    if "ema" in sd:
        ema = ldm.EMAModel(unet)
        ema.load_state_dict(sd["ema"], device="cpu")
        ema.apply_to(unet)
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False

    if device.startswith("cuda") and torch.cuda.is_available():
        vae = vae.to(device); unet = unet.to(device)

    _model_cache[key] = ((vae, unet), mtime)
    return (vae, unet)


def _get_prostate_spade(device: str = "cuda"):
    """Lazy-load the prostate SPADE (anatomy_mask → T2) generator."""
    key = ("prostate", "spade_mask_t2")
    ck = PROSTATE_SPADE_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No prostate SPADE checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]
    import prostate_spade as sp
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    args = sd.get("args", {})
    ngf = int(args.get("ngf", 32))
    G = sp.ProstateSPADEGenerator(mask_nc=sp.PROSTATE_LABEL_NC, image_ch=1,
                                   ngf=ngf, z_dim=256)
    G.load_state_dict(sd["G"])
    G.eval()
    for p in G.parameters(): p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G = G.to(device)
    _model_cache[key] = (G, mtime)
    return G


@torch.no_grad()
def prostate_sample_mask_to_t2(mask_long: torch.Tensor,
                                seed: Optional[int] = None,
                                device: str = "cuda",
                                watermark: bool = True) -> Image.Image:
    """SPADE on a (H, W) 3-class anatomy mask → grayscale T2 PIL."""
    import prostate_spade as sp
    G = _get_prostate_spade(device=device)
    if mask_long.ndim == 2:
        mask_long = mask_long.unsqueeze(0)
    mask_oh = sp.one_hot_mask(mask_long)
    if device.startswith("cuda") and torch.cuda.is_available():
        mask_oh = mask_oh.to(device)
    if seed is not None:
        torch.manual_seed(int(seed))
    z = torch.randn(mask_oh.size(0), G.z_dim, device=mask_oh.device,
                    dtype=mask_oh.dtype)
    fake = G(mask_oh, z=z)
    arr = fake.clamp(-1, 1).add(1).div(2).mul(255).round().byte().squeeze().cpu().numpy()
    pil = Image.fromarray(arr, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"),
                         text="SYNTHETIC T2 (from anatomy mask) · research only")
    return pil


def _get_prostate_cyclegan(device: str = "cuda"):
    """Lazy-load (G_AB, G_BA) for prostate CycleGAN. A=T2, B=ADC."""
    key = ("prostate", "cyclegan")
    ck = PROSTATE_CYCLEGAN_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No prostate CycleGAN checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]
    import pasd_cyclegan as cg
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    ngf = int(sd.get("args", {}).get("ngf", 64))
    G_AB = cg.ResnetGenerator(1, 1, ngf=ngf, n_blocks=9)  # T2 → ADC
    G_BA = cg.ResnetGenerator(1, 1, ngf=ngf, n_blocks=9)  # ADC → T2
    G_AB.load_state_dict(sd["G_AB"]); G_BA.load_state_dict(sd["G_BA"])
    for g in (G_AB, G_BA):
        g.eval()
        for p in g.parameters(): p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G_AB = G_AB.to(device); G_BA = G_BA.to(device)
    _model_cache[key] = ((G_AB, G_BA), mtime)
    return G_AB, G_BA


@torch.no_grad()
def prostate_translate_cyclegan(image: Image.Image, direction: str = "T2_to_ADC",
                                 device: str = "cuda",
                                 watermark: bool = True) -> Image.Image:
    """CycleGAN T2 ↔ ADC translation.

    direction ∈ {"T2_to_ADC", "ADC_to_T2"}.
    """
    G_AB, G_BA = _get_prostate_cyclegan(device=device)
    if direction == "T2_to_ADC":
        G = G_AB; tag = "ADC (CycleGAN from T2)"
    elif direction == "ADC_to_T2":
        G = G_BA; tag = "T2 (CycleGAN from ADC)"
    else:
        raise ValueError("direction must be T2_to_ADC or ADC_to_T2")
    img = image.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)
    out_t = G(t)
    out = out_t.clamp(-1, 1).add(1).div(2).mul(255).round().byte().squeeze().cpu().numpy()
    pil = Image.fromarray(out, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"), text=f"SYNTHETIC {tag} · research only")
    return pil


@torch.no_grad()
def prostate_translate_t2_to_adc_ldm(t2_pil: Image.Image,
                                      num_inference_steps: int = 25,
                                      guidance_scale: float = 1.0,
                                      seed: Optional[int] = None,
                                      n_ensemble: int = 1,
                                      device: str = "cuda",
                                      watermark: bool = True) -> Image.Image:
    """T2 slice → synthetic ADC via latent diffusion.

    n_ensemble > 1 samples that many times (different seeds) and averages —
    reduces stochastic variance, lifting SSIM/PSNR at some cost to FID/sharpness
    (perception-distortion trade-off; verified on the prostate test set).
    """
    import prostate_ldm as ldm
    vae, unet = _get_prostate_ldm(device=device)

    img = t2_pil.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1
    t = t.repeat(1, 3, 1, 1)
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)

    n_ensemble = max(1, int(n_ensemble))
    acc = None
    for k in range(n_ensemble):
        gen = None
        if seed is not None:
            gen = torch.Generator(device=device).manual_seed(int(seed) + k)
        elif n_ensemble > 1:
            gen = torch.Generator(device=device).manual_seed(k)
        image = ldm.sample(unet, vae, t,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=guidance_scale,
                           generator=gen, device=device)
        gray = image[0].clamp(-1, 1).add(1).div(2).mean(0)  # [0,1]
        acc = gray if acc is None else acc + gray
    gray = (acc / n_ensemble).mul(255).round().byte().cpu().numpy()
    pil = Image.fromarray(gray, "L")
    if watermark:
        tag = (f"SYNTHETIC ADC (LDM×{n_ensemble}, from T2)" if n_ensemble > 1
               else "SYNTHETIC ADC (LDM, from T2)")
        pil = _watermark(pil.convert("RGB"), text=f"{tag} · research only")
    return pil


def _get_prostate_ldm_combined(device: str = "cuda"):
    """Lazy-load (VAE, UNet) for the combined prostate LDM (T2 + mask → ADC)."""
    key = ("prostate", "ldm_combined")
    ck = PROSTATE_LDM_COMBINED_CKPT
    if not ck.exists():
        raise FileNotFoundError(f"No combined prostate LDM checkpoint at {ck}")
    mtime = ck.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import prostate_vae as vae_mod
    import prostate_ldm_combined as ldm_c

    vae = vae_mod.load_vae(freeze_encoder=True)
    if PROSTATE_VAE_CKPT.exists():
        sd = torch.load(str(PROSTATE_VAE_CKPT), map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    unet = ldm_c.build_unet()
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["unet"])
    if "ema" in sd:
        ema = ldm_c.EMAModel(unet)
        ema.load_state_dict(sd["ema"], device="cpu")
        ema.apply_to(unet)
    unet.eval()
    for p in unet.parameters(): p.requires_grad = False

    if device.startswith("cuda") and torch.cuda.is_available():
        vae = vae.to(device); unet = unet.to(device)

    _model_cache[key] = ((vae, unet), mtime)
    return (vae, unet)


@torch.no_grad()
def prostate_translate_t2_mask_to_adc_ldm(t2_pil: Image.Image,
                                           mask_long: torch.Tensor,
                                           num_inference_steps: int = 25,
                                           guidance_scale: float = 1.0,
                                           seed: Optional[int] = None,
                                           device: str = "cuda",
                                           watermark: bool = True) -> Image.Image:
    """Combined-LDM inference: T2 slice + anatomy mask → synthetic ADC slice."""
    import prostate_ldm_combined as ldm_c
    vae, unet = _get_prostate_ldm_combined(device=device)

    img = t2_pil.convert("L").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0) * 2 - 1
    t = t.repeat(1, 3, 1, 1)
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)

    if mask_long.ndim == 2:
        mask_long = mask_long.unsqueeze(0)
    if device.startswith("cuda") and torch.cuda.is_available():
        mask_long = mask_long.to(device)

    g = None
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(int(seed))

    image = ldm_c.sample(unet, vae, t, mask_long,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         generator=g, device=device)
    arr = image[0].clamp(-1, 1).add(1).div(2).mul(255).round().byte().cpu().numpy()
    gray = arr.mean(axis=0).astype(np.uint8)
    pil = Image.fromarray(gray, "L")
    if watermark:
        pil = _watermark(pil.convert("RGB"),
                         text="SYNTHETIC ADC (LDM, T2+mask) · research only")
    return pil


def _get_pix2pix(modality: str, device: str = "cuda") -> torch.nn.Module:
    """Lazy-load (and auto-reload on checkpoint update) the pix2pix generator."""
    key = (modality, "pix2pix")
    ckpt_path = PIX2PIX_CKPTS[modality]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    mtime = ckpt_path.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    G = p2p.UnetGenerator(in_ch=1, out_ch=3, ngf=64)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    G.load_state_dict(state["G"] if "G" in state else state)
    G.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        G = G.to(device)
    _model_cache[key] = (G, mtime)
    return G


def _get_spade(modality: str, device: str = "cuda") -> torch.nn.Module:
    """Lazy-load the SPADE generator. Uses the EMA-applied latest_G.pt."""
    key = (modality, "spade")
    ckpt_path = SPADE_CKPTS[modality]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No SPADE checkpoint at {ckpt_path}")
    mtime = ckpt_path.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import pasd_spade as sp
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt_args = sd.get("args", {})
    ngf = int(ckpt_args.get("ngf", 32))
    G = sp.SPADEGenerator(mask_ch=1, image_ch=3, ngf=ngf, z_dim=256)
    G.load_state_dict(sd["G"])
    G.eval()
    for p in G.parameters():
        p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G = G.to(device)
    _model_cache[key] = (G, mtime)
    return G


def _get_ldm(modality: str, device: str = "cuda"):
    """Lazy-load (VAE, UNet) pair for LDM inference. EMA weights applied."""
    key = (modality, "ldm")
    ldm_ckpt = LDM_CKPTS[modality]
    if not ldm_ckpt.exists():
        raise FileNotFoundError(f"No LDM checkpoint at {ldm_ckpt}")
    mtime = ldm_ckpt.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import pasd_vae as vae_mod
    import pasd_ldm as ldm

    # Load VAE — pretrained encoder + (optionally) our fine-tuned decoder.
    vae = vae_mod.load_vae(freeze_encoder=True)
    vae_ckpt = VAE_CKPTS[modality]
    if vae_ckpt.exists():
        sd = torch.load(str(vae_ckpt), map_location="cpu", weights_only=False)
        vae.decoder.load_state_dict(sd["decoder"])
        vae.post_quant_conv.load_state_dict(sd["post_quant_conv"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Load UNet; use EMA shadow for inference if present.
    unet = ldm.build_unet()
    sd = torch.load(str(ldm_ckpt), map_location="cpu", weights_only=False)
    unet.load_state_dict(sd["unet"])
    if "ema" in sd:
        ema = ldm.EMAModel(unet)
        ema.load_state_dict(sd["ema"], device="cpu")
        ema.apply_to(unet)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    if device.startswith("cuda") and torch.cuda.is_available():
        vae = vae.to(device)
        unet = unet.to(device)

    _model_cache[key] = ((vae, unet), mtime)
    return (vae, unet)


def _get_cyclegan(device: str = "cuda"):
    """Lazy-load the bidirectional CycleGAN. Returns (G_AB, G_BA)
    where A=BTFE, B=TSE. Cache is invalidated when the checkpoint changes."""
    key = ("cyclegan", "both")
    if not CYCLEGAN_CKPT.exists():
        raise FileNotFoundError(f"No CycleGAN checkpoint at {CYCLEGAN_CKPT}")
    mtime = CYCLEGAN_CKPT.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import pasd_cyclegan as cg
    sd = torch.load(str(CYCLEGAN_CKPT), map_location="cpu", weights_only=False)
    ngf = int(sd.get("args", {}).get("ngf", 64))
    G_AB = cg.ResnetGenerator(3, 3, ngf=ngf, n_blocks=9)
    G_BA = cg.ResnetGenerator(3, 3, ngf=ngf, n_blocks=9)
    G_AB.load_state_dict(sd["G_AB"]); G_BA.load_state_dict(sd["G_BA"])
    for g in (G_AB, G_BA):
        g.eval()
        for p in g.parameters():
            p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        G_AB = G_AB.to(device); G_BA = G_BA.to(device)
    _model_cache[key] = ((G_AB, G_BA), mtime)
    return G_AB, G_BA


@torch.no_grad()
def translate(image: Image.Image, direction: str = "BTFE_to_TSE",
              device: str = "cuda", watermark: bool = True) -> Image.Image:
    """Cross-modality translation via CycleGAN.

    direction = "BTFE_to_TSE" runs G_AB; "TSE_to_BTFE" runs G_BA.
    Input PIL image (any size, any mode) is resized to 256x256 and
    grayscale-replicated if single-channel.
    """
    G_AB, G_BA = _get_cyclegan(device=device)
    if direction == "BTFE_to_TSE":
        G = G_AB
    elif direction == "TSE_to_BTFE":
        G = G_BA
    else:
        raise ValueError(f"direction must be BTFE_to_TSE or TSE_to_BTFE")

    img = image.convert("RGB").resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)
    out_t = G(t)
    pil = _tensor_to_pil(out_t[0])
    if watermark:
        pil = _watermark(pil, text="SYNTHETIC (cross-modality) · research only")
    return pil


def _get_segmenter(modality: str, device: str = "cuda"):
    """Lazy-load the reference TSTR segmenter for mask-faithfulness scoring.

    Returns None if no checkpoint exists (silently disables the metric).
    """
    key = (modality, "segmenter")
    ckpt = SEGMENTER_CKPTS.get(modality)
    if ckpt is None or not ckpt.exists():
        return None
    mtime = ckpt.stat().st_mtime
    if key in _model_cache and _model_cache[key][1] == mtime:
        return _model_cache[key][0]

    import pasd_seg as seg
    model = seg.SmallUNet(in_ch=3, out_ch=1)
    sd = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device)
    _model_cache[key] = (model, mtime)
    return model


@torch.no_grad()
def compute_sample_metrics(synth_pil: Image.Image,
                           mask_pil: Image.Image,
                           real_pil: Optional[Image.Image] = None,
                           modality: str = "BTFE",
                           device: str = "cuda") -> dict:
    """Per-sample quality metrics for the UI.

    Always computes (if a reference segmenter exists):
        mask_dice : Dice(segmenter(synthetic), input_mask) — mask faithfulness
    When real_pil is provided (dataset mask source):
        psnr      : pixel PSNR vs paired real (dB)
        ssim      : structural similarity vs paired real
    """
    out: dict = {}

    # Drop the bottom-bar watermark before measurement (matches eval_pasd).
    def _strip(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.crop((0, 0, w, h - 22))

    s_pil = _strip(synth_pil).convert("RGB").resize((256, 256))
    s_arr = np.asarray(s_pil, dtype=np.float32) / 255.0  # (H, W, 3) in [0,1]

    # ── Mask faithfulness via the trained segmenter ──
    segmenter = _get_segmenter(modality, device=device)
    if segmenter is not None:
        # Convert to label-array directly — handles both palette PNGs (values
        # 0/1) and grayscale PNGs (0/255) without losing pixels.
        raw = np.asarray(mask_pil)
        if raw.ndim == 3:
            raw = raw[..., 0]
        m_arr = (raw > 0).astype(np.uint8) * 255
        m_pil = Image.fromarray(m_arr, "L").resize((256, 256), Image.NEAREST)
        m_arr = (np.asarray(m_pil) > 127).astype(np.float32)  # (H, W) in {0,1}

        s_t = torch.from_numpy(s_arr).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        m_t = torch.from_numpy(m_arr).unsqueeze(0).unsqueeze(0)
        if device.startswith("cuda") and torch.cuda.is_available():
            s_t = s_t.to(device); m_t = m_t.to(device)

        logits = segmenter(s_t)
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter = (pred * m_t).sum()
        union = pred.sum() + m_t.sum()
        out["mask_dice"] = float(((2 * inter + 1e-6) / (union + 1e-6)).cpu())

    # ── Paired metrics vs real (dataset mode only) ──
    if real_pil is not None:
        r_pil = real_pil.convert("RGB").resize((256, 256))
        r_arr = np.asarray(r_pil, dtype=np.float32) / 255.0
        mse = float(np.mean((s_arr - r_arr) ** 2))
        out["psnr"] = float(10 * np.log10(1.0 / max(mse, 1e-10)))
        try:
            from skimage.metrics import structural_similarity
            out["ssim"] = float(structural_similarity(
                s_arr, r_arr, channel_axis=2, data_range=1.0))
        except Exception:
            pass  # ssim is optional; skip if skimage unavailable

    return out


def clear_cache():
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Mask perturbation (for generating novel mask+image pairs) ──

def perturb_mask(mask: Image.Image, seed: int,
                 max_rotation_deg: float = 15.0,
                 max_translate_frac: float = 0.08,
                 scale_range: tuple[float, float] = (0.85, 1.15),
                 p_flip: float = 0.5,
                 size: int = 256) -> Image.Image:
    """Apply a deterministic-by-seed affine perturbation to a binary mask.

    Used to generate novel (mask, image) pairs: each sample seed yields a
    distinct mask, which the generator then maps to a distinct synthetic
    image. The output is binary (0/255) PIL 'L' mode at `size x size`.
    """
    rng = np.random.default_rng(seed)
    if mask.mode != "L":
        mask = mask.convert("L")
    mask = mask.resize((size, size), Image.NEAREST)

    angle  = float(rng.uniform(-max_rotation_deg, max_rotation_deg))
    tx     = float(rng.uniform(-max_translate_frac, max_translate_frac) * size)
    ty     = float(rng.uniform(-max_translate_frac, max_translate_frac) * size)
    scale  = float(rng.uniform(*scale_range))
    do_flip = rng.random() < p_flip

    # torchvision functional handles affine + nearest interp cleanly
    from torchvision.transforms import functional as TF
    if do_flip:
        mask = TF.hflip(mask)
    mask = TF.affine(mask, angle=angle, translate=(tx, ty),
                     scale=scale, shear=0.0,
                     interpolation=TF.InterpolationMode.NEAREST, fill=0)
    # Re-binarize after interpolation
    arr = (np.array(mask) > 0).astype(np.uint8) * 255
    return Image.fromarray(arr, "L")


# ── Mask preprocessing ──

def _mask_to_tensor(mask: Image.Image, size: int = 256,
                    device: str = "cuda") -> torch.Tensor:
    """PIL mask (any size/mode) -> (1, 1, size, size) float in [-1, 1]."""
    if mask.mode != "L":
        mask = mask.convert("L")
    mask = mask.resize((size, size), Image.NEAREST)
    arr = (np.array(mask) > 0).astype(np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = t * 2.0 - 1.0
    if device.startswith("cuda") and torch.cuda.is_available():
        t = t.to(device)
    return t


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) tensor in [-1,1] -> PIL RGB."""
    arr = t.clamp(-1, 1).add(1).div(2).mul(255).round().byte()
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")


# ── Watermark (matches the plan: synthetic images are NEVER unmarked) ──

def _watermark(img: Image.Image, text: str = "SYNTHETIC · research only"
               ) -> Image.Image:
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=12)
    except Exception:
        font = ImageFont.load_default()
    pad = 4
    w, h = img.size
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([(0, h - th - 2 * pad), (tw + 2 * pad, h)],
                   fill=(0, 0, 0))
    draw.text((pad, h - th - pad), text, fill=(255, 80, 128), font=font)
    return img


# ── Public sample API ──

@torch.no_grad()
def sample(mask: Image.Image, modality: str = "BTFE",
           model: str = "pix2pix", num_samples: int = 1,
           seed: Optional[int] = None,
           device: str = "cuda",
           watermark: bool = True,
           # LDM-only kwargs (ignored for pix2pix):
           # Defaults from sweep_ldm_eval: g=1.5 wins on every metric.
           num_inference_steps: int = 25,
           guidance_scale: float = 1.5) -> list[Image.Image]:
    """Sample synthetic MRI slices conditioned on a single mask.

    pix2pix is deterministic given a mask, so num_samples > 1 yields
    identical outputs unless the mask is perturbed (see sample_paired).
    The LDM is stochastic — each sample uses a different random noise
    vector and produces a distinct image.
    """
    if model == "pix2pix":
        G = _get_pix2pix(modality, device=device)
        if seed is not None:
            torch.manual_seed(seed)
        mask_t = _mask_to_tensor(mask, size=256, device=device)
        outputs = []
        for _ in range(num_samples):
            fake = G(mask_t)
            img = _tensor_to_pil(fake[0])
            if watermark:
                img = _watermark(img)
            outputs.append(img)
        return outputs

    if model == "ldm":
        return _sample_ldm(mask, modality, num_samples, seed,
                           num_inference_steps, guidance_scale,
                           device, watermark)

    if model == "spade":
        return _sample_spade(mask, modality, num_samples, seed,
                             device, watermark)

    raise NotImplementedError(f"model={model} not wired yet")


@torch.no_grad()
def _sample_spade(mask: Image.Image, modality: str, num_samples: int,
                  seed: Optional[int], device: str,
                  watermark: bool) -> list[Image.Image]:
    G = _get_spade(modality, device=device)
    mask_t = _mask_to_tensor(mask, size=256, device=device)
    mask_t = mask_t.expand(num_samples, -1, -1, -1).contiguous()
    if seed is not None:
        torch.manual_seed(int(seed))
    # SPADE is stochastic: each sample uses a fresh z. num_samples > 1 yields
    # *different* outputs from the same mask, unlike pix2pix.
    z = torch.randn(num_samples, G.z_dim, device=device, dtype=mask_t.dtype)
    fake = G(mask_t, z=z)
    outs = []
    for i in range(num_samples):
        pil = _tensor_to_pil(fake[i])
        if watermark:
            pil = _watermark(pil)
        outs.append(pil)
    return outs


@torch.no_grad()
def _sample_ldm(mask: Image.Image, modality: str, num_samples: int,
                seed: Optional[int], num_inference_steps: int,
                guidance_scale: float, device: str,
                watermark: bool) -> list[Image.Image]:
    import pasd_ldm as ldm
    vae, unet = _get_ldm(modality, device=device)

    mask_t = _mask_to_tensor(mask, size=256, device=device)  # (1,1,256,256)
    # Expand mask across batch — each replica gets a different random latent
    mask_t = mask_t.expand(num_samples, -1, -1, -1).contiguous()

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(int(seed))

    images = ldm.sample(unet, vae, mask_t, modality=modality,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=gen, device=device)
    outputs = []
    for i in range(num_samples):
        pil = _tensor_to_pil(images[i])
        if watermark:
            pil = _watermark(pil)
        outputs.append(pil)
    return outputs


@torch.no_grad()
def sample_paired(base_mask: Image.Image, modality: str = "BTFE",
                  model: str = "pix2pix", num_samples: int = 4,
                  seed: Optional[int] = None,
                  device: str = "cuda",
                  watermark: bool = True,
                  num_inference_steps: int = 25,
                  guidance_scale: float = 1.5
                  ) -> list[tuple[Image.Image, Image.Image]]:
    """Generate N novel (mask, image) pairs by perturbing the base mask.

    Each sample applies a different deterministic perturbation to the base
    mask, then runs the chosen generator on it. The returned mask is the
    actual GT for the synthetic image.
    """
    base_seed = int(seed) if seed is not None else int(
        torch.randint(0, 2**31 - 1, (1,)).item())
    pairs = []
    for i in range(num_samples):
        per_seed = base_seed + i
        p_mask = perturb_mask(base_mask, seed=per_seed, size=256)
        # Reuse the single-mask path — one sample per call keeps the API simple
        imgs = sample(p_mask, modality=modality, model=model, num_samples=1,
                      seed=per_seed, device=device, watermark=watermark,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale)
        pairs.append((p_mask, imgs[0]))
    return pairs


def encode_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
