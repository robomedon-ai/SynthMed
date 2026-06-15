"""CUT (Contrastive Unpaired Translation) building blocks for T2 → ADC.

Park et al. 2020, "Contrastive Learning for Unpaired Image-to-Image Translation".
Here used in a PAIRED flavor (AI-ADC-style): the PatchNCE contrastive loss
enforces content correspondence (and is robust to the EPI geometric distortion
between T2 and ADC), while a conditional PatchGAN gives realism and a modest L1
anchor exploits the pairing without the mean-blur that caps plain pix2pix.

Components:
  - ResnetGenerator: standard CUT ResNet-9 generator, with a layered forward
    that can return intermediate encoder features for the contrastive loss.
  - PatchSampleF: per-layer MLP head that samples + projects + L2-normalizes
    feature patches (lazily builds MLPs on first call).
  - PatchNCELoss: InfoNCE over corresponding patches (positive = same spatial
    location across input/output; negatives = other patches in the same image).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Generator (CUT ResNet-9, layered forward) ──

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), norm_layer(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), norm_layer(dim),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """ResNet generator whose `model` is a flat nn.Sequential so we can index
    layers for feature extraction (CUT nce_layers index into this Sequential)."""

    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9,
                 norm_layer=nn.InstanceNorm2d):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7), norm_layer(ngf), nn.ReLU(True),
        ]
        # downsample x2
        mult = 1
        for _ in range(2):
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                norm_layer(ngf * mult * 2), nn.ReLU(True),
            ]
            mult *= 2
        # resnet blocks
        for _ in range(n_blocks):
            layers += [ResnetBlock(ngf * mult, norm_layer)]
        # upsample x2
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2,
                                   padding=1, output_padding=1),
                norm_layer(ngf * mult // 2), nn.ReLU(True),
            ]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x, layers=None, encode_only=False):
        if layers:
            feats = []
            feat = x
            for i, layer in enumerate(self.model):
                feat = layer(feat)
                if i in layers:
                    feats.append(feat)
                if i == layers[-1] and encode_only:
                    return feats
            return feat, feats
        return self.model(x)


# Default NCE layer indices into the Sequential above
# 0=input pad, 4=after 1st downsample relu, 8=after 2nd downsample relu,
# 12 & 16 = inside resnet blocks.
DEFAULT_NCE_LAYERS = [0, 4, 8, 12, 16]


# ── Patch sampler (MLP head) ──

class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=True, nc=256):
        super().__init__()
        self.use_mlp = use_mlp
        self.nc = nc
        self.mlp_init = False
        self.mlps = nn.ModuleList()

    def _build_mlps(self, feats):
        for feat in feats:
            C = feat.shape[1]
            self.mlps.append(nn.Sequential(
                nn.Linear(C, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)))
        self.mlp_init = True

    def forward(self, feats, num_patches=256, patch_ids=None):
        return_feats, return_ids = [], []
        if self.use_mlp and not self.mlp_init:
            self._build_mlps(feats)
            self.to(feats[0].device)
        for i, feat in enumerate(feats):
            B, C, H, W = feat.shape
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
            if num_patches > 0:
                if patch_ids is not None:
                    pid = patch_ids[i]
                else:
                    pid = torch.randperm(H * W, device=feat.device)[:num_patches]
                sample = feat_flat[:, pid, :].reshape(-1, C)   # (B*np, C)
            else:
                sample = feat_flat.reshape(-1, C); pid = None
            if self.use_mlp:
                sample = self.mlps[i](sample)
            sample = F.normalize(sample, dim=1)
            return_feats.append(sample)
            return_ids.append(pid)
        return return_feats, return_ids


# ── PatchNCE loss ──

class PatchNCELoss(nn.Module):
    def __init__(self, nce_T=0.07):
        super().__init__()
        self.nce_T = nce_T
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat_q, feat_k):
        """feat_q, feat_k: (B*num_patches, C). Positive = same index; negatives =
        other patches within the same image (block-diagonal masking)."""
        n = feat_q.shape[0]
        feat_k = feat_k.detach()
        # positive logit: per-patch dot product
        l_pos = (feat_q * feat_k).sum(dim=1, keepdim=True)            # (n, 1)
        # negative logits: within-image patch-vs-patch. Infer batch from caller
        # via the module attribute set before the call.
        npatch = self.num_patches
        B = n // npatch
        q = feat_q.view(B, npatch, -1)
        k = feat_k.view(B, npatch, -1)
        l_neg = torch.bmm(q, k.transpose(1, 2))                       # (B, np, np)
        # mask out the diagonal (the positive) with -inf
        diag = torch.eye(npatch, device=feat_q.device, dtype=torch.bool)[None, :, :]
        l_neg.masked_fill_(diag, -1e9)
        l_neg = l_neg.reshape(B * npatch, npatch)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.nce_T        # (n, 1+np)
        target = torch.zeros(n, dtype=torch.long, device=feat_q.device)
        return self.ce(logits, target).mean()


def init_weights(net, gain=0.02):
    def fn(m):
        cn = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cn or "Linear" in cn):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
    net.apply(fn)
