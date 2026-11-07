"""Signature verification core: preprocessing + Siamese model + inference helper.

Shared by the Kaggle training notebook and (later) the FastAPI service, so there is no
train/serve skew. Only Apache-2.0 / MIT code and weights are used (timm backbone).
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps


# ----------------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------------
def to_gray_array(img):
    """PIL image / path -> uint8 grayscale array, dark ink on light background."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    a = np.asarray(img.convert("L"), dtype=np.uint8)
    if np.median(a) < 110:          # dark background / light ink -> invert
        a = 255 - a
    return a


def otsu_threshold(a):
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    levels = np.arange(256, dtype=np.float64)
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    s0 = np.cumsum(levels * hist)
    m0 = s0 / np.maximum(w0, 1)
    m1 = (s0[-1] - s0) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return int(np.argmax(between))


def crop_ink(a, margin=0.05):
    """Crop to the ink bounding box (+ margin). Falls back to the full image if ink is implausible."""
    ink = a < otsu_threshold(a)
    frac = float(ink.mean())
    if frac < 1e-4 or frac > 0.5:
        return a
    cols = np.where(ink.sum(0) >= 2)[0]
    rows = np.where(ink.sum(1) >= 2)[0]
    if len(cols) == 0 or len(rows) == 0:
        return a
    y0, y1, x0, x1 = rows[0], rows[-1], cols[0], cols[-1]
    pad = int(round(margin * max(y1 - y0 + 1, x1 - x0 + 1)))
    y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
    y1, x1 = min(a.shape[0] - 1, y1 + pad), min(a.shape[1] - 1, x1 + pad)
    return a[y0:y1 + 1, x0:x1 + 1]


def stretch_contrast(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < 10:
        return a
    out = (a.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def letterbox(a, out_h, out_w, pad_value=255):
    """Resize keeping aspect ratio, pad with white to (out_h, out_w)."""
    h, w = a.shape
    scale = min(out_h / h, out_w / w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    im = Image.fromarray(a).resize((nw, nh), Image.LANCZOS)
    canvas = np.full((out_h, out_w), pad_value, np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = np.asarray(im)
    return canvas


def preprocess(img, out_h, out_w, margin=0.05):
    """Any signature image -> uint8 [out_h, out_w], dark ink on white."""
    a = to_gray_array(img)
    a = crop_ink(a, margin)
    a = stretch_contrast(a)
    return letterbox(a, out_h, out_w)


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
class SigVerifier(nn.Module):
    """Siamese verifier: shared timm backbone -> L2-normalised embedding -> symmetric pair head.

    forward(xa, xb) returns a logit; sigmoid(logit) = P(both signatures are from the same, genuine hand).
    Inputs are [B, 1, H, W] floats in [0, 1] (white background = 1).
    """

    def __init__(self, backbone="convnext_tiny.fb_in22k_ft_in1k", embed_dim=256, pretrained=True, drop=0.1):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat = self.backbone.num_features
        cfgm = getattr(self.backbone, "pretrained_cfg", {}) or {}
        mean = cfgm.get("mean", (0.485, 0.456, 0.406))
        std = cfgm.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.proj = nn.Sequential(nn.Linear(feat, embed_dim), nn.LayerNorm(embed_dim))
        self.head = nn.Sequential(nn.Linear(2 * embed_dim + 1, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 1))
        self.mean_list, self.std_list = [float(m) for m in mean], [float(s) for s in std]

    def embed(self, x):
        x = x.expand(-1, 3, -1, -1)
        x = (x - self.mean) / self.std
        return F.normalize(self.proj(self.backbone(x)), dim=-1)

    def pair_logit(self, ea, eb):
        cos = (ea * eb).sum(-1, keepdim=True)
        feat = torch.cat([(ea - eb).abs(), ea * eb, cos], dim=-1)      # symmetric in (a, b)
        return self.head(feat).squeeze(-1)

    def forward(self, xa, xb):
        b = xa.shape[0]
        e = self.embed(torch.cat([xa, xb], dim=0))                     # one pass for both images
        return self.pair_logit(e[:b], e[b:])


# ----------------------------------------------------------------------------------------
# Checkpoint + inference helpers
# ----------------------------------------------------------------------------------------
def save_checkpoint(path, model, cfg):
    """cfg must contain only plain python types (backbone, embed_dim, img_h, img_w, threshold, ...)."""
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = SigVerifier(cfg["backbone"], cfg["embed_dim"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


def _to_tensor(arrs, device):
    return torch.from_numpy(np.stack(arrs)).float().div_(255.0).unsqueeze(1).to(device)


@torch.no_grad()
def verify(model, cfg, references, query, device="cpu"):
    """Compare one query signature against one or more genuine reference signatures.

    references / query: file paths or PIL images. Returns the mean genuine-probability over references.
    """
    model.eval()
    h, w = cfg["img_h"], cfg["img_w"]
    refs = [preprocess(r, h, w) for r in references]
    q = preprocess(query, h, w)
    xr = _to_tensor(refs, device)
    xq = _to_tensor([q] * len(refs), device)
    p = torch.sigmoid(model(xr, xq).float()).cpu().numpy()
    score = float(p.mean())
    return {"score": score, "is_genuine": bool(score >= cfg["threshold"]), "per_reference": [float(v) for v in p]}
