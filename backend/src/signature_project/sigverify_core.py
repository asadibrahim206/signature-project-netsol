"""sigverify_core.py - preprocessing, Siamese model and inference helpers.

The same code is used for training (notebook) and, later, for serving (FastAPI).
Needs: torch, timm, numpy, Pillow.  Licences: timm Apache-2.0; ConvNeXt code MIT.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------- preprocessing
def _to_gray(src):
    """path or PIL image -> uint8 grayscale array (transparent PNGs are composited on white)."""
    im = src if isinstance(src, Image.Image) else Image.open(src)
    if im.mode in ("I", "I;16", "F"):
        a = np.asarray(im, dtype=np.float32)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0
        return a.astype(np.uint8)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return np.asarray(im.convert("L"), dtype=np.uint8)


def _otsu(g):
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    sb = np.where(np.isfinite(sb), sb, -1.0)
    return int(np.argmax(sb)) if sb.max() > 0 else 127


def preprocess(src, h=192, w=384, margin=6):
    """Grayscale -> ink on white -> crop to the ink -> contrast stretch -> letterbox to (h, w). Returns uint8 [h, w]."""
    g = _to_gray(src)
    if np.median(g) < 110:                       # light ink on a dark background: flip so ink is dark
        g = 255 - g
    thr = _otsu(g)
    ink = g <= thr
    if ink.mean() > 0.5:                         # ink must be the minority class
        ink = ~ink
    if ink.sum() < 10:
        y0, y1, x0, x1 = 0, g.shape[0], 0, g.shape[1]
    else:
        cols = ink.sum(0).astype(np.float64)
        rows = ink.sum(1).astype(np.float64)
        cx, cy = np.cumsum(cols) / cols.sum(), np.cumsum(rows) / rows.sum()
        x0, x1 = int(np.searchsorted(cx, 0.002)), int(np.searchsorted(cx, 0.998)) + 1   # percentile crop ignores specks
        y0, y1 = int(np.searchsorted(cy, 0.002)), int(np.searchsorted(cy, 0.998)) + 1
        px, py = int(0.02 * (x1 - x0)) + 2, int(0.05 * (y1 - y0)) + 2
        x0, x1 = max(0, x0 - px), min(g.shape[1], x1 + px)
        y0, y1 = max(0, y0 - py), min(g.shape[0], y1 + py)
    c = g[y0:y1, x0:x1].astype(np.float32)
    bg, fg = float(np.percentile(c, 90)), float(np.percentile(c, 1))
    if bg - fg > 20:                             # contrast stretch: background -> 255, darkest ink -> 0
        c = np.clip((c - fg) / (bg - fg), 0.0, 1.0) * 255.0
    ch, cw = c.shape
    scale = min((h - 2 * margin) / ch, (w - 2 * margin) / cw)
    nh, nw = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
    im = Image.fromarray(c.astype(np.uint8)).resize((nw, nh), Image.LANCZOS if scale < 1 else Image.BILINEAR)
    out = np.full((h, w), 255, np.uint8)
    top, left = (h - nh) // 2, (w - nw) // 2
    out[top:top + nh, left:left + nw] = np.asarray(im)
    return out


# ---------------------------------------------------------------- model
class SigVerifier(nn.Module):
    """Siamese network: shared backbone -> embedding; a symmetric head scores 'same writer, genuine'."""

    def __init__(self, backbone="convnext_tiny.fb_in22k_ft_in1k", embed_dim=256, pretrained=True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=1)
        nf = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(nf, embed_dim), nn.LayerNorm(embed_dim))
        self.head = nn.Sequential(nn.Linear(embed_dim * 3, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def embed(self, x):                          # x: float [B, 1, H, W] in 0..1
        return self.proj(self.backbone((x - 0.5) / 0.5))

    def forward(self, xa, xb):                   # returns logits [B, 1]; head(a, b) == head(b, a)
        e = self.embed(torch.cat([xa, xb], 0))
        a, b = e.chunk(2, 0)
        return self.head(torch.cat([(a - b).abs(), a * b, a + b], 1))


# ---------------------------------------------------------------- checkpoint + inference
def save_checkpoint(path, model, cfg):
    """cfg: plain python types only (backbone, embed_dim, img_h, img_w, threshold, ...)."""
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def load_checkpoint(path, device="cpu"):
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except TypeError:                            # very old torch without weights_only
        ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = SigVerifier(cfg["backbone"], cfg["embed_dim"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), cfg


def _to_tensor(arrs, device):
    return torch.from_numpy(np.stack(arrs)).float().div_(255.0).unsqueeze(1).to(device)


@torch.no_grad()
def verify(model, cfg, references, query, device="cpu"):
    """Compare one query signature with one or more GENUINE reference signatures (paths or PIL images).

    Returns {"score": mean genuine-probability, "is_genuine": score >= threshold, "per_reference": [...]}.
    """
    model.eval()
    h, w = cfg["img_h"], cfg["img_w"]
    refs = [preprocess(r, h, w) for r in references]
    q = preprocess(query, h, w)
    p = torch.sigmoid(model(_to_tensor(refs, device), _to_tensor([q] * len(refs), device)).float()).squeeze(1).cpu().numpy()
    score = float(p.mean())
    return {"score": score, "is_genuine": bool(score >= cfg["threshold"]), "per_reference": [float(v) for v in p]}
