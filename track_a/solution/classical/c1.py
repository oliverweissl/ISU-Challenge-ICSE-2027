"""C1: pure classical sim->real image processing (numpy/PIL/skimage only, no network).

Variants (see research/exp_classical.md):
  hand    : no real data. Hand-set warm tint, saturation boost, S-curve, vignette, blur, grain, JPEG.
  reinhard: _R. Global Lab mean/std matching to pooled stats of real photos 0-39 (Reinhard 2001).
  hist    : _R. Per-channel quantile mapping to the pooled real CDF of scenes 0-39 (per-image sim CDF -> pooled real CDF).
  *_fx    : the above + vignette/blur/grain/JPEG post chain.
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage import color

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[2] / "data"
FIT_STEMS = [f"sample_{i:05d}" for i in range(40)]  # real photos allowed for fitting (rule 4)
CACHE = HERE / "real_stats_0_39.npz"


def real_stats():
    """Pooled stats of real photos 0-39 (downscaled to 480x270): Lab mean/std, per-channel quantile curves."""
    if CACHE.exists():
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    px = []
    for s in FIT_STEMS:
        px.append(np.asarray(Image.open(DATA / "real" / f"{s}_real.jpg").convert("RGB").resize((480, 270), Image.LANCZOS), np.float32) / 255)
    a = np.stack(px)  # N,H,W,3
    lab = color.rgb2lab(a)
    q = np.linspace(0, 100, 257)
    quant = np.stack([np.percentile(a[..., c], q) for c in range(3)])  # 3,257
    out = {"lab_mean": lab.mean((0, 1, 2)), "lab_std": lab.std((0, 1, 2)), "quant": quant, "q": q}
    np.savez(CACHE, **out)
    return out


def reinhard(img, st, strength=1.0):
    a = np.asarray(img, np.float32) / 255
    lab = color.rgb2lab(a)
    m, s = lab.mean((0, 1)), lab.std((0, 1)) + 1e-6
    # match global stats of the *pooled* real (not paired real): same target for every scene
    new = (lab - m) / s * st["lab_std"] + st["lab_mean"]
    new = lab + strength * (new - lab)
    return Image.fromarray((np.clip(color.lab2rgb(new), 0, 1) * 255 + .5).astype(np.uint8))


def hist_match(img, st):
    a = np.asarray(img, np.float32) / 255
    out = np.empty_like(a)
    q = st["q"]
    for c in range(3):
        ch = a[..., c]
        src_q = np.percentile(ch, q)
        # monotone mapping src quantile -> real quantile
        out[..., c] = np.interp(ch, src_q + 1e-6 * np.arange(len(q)), st["quant"][c])
    return Image.fromarray((np.clip(out, 0, 1) * 255 + .5).astype(np.uint8))


def hand(img, tint=(1.06, 1.0, 0.86), sat=1.9, gamma=0.9, contrast=1.25):
    a = np.asarray(img, np.float32) / 255
    a = a ** gamma
    a = np.clip((a - 0.5) * contrast + 0.5, 0, 1)
    g = a.mean(-1, keepdims=True)
    a = np.clip(g + (a - g) * sat, 0, 1)
    a = np.clip(a * np.array(tint, np.float32), 0, 1)
    return Image.fromarray((a * 255 + .5).astype(np.uint8))


def fx(img, seed, vignette=0.25, blur=0.7, grain=3.5, jpeg=65):
    rng = np.random.default_rng(seed)
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(blur)) if blur else img, np.float32)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2)
    a = a * (1 - vignette * r ** 2)[..., None]
    if grain:
        n = rng.normal(0, grain, (h, w, 1)) * (0.5 + 0.5 * (1 - a.mean(-1, keepdims=True) / 255)) + rng.normal(0, grain * .4, (h, w, 3))
        a = a + n
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if jpeg:
        b = io.BytesIO(); im.save(b, "JPEG", quality=jpeg); im = Image.open(b).convert("RGB")
    return im


VARIANTS = {
    "hand": dict(real=False, steps=["hand"]),
    "hand_fx": dict(real=False, steps=["hand", "fx"]),
    "reinhard": dict(real=True, steps=["reinhard"]),
    "hist": dict(real=True, steps=["hist"]),
    "hist_fx": dict(real=True, steps=["hist", "fx"]),
    "reinhard_fx": dict(real=True, steps=["reinhard", "fx"]),
}


def process(sim, variant, seed=0, st=None):
    v = VARIANTS[variant]
    st = st if st is not None else (real_stats() if v["real"] else None)
    im = sim.convert("RGB")
    for s in v["steps"]:
        im = {"hand": hand, "reinhard": lambda i: reinhard(i, st), "hist": lambda i: hist_match(i, st), "fx": lambda i: fx(i, seed)}[s](im)
    return im
