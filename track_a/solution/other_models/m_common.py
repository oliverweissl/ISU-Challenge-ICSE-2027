"""Shared helpers for family M (other generative models): control loading, prompts, run wrapper."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOL = HERE.parent
sys.path.insert(0, str(SOL))
from exp_common import DATA, run_submission  # noqa: E402,F401
from prompts import label_prompt  # noqa: E402,F401

NEG = "cartoon, 3d render, cgi, painting, illustration, drawing, oversaturated, plastic skin, deformed, blurry"
NEG_I2I = NEG


def load_canny(stem):
    return Image.open(DATA / "canny" / f"{stem}_canny.png").convert("RGB")


def load_depth(stem):
    """16-bit depth PNG (dataset: bright = far, exterior = noise) -> 8-bit near=bright RGB map; exterior (seg colour
    8,63,122) set to far, rest percentile-stretched. Same recipe as hint_fusion.method.load_control (re-verified: raw
    I;16 saturates under convert('RGB'))."""
    d = np.asarray(Image.open(DATA / "depth/png" / f"{stem}_depth.png"), dtype=np.float32) / 65535.0
    seg = np.asarray(Image.open(DATA / "instance_seg" / f"{stem}_instance_seg.png").convert("RGB"), dtype=np.int32)
    ext = np.abs(seg - np.array([8, 63, 122])).sum(-1) == 0
    v = d[~ext] if (~ext).any() else d.ravel()
    lo, hi = np.percentile(v, 1), np.percentile(v, 99)
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    n[ext] = 1.0
    return Image.fromarray(((1 - n) * 255).astype(np.uint8)).convert("RGB")


def snap(repo):
    """Local snapshot dir of a (partially, pattern-) downloaded repo; snapshot_download(local_files_only) rejects those."""
    import os
    root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")) / ("models--" + repo.replace("/", "--"))
    return str(root / "snapshots" / (root / "refs" / "main").read_text().strip())


def fit(img, size):
    return img.resize(size, Image.LANCZOS if img.size[0] > size[0] else Image.BICUBIC)


def run(name, gen_fn, config, n=60, evaluate=True):
    import torch
    config = dict(config)
    config["torch"] = torch.__version__
    return run_submission(name, gen_fn, config=config, n=n, evaluate=evaluate)


# ---- single-feature perturbation: seat colour -------------------------------------------------------------
SEAT_COLORS = {"black": (0.10, 0.10, 0.11), "red": (0.75, 0.10, 0.10), "blue": (0.12, 0.22, 0.70), "brown": (0.45, 0.25, 0.10),
               "green": (0.12, 0.50, 0.20)}
SEAT_CYCLE = ["red", "blue", "black", "brown", "green"]


def seat_color_for(stem):
    from exp_common import all_stems
    return SEAT_CYCLE[all_stems().index(stem) % len(SEAT_CYCLE)]


def seat_mask(stem):
    a = np.asarray(Image.open(DATA / "instance_seg" / f"{stem}_instance_seg.png").convert("RGB"), dtype=np.int32)
    return np.abs(a - np.array([38, 15, 3])).sum(-1) == 0


def recolor_seats(sim, stem, color):
    """Programmatic edit of the sim image: keep luminance shading, replace seat colour (seg class 'seat'). Quantity/presence untouched."""
    m = seat_mask(stem)
    x = np.asarray(sim.convert("RGB"), dtype=np.float32) / 255
    L = x.mean(-1, keepdims=True)
    c = np.array(SEAT_COLORS[color], dtype=np.float32)
    new = np.clip(L * (c / c.mean()) * 1.0 if color != "black" else L * 0.45, 0, 1)
    out = np.where(m[..., None], new, x)
    return Image.fromarray((out * 255).astype(np.uint8))
