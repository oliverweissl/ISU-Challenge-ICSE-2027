"""Build reference/perturbation submissions (ref_*) for evaluator-analyst runs.

    python build_ref_submissions.py id_sim real_rs ...   (names below)
Layout copied from run_experiment.py: simulated/ generated/ reference/ labels/ + generation_times.json
"""
import json, shutil, sys, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

ROOT = Path(__file__).resolve().parents[3]
DATA, SUBS = ROOT / "data", ROOT / "track_a" / "submissions"
N = 60


def noise(sigma, seed):
    def f(sim, real, i):
        rng = np.random.default_rng(seed * 1000 + i)
        a = np.asarray(sim, np.float32) + rng.normal(0, sigma, (sim.size[1], sim.size[0], 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return f


def jpeg(q):
    def f(sim, real, i):
        b = io.BytesIO(); real.save(b, "JPEG", quality=q); b.seek(0)
        return Image.open(b).convert("RGB")
    return f


def jitter(sim, real, i):
    rng = np.random.default_rng(i)
    im = ImageEnhance.Brightness(sim).enhance(rng.uniform(0.6, 1.4))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.6, 1.4))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.4, 1.6))
    h = Image.fromarray(np.asarray(im.convert("HSV")).copy())
    a = np.asarray(h).copy(); a[..., 0] = (a[..., 0].astype(int) + int(rng.uniform(-20, 20))) % 256
    return Image.fromarray(a, "HSV").convert("RGB")


VARIANTS = {
    "id_sim": lambda sim, real, i: sim,
    "real_rs": lambda sim, real, i: real.resize(sim.size, Image.LANCZOS),
    "real_jpeg": jpeg(30),                      # full-res real re-encoded (same size as real)
    "sim_blur": lambda sim, real, i: sim.filter(ImageFilter.GaussianBlur(6)),
    "sim_jitter": jitter,
    "sim_noise30": noise(30, 0),
    "sim_noise3_s0": noise(3, 0),
    "sim_noise3_s1": noise(3, 1),
    "shuf_real": None,                          # real of the next scene, resized (gaming)
    "black": lambda sim, real, i: Image.new("RGB", sim.size, (0, 0, 0)),
    "blend": lambda sim, real, i: Image.blend(sim, real.resize(sim.size, Image.LANCZOS), 0.5),
    "sim_gray": lambda sim, real, i: sim.convert("L").convert("RGB"),
}


def build(name):
    sub = SUBS / f"ref_{name}"
    for d in ("simulated", "generated", "reference", "labels"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    for p in sorted((DATA / "real").iterdir()):
        shutil.copy(p, sub / "reference" / p.name)
    stems = [p.name[:-len("_sim.png")] for p in sorted((DATA / "images").glob("*_sim.png"))][:N]
    reals = [Image.open(DATA / "real" / f"{s}_real.jpg").convert("RGB") for s in stems]
    for i, s in enumerate(stems):
        sim = Image.open(DATA / "images" / f"{s}_sim.png").convert("RGB")
        real = reals[i]
        if name == "shuf_real":
            out = reals[(i + 1) % N].resize(sim.size, Image.LANCZOS)
        else:
            out = VARIANTS[name](sim, real, i)
        out.save(sub / "generated" / f"{s}.png")
        shutil.copy(DATA / "images" / f"{s}_sim.png", sub / "simulated" / f"{s}_sim.png")
        shutil.copy(DATA / "labels" / f"{s}.json", sub / "labels" / f"{s}_label.json")
    (sub / "generation_times.json").write_text(json.dumps({"total_seconds": 0.0, "scenes": N, "hardware": "n/a"}))
    print("built", sub)


if __name__ == "__main__":
    for n in sys.argv[1:] or VARIANTS:
        build(n)
