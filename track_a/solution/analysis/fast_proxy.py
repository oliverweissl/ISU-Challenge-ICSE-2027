"""Fast proxy for the Track A evaluator (no VLM). Target < 1 min on a GPU for 60 scenes.

    python fast_proxy.py <submission_dir> [--out proxy.json]

Outputs
  sam_proxy   : 1 - matched_iou between SAM partitions of simulated and generated, using the evaluator's own
                partition_metrics, but a coarser point grid (points_per_crop=12) and 480x270 inputs.
                Simulated partitions are cached in analysis/cache/ (keyed by pixel hash) so only generated runs SAM.
  kid_r50     : unbiased polynomial-kernel MMD (KID) between ResNet50 pooled features of generated and real refs.
  fd_diag     : diagonal Frechet distance on the same features (stable at n=60).
  hf_ratio    : gradient-energy of generated / real (1 = same sharpness; <1 blur; >1 noise).
  col_emd     : mean over RGB channels of Wasserstein between pixel histograms of generated and real.
  fail_proxy  : mean (1 - cosine) between ResNet50 embeddings of generated and its own real ref, per scene
                (VLM-free surrogate of semantic mismatch, NOT calibrated to Qwen).
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "track_a"))
CACHE = HERE / "cache"
SZ = (480, 270)


def stems_of(sub):
    ref = {p.name[:-len("_real.jpg")] for p in (sub / "reference").glob("*_real.jpg")}
    gen = {p.stem for p in (sub / "generated").glob("*.png")}
    return sorted(ref & gen)


def sam_proxy(sub, stems, dev):
    from transformers import pipeline
    from image_metrics_utils import partition_metrics
    gen = pipeline("mask-generation", model="facebook/sam-vit-base", device=0 if dev == "cuda" else -1)

    def part(img):
        out = gen(img, points_per_batch=64, points_per_crop=12)
        masks = sorted((np.asarray(m, bool) for m in out["masks"]), key=np.count_nonzero, reverse=True)
        lab = np.zeros((img.height, img.width), np.int64)
        for i, m in enumerate(masks, 1):
            lab[m] = i
        return lab
    CACHE.mkdir(exist_ok=True)
    scores = []
    for s in stems:
        sim = Image.open(sub / "simulated" / f"{s}_sim.png").convert("RGB").resize(SZ, Image.BILINEAR)
        h = hashlib.sha256(np.asarray(sim).tobytes()).hexdigest()[:16]
        f = CACHE / f"sim_{h}.npy"
        if f.exists():
            a = np.load(f)
        else:
            a = part(sim); np.save(f, a)
        g = Image.open(sub / "generated" / f"{s}.png").convert("RGB").resize(SZ, Image.BILINEAR)
        gh = hashlib.sha256(np.asarray(g).tobytes()).hexdigest()[:16]
        if gh == h:
            b = a
        else:
            b = part(g)
        scores.append(1.0 - partition_metrics(a, b, -1)["matched_iou"])
    return float(np.mean(scores))


def feats(paths, dev):
    import torch, torchvision
    from torchvision.models import ResNet50_Weights, resnet50
    w = ResNet50_Weights.IMAGENET1K_V1
    m = resnet50(weights=w); m.fc = torch.nn.Identity(); m = m.to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(paths), 16):
        x = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((384, 216), Image.BILINEAR)).copy()).permute(2, 0, 1) for p in paths[i:i + 16]]).to(dev).float() / 255
        with torch.inference_mode():
            out.append(m((x - mean) / std).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)


def kid(x, y, subsets=50, size=None, seed=0):
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y)); size = size or min(n, 30)
    d = x.shape[1]; vals = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), size, replace=False)]; b = y[rng.choice(len(y), size, replace=False)]
        k = lambda u, v: (u @ v.T / d + 1) ** 3
        kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
        m = size
        vals.append((kaa.sum() - np.trace(kaa)) / (m * (m - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())
    return float(np.mean(vals))


def grad_energy(p):
    g = np.asarray(Image.open(p).convert("L").resize(SZ, Image.BILINEAR), np.float32)
    return float(np.mean(np.abs(np.diff(g, axis=1))) + np.mean(np.abs(np.diff(g, axis=0))))


def col_emd(gp, rp):
    from scipy.stats import wasserstein_distance
    a = np.asarray(Image.open(gp).convert("RGB").resize(SZ), np.uint8).reshape(-1, 3)
    b = np.asarray(Image.open(rp).convert("RGB").resize(SZ), np.uint8).reshape(-1, 3)
    return float(np.mean([wasserstein_distance(a[:, c], b[:, c]) for c in range(3)]))


def run(sub, dev="cuda"):
    sub = Path(sub); t0 = time.time()
    stems = stems_of(sub)
    gp = [sub / "generated" / f"{s}.png" for s in stems]
    rp = [sub / "reference" / f"{s}_real.jpg" for s in stems]
    res = {"n": len(stems)}
    res["sam_proxy"] = sam_proxy(sub, stems, dev)
    fg, fr = feats(gp, dev), feats(rp, dev)
    res["kid_r50"] = kid(fg, fr)
    res["fd_diag"] = float(np.sum((fg.mean(0) - fr.mean(0)) ** 2) + np.sum((fg.std(0) - fr.std(0)) ** 2))
    res["hf_ratio"] = float(np.mean([grad_energy(p) for p in gp]) / np.mean([grad_energy(p) for p in rp]))
    res["col_emd"] = float(np.mean([col_emd(a, b) for a, b in zip(gp, rp)]))
    n = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    res["fail_proxy"] = float(np.mean(1 - np.sum(n(fg) * n(fr), axis=1)))
    res["seconds"] = time.time() - t0
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sub"); ap.add_argument("--out"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    r = run(a.sub, a.device)
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=2))
