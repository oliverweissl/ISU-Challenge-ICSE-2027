"""Smoke test: every fusion mode + single-modality references on a few scenes. Writes side-by-side images to ../out/."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hint_fusion import FusionConfig, ZImageControlHintFusion, load_control  # noqa: E402
from prompts import label_prompt  # noqa: E402

DATA = HERE.parents[2] / "data"
OUT = HERE.parent / "out"
OUT.mkdir(exist_ok=True)
stems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sample_00000", "sample_00003"]
memory_mode = sys.argv[2] if len(sys.argv) > 2 else "model_full_load"

variants = {
    "canny_only(w=1,0)": dict(mode="sum", weights={"canny": 1.0, "depth": 0.0}),
    "depth_only(w=0,.5)": dict(mode="sum", weights={"canny": 0.0, "depth": 0.5}),
    "sum": dict(mode="sum"),
    "mean": dict(mode="mean"),
    "region_gated": dict(mode="region_gated"),
    "phase_gated": dict(mode="phase_gated"),
    "conflict_aware": dict(mode="conflict_aware"),
}


def ctrl(kind, stem):
    return load_control(DATA / ("depth/png" if kind == "depth" else kind) / f"{stem}_{kind}.png",
                        DATA / "instance_seg" / f"{stem}_instance_seg.png")


m = ZImageControlHintFusion(FusionConfig(), memory_mode=memory_mode)
res = {}
for stem in stems:
    label = json.loads((DATA / "labels" / f"{stem}.json").read_text())
    sim = Image.open(DATA / "images" / f"{stem}_sim.png").convert("RGB")
    seg = Image.open(DATA / "instance_seg" / f"{stem}_instance_seg.png")
    prompt = label_prompt(label)
    cs = {k: ctrl(k, stem) for k in ("canny", "depth")}
    tiles = [("sim", sim), ("canny", cs["canny"]), ("depth", cs["depth"])]
    if stem == stems[0]:
        m.translate(cs, prompt, sim.size, steps=2, seg=seg)  # warmup
    for name, kw in variants.items():
        m.cfg = FusionConfig(**kw)
        torch.cuda.synchronize(m.device)
        torch.cuda.reset_peak_memory_stats(m.device)
        t = time.perf_counter()
        img = m.translate(cs, prompt, sim.size, scale=0.8, seed=1, seg=seg)
        torch.cuda.synchronize(m.device)
        dt = time.perf_counter() - t
        peak = torch.cuda.max_memory_allocated(m.device) / 2**30
        extra = f" conflict_frac={np.mean(m.st.conflict_frac):.4f} mean_cos={np.mean(m.st.cos_mean):.3f} per-layer-frac-max={np.max(m.st.conflict_frac):.4f}" if m.st.conflict_frac else ""
        print(f"{stem} {name}: {dt:.2f}s peak {peak:.1f}GiB{extra}", flush=True)
        res[f"{stem}/{name}"] = dict(s=dt, peak_gib=peak)
        tiles.append((name, img))
    w, h = sim.size
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows))
    for i, (n, im) in enumerate(tiles):
        im = im.copy()
        ImageDraw.Draw(im).text((8, 8), n, fill=(255, 255, 0))
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(OUT / f"hint_fusion_{stem}.png")
    # identity check vs baseline path (same seed): canny_only should equal ZImageControlBaseline output
    from baseline_zimage_cn import ZImageControlBaseline
    base = ZImageControlBaseline.translate(m, cs["canny"], prompt, sim.size, scale=0.8, seed=1)
    d = np.abs(np.asarray(base, np.int16) - np.asarray(tiles[3][1], np.int16)).max()
    print(f"{stem} identity check canny_only vs baseline translate: max abs diff {d}")
    res[f"{stem}/identity_maxdiff"] = int(d)
(OUT / "hint_fusion_smoke.json").write_text(json.dumps(res, indent=2))
