"""Smoke: strengths 0.5/0.7/0.85 (+ s=1 reference) on a few scenes, canny+depth sum. Writes ../out/init_sim_<stem>.png"""
import json, sys, time
from pathlib import Path
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hint_fusion import FusionConfig, load_control  # noqa: E402
from init_sim import ZImageInitSim  # noqa: E402
from prompts import label_prompt  # noqa: E402

DATA = HERE.parents[2] / "data"
OUT = HERE.parent / "out"
stems = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sample_00000", "sample_00003"]
mm = sys.argv[2] if len(sys.argv) > 2 else "model_full_load"
cfg = FusionConfig(mode="sum")
m = ZImageInitSim(cfg, memory_mode=mm)
ctrl = lambda k, s: load_control(DATA / ("depth/png" if k == "depth" else k) / f"{s}_{k}.png", DATA / "instance_seg" / f"{s}_instance_seg.png")
res = {}
for stem in stems:
    sim = Image.open(DATA / "images" / f"{stem}_sim.png").convert("RGB")
    real = Image.open(DATA / "real" / f"{stem}_real.png").convert("RGB") if (DATA / "real" / f"{stem}_real.png").exists() else sim
    c = {k: ctrl(k, stem) for k in cfg.modalities}
    prompt = label_prompt(json.loads((DATA / "labels" / f"{stem}.json").read_text()))
    seg = Image.open(DATA / "instance_seg" / f"{stem}_instance_seg.png")
    m.translate(c, prompt, sim.size, steps=2, seg=seg)
    tiles = [("sim", sim), ("real", real)]
    for s in (None, 0.85, 0.7, 0.5):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
        out = m.translate(c, prompt, sim.size, scale=0.8, seed=1, seg=seg, sim=sim, strength=s)
        torch.cuda.synchronize(); dt = time.perf_counter() - t
        name = "s=1(base-fused)" if s is None else f"s={s} sigma0={m.last_sigma:.3f}"
        res[f"{stem}/{name}"] = {"sec": round(dt, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(stem, name, res[f"{stem}/{name}"], flush=True)
        tiles.append((name, out))
    W, H = sim.size
    sheet = Image.new("RGB", (W * 3, H * 2))
    for i, (n, im) in enumerate(tiles):
        im = im.resize((W, H)); ImageDraw.Draw(im).text((8, 8), n, fill=(255, 255, 0))
        sheet.paste(im, ((i % 3) * W, (i // 3) * H))
    sheet.save(OUT / f"init_sim_{stem}.png")
(OUT / "init_sim_smoke.json").write_text(json.dumps(res, indent=2))
