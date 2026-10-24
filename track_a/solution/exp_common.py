"""Shared harness for Track A experiments: builds a submission folder from any `gen_fn`, optionally evaluates, records.

    from exp_common import run_submission
    run_submission("myexp_v1", gen_fn, config={...}, n=60, evaluate=True)

gen_fn(ctx) -> PIL.Image (any size; resized to sim size). ctx: stem, sim (PIL RGB), label (dict), seed, idx,
canny/depth/seg paths (ctx.paths). Times only gen_fn (include all model calls inside it). Warm-up: call gen_fn once on scene 0 first (untimed).
Run under tools/gpu_run.sh. Never writes real photos into generated/ (data-leakage rule, research/PROTOCOL.md).
"""
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
TRACK_A = ROOT / "track_a"
sys.path.insert(0, str(HERE))
from prompts import GENERIC, label_prompt  # noqa: E402,F401


@dataclass
class Ctx:
    stem: str
    idx: int
    seed: int
    sim: Image.Image
    label: dict

    @property
    def paths(self):
        s = self.stem
        return {"canny": DATA / "canny" / f"{s}_canny.png", "depth": DATA / "depth/png" / f"{s}_depth.png",
                "seg": DATA / "instance_seg" / f"{s}_instance_seg.png", "sim": DATA / "images" / f"{s}_sim.png",
                "real_do_not_submit": DATA / "real" / f"{s}_real.jpg"}


def all_stems():
    return [p.name[: -len("_sim.png")] for p in sorted((DATA / "images").glob("*_sim.png"))]


def stems_with_real():
    return all_stems()[:60]


def run_submission(name, gen_fn, config=None, n=60, seed=0, evaluate=True, stems=None, failure_model="qwen"):
    sub = TRACK_A / "submissions" / name
    assert not (sub / "generated").exists() or not any((sub / "generated").iterdir()), f"{name} exists; use a new name"
    for d in ("simulated", "generated", "reference", "labels"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    for p in sorted((DATA / "real").iterdir()):
        shutil.copy(p, sub / "reference" / p.name)
    stems = (stems or all_stems())[:n]

    def ctx(i, stem):
        return Ctx(stem, i, seed + i, Image.open(DATA / "images" / f"{stem}_sim.png").convert("RGB"),
                   json.loads((DATA / "labels" / f"{stem}.json").read_text()))

    gen_fn(ctx(0, stems[0]))  # warm-up, untimed
    total = 0.0
    for i, stem in enumerate(stems, 1):
        c = ctx(i, stem)
        t = time.perf_counter()
        out = gen_fn(c)
        total += time.perf_counter() - t
        out.convert("RGB").resize(c.sim.size, Image.LANCZOS).save(sub / "generated" / f"{stem}.png")
        shutil.copy(DATA / "images" / f"{stem}_sim.png", sub / "simulated" / f"{stem}_sim.png")
        shutil.copy(DATA / "labels" / f"{stem}.json", sub / "labels" / f"{stem}_label.json")
        print(f"{i}/{len(stems)} {stem} {total / i:.2f}s/img", flush=True)
    (sub / "generation_times.json").write_text(json.dumps({"total_seconds": total, "scenes": len(stems), "hardware": "NVIDIA L40S"}, indent=2))
    (sub / "config.json").write_text(json.dumps({"name": name, "n": n, "seed": seed, **(config or {})}, indent=2, default=str))
    if evaluate:
        subprocess.run([sys.executable, "evaluator_cli.py", "evaluate", "--submission-path", str(sub),
                        "--output", str(TRACK_A / "track_a_evaluation_output" / name / "report.json"),
                        "--failure-model", failure_model, "--device", "cuda"], cwd=TRACK_A, check=True)
    return sub
