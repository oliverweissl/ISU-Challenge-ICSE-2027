"""Generate a Track A submission folder for the baseline and (optionally) evaluate it.

    python run_experiment.py --name base_canny_label --control canny --prompt label --n 60 --eval
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
TRACK_A = ROOT / "track_a"
sys.path.insert(0, str(HERE))

from prompts import GENERIC, label_prompt  # noqa: E402


def control_path(kind: str, stem: str) -> Path:
    return DATA / ("depth/png" if kind == "depth" else kind) / f"{stem}_{kind}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--control", choices=["canny", "depth"], default="canny")
    ap.add_argument("--prompt", choices=["generic", "label"], default="label")
    ap.add_argument("--scale", type=float, default=0.8)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["baseline", "hint_fusion", "init_sim"], default="baseline")
    ap.add_argument("--init-strength", type=float, default=0.7, help="init_sim: noise level s of the sim latent (1.0 = pure noise)")
    ap.add_argument("--fusion-mode", choices=["sum", "mean", "region_gated", "phase_gated", "conflict_aware"], default="mean")
    ap.add_argument("--fusion-config", default=None, help="JSON file or inline JSON with FusionConfig fields (overrides flags)")
    ap.add_argument("--w-canny", type=float, default=None)
    ap.add_argument("--w-depth", type=float, default=None)
    ap.add_argument("--gate-table", default=None)
    ap.add_argument("--phase-shape", choices=["linear", "step"], default=None)
    ap.add_argument("--conflict-base", choices=["mean", "sum"], default=None)
    ap.add_argument("--memory-mode", default="model_full_load")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--failure-model", default="qwen")
    args = ap.parse_args()

    init_sim = args.method == "init_sim"
    fusion = args.method in ("hint_fusion", "init_sim")
    if fusion:
        from hint_fusion import FusionConfig, ZImageControlHintFusion, load_control
        fc = {"mode": args.fusion_mode}
        if args.w_canny is not None or args.w_depth is not None:
            fc["weights"] = {"canny": 1.0 if args.w_canny is None else args.w_canny, "depth": 0.2 if args.w_depth is None else args.w_depth}
        for k, v in (("gate_table", args.gate_table), ("phase_shape", args.phase_shape), ("conflict_base", args.conflict_base)):
            if v is not None:
                fc[k] = v
        if args.fusion_config:
            fc.update(json.loads(args.fusion_config if args.fusion_config.lstrip().startswith("{") else Path(args.fusion_config).read_text()))
        fcfg = FusionConfig.from_dict(fc)
        args.fusion = fcfg.to_dict()
    else:
        from baseline_zimage_cn import ZImageControlBaseline

    def controls(stem):
        if fusion:
            return {m: load_control(control_path(m, stem), DATA / "instance_seg" / f"{stem}_instance_seg.png") for m in fcfg.modalities}
        if args.control == "depth":  # 16-bit depth PNG saturates under convert("RGB")
            from hint_fusion.method import load_control as load_depth
            return load_depth(control_path("depth", stem), DATA / "instance_seg" / f"{stem}_instance_seg.png")
        return Image.open(control_path(args.control, stem)).convert("RGB")

    def run(model, stem, ctrl, prompt, size, **kw):
        if fusion:
            kw["seg"] = Image.open(DATA / "instance_seg" / f"{stem}_instance_seg.png")
        if init_sim and "scale" in kw:  # not the warmup call
            kw["sim"] = Image.open(DATA / "images" / f"{stem}_sim.png").convert("RGB")
            kw["strength"] = args.init_strength
        return model.translate(ctrl, prompt, size, **kw)

    sub = TRACK_A / "submissions" / args.name
    for d in ("simulated", "generated", "reference", "labels"):
        (sub / d).mkdir(parents=True, exist_ok=True)
    for p in sorted((DATA / "real").iterdir()):
        shutil.copy(p, sub / "reference" / p.name)

    stems = [p.name[: -len("_sim.png")] for p in sorted((DATA / "images").glob("*_sim.png"))][: args.n]
    if init_sim:
        from init_sim import ZImageInitSim
        model = ZImageInitSim(fcfg, device=args.device, memory_mode=args.memory_mode)
    elif fusion:
        model = ZImageControlHintFusion(fcfg, device=args.device, memory_mode=args.memory_mode)
    else:
        model = ZImageControlBaseline(device=args.device, memory_mode=args.memory_mode)
    run(model, stems[0], controls(stems[0]), GENERIC, (960, 540), steps=2)  # warmup

    total = 0.0
    for i, stem in enumerate(stems, 1):
        label = json.loads((DATA / "labels" / f"{stem}.json").read_text())
        sim = Image.open(DATA / "images" / f"{stem}_sim.png").convert("RGB")
        ctrl = controls(stem)
        prompt = label_prompt(label) if args.prompt == "label" else GENERIC
        t = time.perf_counter()
        out = run(model, stem, ctrl, prompt, sim.size, scale=args.scale, seed=args.seed + i)
        total += time.perf_counter() - t
        out.save(sub / "generated" / f"{stem}.png")
        shutil.copy(DATA / "images" / f"{stem}_sim.png", sub / "simulated" / f"{stem}_sim.png")
        shutil.copy(DATA / "labels" / f"{stem}.json", sub / "labels" / f"{stem}_label.json")
        print(f"{i}/{len(stems)} {stem} {total / i:.2f}s/img", flush=True)

    (sub / "generation_times.json").write_text(json.dumps(
        {"total_seconds": total, "scenes": len(stems), "hardware": "NVIDIA L40S"}, indent=2))
    (sub / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.eval:
        subprocess.run([
            sys.executable, "evaluator_cli.py", "evaluate", "--submission-path", str(sub),
            "--output", str(TRACK_A / "track_a_evaluation_output" / args.name / "report.json"),
            "--failure-model", args.failure_model, "--device", "cuda",
        ], cwd=TRACK_A, check=True)


if __name__ == "__main__":
    main()
