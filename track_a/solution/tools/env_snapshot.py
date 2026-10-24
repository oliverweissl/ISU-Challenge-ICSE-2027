"""Write research/env.md.  Usage: env_snapshot.py [--python PATH]  (default: zimage env python)"""
import argparse, datetime, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument("--python", default=os.path.expanduser("~/miniconda3/envs/zimage/bin/python"))
a = ap.parse_args()
KEY = ("torch", "torchvision", "diffusers", "transformers", "accelerate", "safetensors", "xformers", "numpy", "pillow",
       "opencv", "scipy", "scikit", "open-clip", "open_clip", "clip", "segment", "sam", "timm", "lpips", "torchmetrics",
       "peft", "videox", "qwen", "flash")


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=isinstance(cmd, str), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        return f"n/a ({type(e).__name__})"


freeze = sh([a.python, "-m", "pip", "freeze"])
pk = [l for l in freeze.splitlines() if l.lower().startswith(KEY)]
env_name = Path(a.python).parents[1].name
gpu = sh("nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader")
hf = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
snaps = []
if hf.is_dir():
    for m in sorted(hf.glob("models--*")):
        for s in sorted((m / "snapshots").glob("*")) if (m / "snapshots").is_dir() else []:
            snaps.append(f"{m.name[8:].replace('--', '/')}  {s.name}")
vfx = ROOT / "VideoX-Fun"
out = ["# Environment snapshot", f"_{datetime.datetime.now():%Y-%m-%d %H:%M}_", "",
       f"- conda env: `{env_name}` (`{a.python}`)",
       f"- python: {sh([a.python, '--version'])}",
       f"- repo commit: {sh(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])}",
       f"- VideoX-Fun commit: {sh(['git', '-C', str(vfx), 'rev-parse', 'HEAD']) if vfx.exists() else 'n/a (not in repo root)'}",
       "", "## GPUs", "```", gpu, "```", "", "## Key packages", "```", *pk, "```", "",
       "## HF model snapshots", "```", *(snaps or ["n/a"]), "```", ""]
(ROOT / "research").mkdir(exist_ok=True)
(ROOT / "research" / "env.md").write_text("\n".join(out))
print("wrote research/env.md;", len(pk), "packages,", len(snaps), "snapshots")
