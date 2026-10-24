"""Append one finished run to research/runs.csv.  Usage: record_run.py <run_name> [--note TEXT] [--all]
A run is finished only if report.json contains 'efficiency'; otherwise it is skipped (exit code 2)."""
import argparse, csv, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "research" / "runs.csv"
HEADER = "name,method,control,scale,prompt,n,failure_rate,sam_semantic_score,fid_gen_real,kid_gen_real,vgg_gen_real,ssim_gen_real,psnr_gen_real,feature_hamming,clip_div,sec_per_scene,commit,notes".split(",")
PATHS = {
    "failure_rate": "final.failure.failure_rate",
    "sam_semantic_score": "sam.simulated_vs_generated.summary.semantic_segmentation_score",
    "fid_gen_real": "final.realism.generated_vs_real.distribution.fid",
    "kid_gen_real": "final.realism.generated_vs_real.distribution.kid_mean",
    "vgg_gen_real": "final.realism.generated_vs_real.summary.vgg_perceptual_distance",
    "ssim_gen_real": "final.realism.generated_vs_real.summary.ssim",
    "psnr_gen_real": "final.realism.generated_vs_real.summary.psnr",
    "feature_hamming": "final.feature_diversity.mean_pairwise_hamming",
    "clip_div": "final.clip_visual_diversity.mean_cosine_distance",
    "sec_per_scene": "efficiency.seconds_per_scene",
}


def dig(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return "n/a"
        d = d[k]
    return "n/a" if d is None or isinstance(d, (dict, list)) else d


def commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "n/a"


def record(name, note=""):
    rep = ROOT / "track_a" / "track_a_evaluation_output" / name / "report.json"
    cfgp = ROOT / "track_a" / "submissions" / name / "config.json"
    if not rep.exists():
        return f"skip {name}: no report.json"
    r = json.loads(rep.read_text())
    if "efficiency" not in r:
        return f"skip {name}: unfinished (no 'efficiency' in report)"
    cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
    if CSV.exists() and any(row.get("name") == name for row in csv.DictReader(CSV.open())):
        return f"skip {name}: already recorded"
    method = cfg.get("method", "baseline" if cfg else "n/a")
    control = cfg.get("control", "n/a")
    if method == "hint_fusion":
        fu = cfg.get("fusion") or {}
        control = f"fusion:{fu.get('mode', cfg.get('fusion_mode', 'n/a'))}"
    n = cfg.get("n", "n/a")
    notes = [note] if note else []
    if not cfg:
        notes.append("config.json missing")
    if isinstance(n, int) and n < 60:
        notes.append(f"smoke (n={n}, no full real refs)")
    if "dbg" == name:
        notes.append("dbg smoke run")
    row = {"name": name, "method": method, "control": control, "scale": cfg.get("scale", "n/a"),
           "prompt": cfg.get("prompt", "n/a"), "n": n, "commit": commit(), "notes": "; ".join(dict.fromkeys(notes))}
    for k, p in PATHS.items():
        row[k] = dig(r, p)
    CSV.parent.mkdir(exist_ok=True)
    new = not CSV.exists() or CSV.stat().st_size == 0
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, HEADER)
        if new:
            w.writeheader()
        w.writerow(row)
    return f"recorded {name}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--note", default="")
    ap.add_argument("--all", action="store_true", help="try every dir in track_a_evaluation_output")
    a = ap.parse_args()
    names = a.runs
    if a.all:
        names = sorted(p.name for p in (ROOT / "track_a" / "track_a_evaluation_output").iterdir() if p.is_dir())
    if not names:
        ap.error("give run names or --all")
    msgs = [record(n, a.note) for n in names]
    print("\n".join(msgs))
    sys.exit(0 if all(m.startswith(("recorded", "skip " + "")) for m in msgs) else 1)
