"""Generate a full Track A submission via run_experiment.py and validate it against track_a/README.md rules.
Usage: make_submission.py --name TEAM [--n 1060] [--out DIR] [-- <run_experiment.py method args>]
  e.g. make_submission.py --name final --n 1060 -- --method hint_fusion --fusion-mode conflict_aware --scale 0.8
  make_submission.py --validate-only --out ../../submissions/final
Without --out the folder is track_a/submissions/<name>. With --out, the folder is moved there after generation.
Never overwrites an existing folder. Must be run with the zimage python and a free GPU (set CUDA_VISIBLE_DEVICES)."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data"
IMG = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate(sub: Path, expect_n=None):
    errs = []
    for d in ("simulated", "generated", "reference", "labels"):
        if not (sub / d).is_dir():
            errs.append(f"missing dir {d}/")
    if errs:
        return errs
    sims = {p.name[: -len("_sim.png")]: p for p in (sub / "simulated").iterdir() if p.name.endswith("_sim.png")}
    gens = {}
    for p in (sub / "generated").iterdir():
        if p.suffix.lower() not in IMG:
            errs.append(f"generated/{p.name}: bad format")
        elif p.stem in gens:
            errs.append(f"generated/{p.name}: duplicate stem")
        gens[p.stem] = p
    labs = {p.name[: -len("_label.json")] for p in (sub / "labels").glob("*_label.json")}
    allsims = {p.name[: -len("_sim.png")] for p in (DATA / "images").glob("*_sim.png")}
    want = len(allsims) if expect_n is None else expect_n
    if len(sims) != want:
        errs.append(f"simulated count {len(sims)} != {want}")
    if set(sims) != set(gens):
        errs.append(f"stem mismatch sim/generated: missing gen {sorted(set(sims) - set(gens))[:5]}, extra {sorted(set(gens) - set(sims))[:5]}")
    if labs != set(sims):
        errs.append("labels do not match simulated stems")
    if not set(sims) <= allsims:
        errs.append("simulated stems not in data/images")
    for s, p in sims.items():  # simulated must be unmodified, generated same size, RGB
        if s in gens:
            with Image.open(gens[s]) as g, Image.open(p) as im:
                if g.size != im.size:
                    errs.append(f"{s}: generated size {g.size} != sim {im.size}")
                if g.mode not in ("RGB",):
                    errs.append(f"{s}: generated mode {g.mode}")
        if (DATA / "images" / p.name).stat().st_size != p.stat().st_size:
            errs.append(f"{p.name}: simulated modified")
    real_src = {p.name for p in (DATA / "real").iterdir()}
    if {p.name for p in (sub / "reference").iterdir()} != real_src:
        errs.append("reference/ differs from data/real")
    t = sub / "generation_times.json"
    if not t.exists():
        errs.append("generation_times.json missing")
    else:
        j = json.loads(t.read_text())
        if not isinstance(j.get("total_seconds"), (int, float)) or "hardware" not in j:
            errs.append("generation_times.json needs total_seconds and hardware")
    return errs


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--n", type=int, default=1060)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args(argv)
    if a.validate_only:
        if not a.out:
            ap.error("--validate-only needs --out")
        errs = validate(Path(a.out), expect_n=a.n)
        print("VALID" if not errs else "INVALID:\n  " + "\n  ".join(errs))
        sys.exit(bool(errs))
    if not a.name:
        ap.error("--name required")
    default = ROOT / "track_a" / "submissions" / a.name
    final = Path(a.out).resolve() if a.out else default
    if default.exists() or final.exists():
        sys.exit(f"refusing to overwrite existing {default if default.exists() else final}")
    subprocess.run([sys.executable, "-W", "ignore", str(HERE.parent / "run_experiment.py"), "--name", a.name,
                    "--n", str(a.n), *extra], cwd=HERE.parent, check=True)
    if final != default:
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(default), str(final))
    errs = validate(final, expect_n=a.n)
    print("VALID" if not errs else "INVALID:\n  " + "\n  ".join(errs[:30]))
    sys.exit(bool(errs))


if __name__ == "__main__":
    main()
