"""Per-feature agreement (generated vs real, Moondream/Qwen feature_matches) for two runs with paired per-scene bootstrap CIs.
Usage: per_feature.py RUN_A RUN_B [--B 10000] [--seed 0]   (diff = B - A; reads track_a_evaluation_output/<run>/report.json)"""
import argparse, json
from pathlib import Path
import numpy as np

EV = Path(__file__).resolve().parents[2] / "track_a_evaluation_output"


def load(run):
    imgs = json.loads((EV / run / "report.json").read_text())["final"]["failure"]["images"]
    return {i["image"]: i["feature_matches"] for i in imgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--B", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    x = ap.parse_args()
    A, B = load(x.a), load(x.b)
    stems = sorted(set(A) & set(B))
    feats = sorted(set.intersection(*[set(A[s]) & set(B[s]) for s in stems]))
    ma = np.array([[A[s][f] for f in feats] for s in stems], float)
    mb = np.array([[B[s][f] for f in feats] for s in stems], float)
    # add "all features match" (= 1 - failure) and mean over features
    ma = np.c_[ma, ma.min(1), ma.mean(1)]; mb = np.c_[mb, mb.min(1), mb.mean(1)]
    names = feats + ["ALL_MATCH(1-failure)", "MEAN_FEATURE"]
    rng = np.random.default_rng(x.seed)
    idx = rng.integers(0, len(stems), (x.B, len(stems)))
    d = mb - ma
    boots = d[idx].mean(1)  # B x F
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"A={x.a}  B={x.b}  scenes={len(stems)}  bootstrap={x.B}  (diff = B - A, agreement; positive = B agrees more with the real photo)")
    print("| feature | A agree | B agree | diff | 95% CI | sig |")
    print("|---|---|---|---|---|---|")
    for j, n in enumerate(names):
        sig = "*" if lo[j] > 0 or hi[j] < 0 else ""
        print(f"| {n} | {ma[:, j].mean():.3f} | {mb[:, j].mean():.3f} | {d[:, j].mean():+.3f} | [{lo[j]:+.3f}, {hi[j]:+.3f}] | {sig} |")


main()
