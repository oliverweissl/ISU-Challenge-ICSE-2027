"""Spearman rank correlation between fast_proxy.py outputs (proxy_out/*.json) and full-evaluator reports."""
import json
from pathlib import Path
from scipy.stats import spearmanr
from summarize_refs import rows
H = Path(__file__).resolve().parent
P = {p.stem: json.loads(p.read_text()) for p in (H / "proxy_out").glob("*.json")}
names = [n for n in P if n in rows]
pairs = [("sam_proxy", "sam"), ("kid_r50", "fid"), ("kid_r50", "kid"), ("fd_diag", "fid"), ("fail_proxy", "fail"),
         ("col_emd", "wass"), ("hf_ratio", "glcm")]
def corr(ns, tag):
    print(f"--- {tag} (n={len(ns)})")
    for a, b in pairs:
        r = spearmanr([P[n][a] for n in ns], [rows[n][b] for n in ns])
        print(f"  {a:10s} vs {b:5s} rho={r.statistic:+.2f} p={r.pvalue:.3g}")
if __name__ == "__main__":
    corr(names, "all runs")
    gen = [n for n in names if not n.startswith("ref_")]
    corr(gen, "generator runs only")
    print("secs mean", sum(P[n]["seconds"] for n in names) / len(names))
