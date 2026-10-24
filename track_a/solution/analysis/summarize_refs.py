"""Summarise evaluator reports of ref_* (and any other) runs: headline table + per-feature failure table.
    python summarize_refs.py [report_dir_glob]"""
import json, sys, glob
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[3] / "track_a" / "track_a_evaluation_output"
rows = {}; feat = {}
for f in sorted(ROOT.glob("*/report.json")):
    r = json.loads(f.read_text())
    if "efficiency" not in r: continue
    n = f.parent.name; fin = r["final"]; g = fin["realism"]["generated_vs_real"]
    d = g["distribution"]; s = g["summary"]
    rows[n] = dict(sam=r["sam"]["simulated_vs_generated"]["summary"]["semantic_segmentation_score"],
        fail=fin["failure"]["failure_rate"], nfail=fin["failure"]["failures"],
        fid=d["fid"], kid=d["kid_mean"], IS=d["inception_score_mean"], ssim=s["ssim"], psnr=s["psnr"],
        lpips_vgg=s["vgg_perceptual_distance"], lbp=s["lbp_histogram_similarity"], glcm=s["glcm_texture_distance"],
        wass=s["wasserstein_distance"], hist=s["histogram_intersection"], corr=s["correlation_coefficient"],
        fractal=s["fractal_dimension_similarity"], nmi=s["normalized_mutual_information"], kl=s["kl_divergence"],
        hamming=fin["feature_diversity"]["mean_pairwise_hamming"], clipdiv=fin["clip_visual_diversity"]["mean_cosine_distance"])
    c = defaultdict(int)
    for im in fin["failure"]["images"]:
        for k, v in im["feature_matches"].items(): c[k] += (not v)
    feat[n] = dict(c)
if __name__ == "__main__":
    keys = list(next(iter(rows.values())).keys())
    print("run," + ",".join(keys))
    for n, r in rows.items(): print(n + "," + ",".join(f"{r[k]:.4g}" for k in keys))
    allk = sorted({k for v in feat.values() for k in v})
    print("\nper-feature mismatch counts (of 60)")
    print("feature," + ",".join(feat))
    for k in allk: print(k + "," + ",".join(str(feat[n].get(k, 0)) for n in feat))
