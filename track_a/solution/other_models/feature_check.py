"""Controllability check for the seat-colour perturbation: for each scene the target colour is m_common.seat_color_for(stem).
Measures (a) Qwen2.5-VL answer to 'what colour are the car seats' on a seat-mask crop == target (match rate; chance ~1/7) and
(b) mean seat-mask RGB of the output.  Run on the perturbed run and on an unperturbed control run (e.g. m2_flux_i2i_s0.6_60).
usage (gpu_run): feature_check.py <run> [n=60]"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import m_common as C
from presence_check import VLM, crop, ROOT

COLS = "black, red, blue, brown, green, beige, gray"
if __name__ == "__main__":
    run, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
    stems = C.__dict__["all_stems"]()[:n] if "all_stems" in C.__dict__ else __import__("exp_common").all_stems()[:n]
    vlm, hit, rows = VLM(), 0, {}
    for s in stems:
        img = Image.open(ROOT / "track_a/submissions" / run / "generated" / f"{s}.png").convert("RGB").resize((960, 540))
        m = C.seat_mask(s)
        tgt = C.seat_color_for(s)
        a = vlm.ask(crop(img, m), f"What color are the car seats in this image? Answer with one word from: {COLS}.")
        rgb = np.asarray(img, dtype=np.float32)[m].mean(0).round(1).tolist()
        rows[s] = {"target": tgt, "answer": a, "seat_rgb": rgb, "hit": tgt in a}
        hit += tgt in a
    rep = {"run": run, "n": len(stems), "match_rate": hit / len(stems)}
    d = Path(__file__).resolve().parents[1] / "out" / "presence"
    (d / f"feature_{run}.json").write_text(json.dumps({"summary": rep, "scenes": rows}, indent=1))
    print(json.dumps(rep))
