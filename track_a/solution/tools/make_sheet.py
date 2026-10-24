"""Blind-shuffled contact sheet.  Usage: make_sheet.py <run>... [--n 6] [--seed 0] [--name SHEET]
Columns: sim | real ref | candidates (shuffled). Output: ../out/sheets/<name>.png + <name>.key.json (key kept out of image).
Real ref = the stem-paired real photo data/real/<stem>_real.jpg (verified paired by the critic)."""
import argparse, json, random
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "out" / "sheets"
W = 480

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--name", default=None)
a = ap.parse_args()

subs = {r: ROOT / "track_a" / "submissions" / r for r in a.runs}
for r, p in subs.items():
    if not (p / "generated").is_dir():
        raise SystemExit(f"missing generated/ for {r}")
stems = sorted(set.intersection(*[{f.stem for f in (p / "generated").iterdir()} for p in subs.values()]))
rng = random.Random(a.seed)
stems = sorted(rng.sample(stems, min(a.n, len(stems))))
order = list(a.runs)
rng.shuffle(order)


def fit(im):
    return im.convert("RGB").resize((W, int(im.height * W / im.width)))


rows = []
for i, s in enumerate(stems):
    sim = Image.open(ROOT / "data" / "images" / f"{s}_sim.png")
    cells = [fit(sim), fit(Image.open(ROOT / "data" / "real" / f"{s}_real.jpg"))]
    for r in order:
        cells.append(fit(Image.open(next((subs[r] / "generated").glob(s + ".*")))))
    rows.append(cells)
H = max(c.height for row in rows for c in row)
hdr = 24
sheet = Image.new("RGB", (W * (2 + len(order)), hdr + H * len(rows)), "white")
d = ImageDraw.Draw(sheet)
for j, t in enumerate(["sim", "real ref"] + [f"cand {chr(65 + k)}" for k in range(len(order))]):
    d.text((j * W + 6, 6), t, fill="black")
for i, row in enumerate(rows):
    for j, c in enumerate(row):
        sheet.paste(c, (j * W, hdr + i * H))
OUT.mkdir(parents=True, exist_ok=True)
name = a.name or "_".join(sorted(a.runs))[:80]
sheet.save(OUT / f"{name}.png")
(OUT / f"{name}.key.json").write_text(json.dumps(
    {"columns": {f"cand {chr(65 + k)}": r for k, r in enumerate(order)}, "scenes": stems, "seed": a.seed}, indent=2))
print("wrote", OUT / f"{name}.png", "and key.json")
