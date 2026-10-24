"""Paired presence analysis vs the sim (checker noise cancels): loss = check passed on sim image but fails on the generated image.
Reports over ALL checks and over RELIABLE checks (item types where the checker's own error on sim and real is <= 10%)."""
import json, sys
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "out" / "presence"
L = lambda n: json.loads((D / f"{n}.json").read_text())["scenes"]
sim, real = L("sim"), L("real")
def err(sc):
    e = {}
    for r in sc.values():
        for k, it, _, _, v in r:
            a = e.setdefault(f"{k}:{it}", [0, 0]); a[0] += 1; a[1] += v
    return {k: b / a for k, (a, b) in e.items()}
es, er = err(sim), err(real)
rel = {k for k in es if es[k] <= 0.10 and er.get(k, 0) <= 0.10}
print("reliable item checks:", sorted(rel))
print("run | all-checks loss | reliable loss (n) | reliable raw viol")
for run in sys.argv[1:]:
    g = L(run)
    la = na = lr = nr = vr = tr = 0
    for s, r in g.items():
        for (k, it, _, _, vg), (_, _, _, _, vs) in zip(r, sim[s]):
            key = f"{k}:{it}"
            if not vs:
                na += 1; la += vg
                if key in rel: nr += 1; lr += vg
            if key in rel: tr += 1; vr += vg
    print(f"{run} | {la/na:.3f} ({la}/{na}) | {lr/max(nr,1):.3f} ({lr}/{nr}) | {vr/max(tr,1):.3f} ({vr}/{tr})")
