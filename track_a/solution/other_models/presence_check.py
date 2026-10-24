"""Presence-preservation check (task goal: perturb FEATURES, never quantity/presence of people and items).

For each scene the instance segmentation (data/instance_seg, class ids from instance_seg_class_map.json) tells which items exist.
Qwen2.5-VL-3B (same family as the evaluator's failure model, greedy) answers yes/no / count questions on the image under test:
  * present objects (phone_codriver, cola bottle, cola can, suitcase, baby seat, baby, driver phone): crop to the padded bbox of the
    item's mask and ask "Is there a <item> in this image?"   -> violation if "no"  (checks the item is still there at its place)
  * absent object types (phone, cola bottle, cola can, suitcase, baby seat): whole image "Is there a <item> ...?" -> violation if "yes"
  * people: whole image "How many people are visible?" vs number of person masks (driver + passengers) -> violation if different
presence_violation_rate = violations / checks over all scenes; scene_any = share of scenes with >=1 violation. The checker has its own error:
run the same check on source=sim (images are correct by construction) and source=real (paired real photo, analysis only) for the noise floor.
usage (under tools/gpu_run.sh):  presence_check.py <run|sim|real> [n=60]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data"
CMAP = {d["id"]: d for d in json.loads((DATA / "instance_seg/instance_seg_class_map.json").read_text())}
OBJ = {9: "phone", 10: "cola bottle", 11: "cola can", 12: "suitcase", 13: "baby seat", 14: "baby", 15: "phone held to the ear"}
ABSENT_Q = {9: "phone", 10: "cola bottle", 11: "cola can", 12: "suitcase", 13: "baby seat"}
PERSONS = (1, 2, 3, 4)
MIN_PIX = 120


def masks(stem):
    a = np.asarray(Image.open(DATA / "instance_seg" / f"{stem}_instance_seg.png").convert("RGB"), dtype=np.int32)
    out = {}
    for i, d in CMAP.items():
        m = (np.abs(a - np.array(d["color_rgb"])).sum(-1) == 0)
        out[i] = m
    return out


class VLM:
    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        n = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.p = AutoProcessor.from_pretrained(n)
        self.m = Qwen2_5_VLForConditionalGeneration.from_pretrained(n, torch_dtype="auto", device_map=device).eval()
        self.dev = device

    @torch.inference_mode()
    def ask(self, img, q):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
        t = self.p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = self.p(text=[t], images=[img], padding=True, return_tensors="pt")
        x = {k: v.to(self.dev) if hasattr(v, "to") else v for k, v in x.items()}
        g = self.m.generate(**x, max_new_tokens=4, do_sample=False)
        return self.p.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()


def crop(img, m):
    ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    pad = max(int(0.5 * max(w, h)), 60)
    box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, img.width), min(y1 + pad, img.height))
    c = img.crop(box)
    s = 336 / max(c.size)
    return c.resize((max(int(c.width * s), 56), max(int(c.height * s), 56)), Image.BICUBIC)


def check_scene(vlm, img, stem):
    M = masks(stem)
    res = []  # (kind, item, expected, answer, violation)
    for i, name in OBJ.items():
        if M[i].sum() >= MIN_PIX:
            a = vlm.ask(crop(img, M[i]), f"Is there a {name} in this image? Answer yes or no.")
            res.append(("present", name, "yes", a, not a.startswith("yes")))
    for i, name in ABSENT_Q.items():
        if i == 14 or any(M[j].sum() >= MIN_PIX for j in ((9,) if i == 9 else (i,))):
            continue
        a = vlm.ask(img, f"Is there a {name} visible in this image? Answer yes or no.")
        res.append(("absent", name, "no", a, a.startswith("yes")))
    exp = sum(M[i].sum() >= MIN_PIX for i in PERSONS)
    a = vlm.ask(img, "How many people are visible in this image? Answer with a single digit.")
    digit = next((c for c in a if c.isdigit()), "?")
    res.append(("people", "count", str(exp), a, digit != str(exp)))
    return res


if __name__ == "__main__":
    src = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    stems = [p.name[:-len("_sim.png")] for p in sorted((DATA / "images").glob("*_sim.png"))][:n]
    vlm = VLM()
    out, tot, viol, anyv = {}, 0, 0, 0
    by = {}
    for s in stems:
        if src == "sim":
            img = Image.open(DATA / "images" / f"{s}_sim.png")
        elif src == "real":
            img = Image.open(DATA / "real" / f"{s}_real.jpg").resize((960, 540))
        else:
            img = Image.open(ROOT / "track_a/submissions" / src / "generated" / f"{s}.png")
        img = img.convert("RGB").resize((960, 540))
        r = check_scene(vlm, img, s)
        out[s] = r
        tot += len(r)
        viol += sum(x[4] for x in r)
        anyv += any(x[4] for x in r)
        for k, it, _, _, v in r:
            b = by.setdefault(f"{k}:{it}", [0, 0])
            b[0] += 1
            b[1] += int(v)
    rep = {"source": src, "n": len(stems), "checks": tot, "violations": viol, "presence_violation_rate": viol / tot,
           "scene_any_violation": anyv / len(stems), "by_item": {k: {"checks": a, "violations": b} for k, (a, b) in by.items()}}
    d = HERE.parent / "out" / "presence"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{src}.json").write_text(json.dumps({"summary": rep, "scenes": out}, indent=1))
    print(json.dumps(rep, indent=1))
