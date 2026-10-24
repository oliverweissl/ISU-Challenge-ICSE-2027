"""C5: single-feature perturbation of the sim image through the instance-seg mask (classical, no network, no real data).
One feature per image, everything else pixel-identical; presence/quantity of items and people cannot change (pixel mask edit only).
  seat_recolor : class 16 'seat' recoloured (hue = per-scene seeded, saturation 0.45, luminance kept)
  interior_dark: class 17 'car_interior' darkened x0.6 (lighting/exposure change of one component)
  babyseat_hue : class 13 'baby_seat' hue-rotated by a seeded angle in [60,300] deg (colour change of one object; no-op when absent)
Masks: exact class colours from data/instance_seg/instance_seg_class_map.json, feathered 2 px.
usage: c5.py <variant> <n> <eval> <name>"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from skimage import color

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from exp_common import run_submission, DATA  # noqa
CLS = {"seat": (38, 15, 3), "car_interior": (28, 15, 150), "baby_seat": (200, 0, 90)}


def mask(stem, cls):
    a = np.asarray(Image.open(DATA / "instance_seg" / f"{stem}_instance_seg.png").convert("RGB")).astype(int)
    m = (np.abs(a - np.array(CLS[cls])).sum(-1) == 0).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(m).filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255, m.mean() / 255


def apply(c, variant):
    rng = np.random.default_rng(c.seed)
    img = np.asarray(c.sim.convert("RGB"), np.float32) / 255
    if variant == "seat_recolor":
        m, _ = mask(c.stem, "seat")
        hsv = color.rgb2hsv(img); hsv[..., 0] = rng.uniform(); hsv[..., 1] = 0.45; hsv[..., 2] = np.clip(hsv[..., 2] * 1.6, 0, 1)
        new = color.hsv2rgb(hsv)
    elif variant == "interior_dark":
        m, _ = mask(c.stem, "car_interior"); new = img * 0.6
    else:
        m, _ = mask(c.stem, "baby_seat")
        hsv = color.rgb2hsv(img); hsv[..., 0] = (hsv[..., 0] + rng.uniform(60, 300) / 360) % 1.0
        new = color.hsv2rgb(hsv)
    out = img * (1 - m) + new * m
    return Image.fromarray((np.clip(out, 0, 1) * 255 + .5).astype(np.uint8))


if __name__ == "__main__":
    v, n, ev, name = sys.argv[1], int(sys.argv[2]), bool(int(sys.argv[3])), sys.argv[4]
    run_submission(name, lambda c: apply(c, v), config=dict(family="C5", variant=v, real_used=False), n=n, evaluate=ev)
