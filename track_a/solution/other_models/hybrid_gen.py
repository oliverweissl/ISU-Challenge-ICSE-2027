"""M6 hybrid: FLUX.1-schnell img2img (s0.6) renders the whole cabin; Realistic Vision (SD1.5 + canny ControlNet, img2img on a CROP)
re-renders each PERSON only (instance-seg person + seatbelt + driver-phone masks), composited back with a feathered mask.
Presence/quantity is preserved by construction: the person mask comes from the sim segmentation, and RV only refines pixels inside it."""
import numpy as np
import torch
from PIL import Image, ImageFilter

import m_common as C
import flux_gen
import sd15_gen

PERSONS = {1: ("driver", "driver"), 2: ("passenger_codriver", "front passenger"), 3: ("passenger_rear_left", "rear passenger"),
           4: ("passenger_rear_right", "rear passenger")}
COLORS = {1: [5, 15], 2: [6], 3: [7], 4: [8]}  # belt classes
CMAP = {d["id"]: d["color_rgb"] for d in __import__("json").loads((C.DATA / "instance_seg/instance_seg_class_map.json").read_text())}


def cls_mask(seg, i):
    return np.abs(seg - np.array(CMAP[i])).sum(-1) == 0


class HybridGen:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.flux = flux_gen.FluxGen(cfg, device)
        rv_cfg = dict(strength=cfg["rv_strength"], steps=30, cfg=6.0, cn=0.8)
        self.rv = sd15_gen.SD15Gen(rv_cfg, device)
        self.rvc = rv_cfg
        self.device = device

    @torch.no_grad()
    def __call__(self, ctx):
        base = self.flux(ctx).convert("RGB").resize(ctx.sim.size, Image.LANCZOS)
        seg = np.asarray(Image.open(C.DATA / "instance_seg" / f"{ctx.stem}_instance_seg.png").convert("RGB"), dtype=np.int32)
        canny = C.load_canny(ctx.stem)
        out = base.copy()
        for pid, (role, _) in PERSONS.items():
            body = cls_mask(seg, pid)
            if body.sum() < 400:
                continue
            m = body.copy()
            for b in COLORS[pid]:
                m |= cls_mask(seg, b)
            ys, xs = np.where(m)
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            pad = int(0.2 * max(x1 - x0, y1 - y0)) + 8
            box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, base.width), min(y1 + pad, base.height))
            w, h = box[2] - box[0], box[3] - box[1]
            s = 640 / max(w, h)
            cw, ch = max(int(round(w * s / 8)) * 8, 64), max(int(round(h * s / 8)) * 8, 64)
            crop_img, crop_can = out.crop(box).resize((cw, ch), Image.BICUBIC), canny.crop(box).resize((cw, ch), Image.NEAREST)
            gender = (ctx.label.get({1: "driver", 2: "passenger_codriver", 3: "passenger_rear_left", 4: "passenger_rear_right"}[pid] + "_gender") or "person").lower()
            col = (ctx.label.get({1: "driver", 2: "passenger_codriver", 3: "passenger_rear_left", 4: "passenger_rear_right"}[pid] + "_tshirt_color") or "").lower()
            prompt = f"RAW photo, a {gender} person wearing a {col} t-shirt sitting in a car, seat belt, natural skin texture, detailed face, soft light"
            ref = self.rv.pipe(prompt=prompt, negative_prompt=C.NEG, image=crop_img, control_image=crop_can,
                               strength=self.rvc["strength"], num_inference_steps=30, guidance_scale=6.0,
                               controlnet_conditioning_scale=0.8, generator=torch.Generator(self.device).manual_seed(ctx.seed + pid)).images[0]
            ref = ref.resize((w, h), Image.LANCZOS)
            mm = Image.fromarray((m[box[1]:box[3], box[0]:box[2]] * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(4))
            out.paste(ref, (box[0], box[1]), mm)
        return out
