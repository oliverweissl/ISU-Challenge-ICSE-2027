"""Family M runner:  run_m.py <variant> <n> [evaluate 0/1]   (always under tools/gpu_run.sh)
Variants are defined in VARIANTS (name -> model kind + parameters). Run name = <variant>_<n>."""
import json
import sys
import time

import torch
from PIL import Image

import m_common as C

# ---- variants ---------------------------------------------------------------------------------------------
# sdxl: mode t2i|i2i, controls list of (name, scale); strength for i2i. res (W,H) multiples of 8.
VARIANTS = {
    # M1 SDXL + ControlNet
    "m1_sdxl_canny":        dict(kind="sdxl", mode="t2i", controls=[("canny", 0.8)], steps=30, cfg=6.0),
    "m1_sdxl_canny_depth":  dict(kind="sdxl", mode="t2i", controls=[("canny", 0.6), ("depth", 0.4)], steps=30, cfg=6.0),
    "m1_sdxl_i2i_s0.6":     dict(kind="sdxl", mode="i2i", controls=[("canny", 0.5)], strength=0.6, steps=30, cfg=6.0),
    "m1_sdxl_i2i_s0.75":    dict(kind="sdxl", mode="i2i", controls=[("canny", 0.5)], strength=0.75, steps=30, cfg=6.0),
    # M2 FLUX schnell img2img
    "m2_flux_i2i_s0.6":     dict(kind="flux", strength=0.6, steps=4, resident=True),
    "m2_flux_i2i_s0.8":     dict(kind="flux", strength=0.8, steps=4, resident=True),
    "m2_flux_i2i_s0.7":     dict(kind="flux", strength=0.7, steps=4, resident=True),
    "m2_flux_i2i_s0.5":     dict(kind="flux", strength=0.5, steps=4, resident=True),
    "m7_flux_seat_s0.6":    dict(kind="flux", strength=0.6, steps=4, resident=True, seat_perturb=True),
    "m7_flux_seat_s0.7":    dict(kind="flux", strength=0.7, steps=4, resident=True, seat_perturb=True),
    "m7_kontext_seat":      dict(kind="kontext", steps=20, cfg=2.5, resident=True, seat_perturb=True,
                                 instr="Change the upholstery of all car seats to {color}. Make this a real photograph. Keep all people and objects exactly the same, same number of people and items."),
    "m3_kontext_keep":      dict(kind="kontext", steps=20, cfg=2.5, resident=True,
                                 instr="Turn this 3D render into a real photograph. Keep all people and objects exactly as they are (same number, same positions, same colours); only change textures, skin, lighting and camera look."),
    "m6_hybrid_flux_rv":    dict(kind="hybrid", strength=0.6, steps=4, resident=True, rv_strength=0.45),
    # M3 instruction edit
    "m3_ip2p":              dict(kind="ip2p", steps=20, cfg=7.5, icfg=1.5,
                                 instr="make this a real photograph taken inside a car"),
    "m3_kontext":           dict(kind="kontext", steps=20, cfg=2.5, resident=True,
                                 instr="Turn this 3D render into a real photograph taken by an in-car camera. Keep the exact same scene, people, objects and layout."),
    "m3_qwen_edit":         dict(kind="qwen", steps=4, cfg=1.0, lora=True,
                                 instr="Turn this 3D render into a real photograph taken by an in-car camera. Keep the exact same scene, people, objects and layout."),
    # M4 SD1.5 realistic checkpoint
    "m4_rv_i2i_s0.6":       dict(kind="sd15", strength=0.6, steps=30, cfg=6.0, cn=0.6),
    "m4_rv_i2i_s0.75_cn1.0": dict(kind="sd15", strength=0.75, steps=30, cfg=6.0, cn=1.0),
    "m3_ip2p_i1.0":         dict(kind="ip2p", steps=20, cfg=7.5, icfg=1.0,
                                 instr="make this a real photograph taken inside a car by a dim infrared camera"),
    "m4_rv_i2i_s0.45":      dict(kind="sd15", strength=0.45, steps=30, cfg=6.0, cn=0.6),
    # M5 cascade SDXL-CN -> Z-Image-Turbo low strength refine
    "m5_sdxl_zimage":       dict(kind="cascade", sdxl="m1_sdxl_canny", strength=0.5),
}


def build(cfg):
    k = cfg["kind"]
    if k == "sdxl":
        import sdxl_gen
        return sdxl_gen.SDXLGen(cfg)
    if k == "flux":
        import flux_gen
        return flux_gen.FluxGen(cfg)
    if k == "hybrid":
        import hybrid_gen
        return hybrid_gen.HybridGen(cfg)
    if k == "kontext":
        import kontext_gen
        return kontext_gen.KontextGen(cfg)
    if k == "qwen":
        import qwen_gen
        return qwen_gen.QwenGen(cfg)
    if k == "ip2p":
        import ip2p_gen
        return ip2p_gen.IP2PGen(cfg)
    if k == "sd15":
        import sd15_gen
        return sd15_gen.SD15Gen(cfg)
    if k == "cascade":
        import cascade_gen
        return cascade_gen.CascadeGen(cfg, VARIANTS[cfg["sdxl"]])
    raise ValueError(k)


if __name__ == "__main__":
    v, n = sys.argv[1], int(sys.argv[2])
    ev = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    cfg = VARIANTS[v]
    gen = build(cfg)
    name = f"{v}_{n}" if n == 60 else f"{v}_smoke{n}"

    def gen_fn(ctx):
        return gen(ctx)

    C.run(name, gen_fn, {"method": v, **cfg}, n=n, evaluate=ev)
    print("peak mem GiB", torch.cuda.max_memory_allocated() / 2**30)
