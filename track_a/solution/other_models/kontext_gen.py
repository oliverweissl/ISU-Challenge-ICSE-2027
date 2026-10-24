"""FLUX.1 Kontext [dev] instruction editing of the sim image. Transformer from Kontext, VAE/T5/CLIP from FLUX.1-schnell
(identical components), bf16, model cpu offload."""
import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel

import m_common as C

RES = (960, 544)


class KontextGen:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        tr = FluxTransformer2DModel.from_pretrained(C.snap("black-forest-labs/FLUX.1-Kontext-dev") + "/transformer", torch_dtype=torch.bfloat16)
        self.pipe = FluxKontextPipeline.from_pretrained(C.snap("black-forest-labs/FLUX.1-schnell"), transformer=tr, torch_dtype=torch.bfloat16)
        if cfg.get("resident"):
            self.pipe.to(device)  # ~34 GB (24 transformer + 9.5 T5), no per-call offload traffic
        else:
            self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device

    @torch.no_grad()
    def __call__(self, ctx):
        c = self.cfg
        instr = c["instr"]
        if c.get("seat_perturb"):
            instr = c["instr"].format(color=C.seat_color_for(ctx.stem))
        if c.get("with_label"):
            instr += " " + C.label_prompt(ctx.label)
        return self.pipe(image=C.fit(ctx.sim, RES), prompt=instr, guidance_scale=c["cfg"], num_inference_steps=c["steps"],
                         height=RES[1], width=RES[0], max_area=RES[0] * RES[1], _auto_resize=False,
                         generator=torch.Generator("cpu").manual_seed(ctx.seed)).images[0]
