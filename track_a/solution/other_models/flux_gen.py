"""FLUX.1-schnell img2img from the sim image (model cpu offload, bf16)."""
import torch
from diffusers import FluxImg2ImgPipeline

import m_common as C

RES = (960, 544)  # multiples of 16


class FluxGen:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.pipe = FluxImg2ImgPipeline.from_pretrained(C.snap("black-forest-labs/FLUX.1-schnell"), torch_dtype=torch.bfloat16)
        if cfg.get("resident"):
            self.pipe.to(device)  # ~34 GB (24 transformer + 9.5 T5), no per-call offload traffic
        else:
            self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device

    @torch.no_grad()
    def __call__(self, ctx):
        c = self.cfg
        sim, prompt = ctx.sim, C.label_prompt(ctx.label)
        if c.get("seat_perturb"):
            col = C.seat_color_for(ctx.stem)
            sim = C.recolor_seats(sim, ctx.stem, col)
            prompt += f" The car seats are {col}."
        return self.pipe(prompt=prompt, image=C.fit(sim, RES), strength=c["strength"],
                         num_inference_steps=c["steps"], guidance_scale=0.0, max_sequence_length=256,
                         height=RES[1], width=RES[0], generator=torch.Generator("cpu").manual_seed(ctx.seed)).images[0]
