"""SD1.5 Realistic Vision V5.1 + control_v11p canny, img2img from the sim image (SDEdit with a photoreal checkpoint)."""
import torch
from diffusers import (AutoencoderKL, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler)

import m_common as C

RES = (768, 432)


class SD15Gen:
    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        dt = torch.float16
        cn = ControlNetModel.from_pretrained(C.snap("lllyasviel/control_v11p_sd15_canny"), torch_dtype=dt, variant="fp16")
        vae = AutoencoderKL.from_pretrained(C.snap("stabilityai/sd-vae-ft-mse"), torch_dtype=dt)
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            C.snap("SG161222/Realistic_Vision_V5.1_noVAE"), controlnet=cn, vae=vae, torch_dtype=dt, safety_checker=None).to(device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device

    @torch.no_grad()
    def __call__(self, ctx):
        c = self.cfg
        return self.pipe(prompt="RAW photo, " + C.label_prompt(ctx.label) + ", dslr, film grain", negative_prompt=C.NEG,
                         image=C.fit(ctx.sim, RES), control_image=C.fit(C.load_canny(ctx.stem), RES), strength=c["strength"],
                         num_inference_steps=c["steps"], guidance_scale=c["cfg"], controlnet_conditioning_scale=c["cn"],
                         generator=torch.Generator(self.device).manual_seed(ctx.seed)).images[0]
