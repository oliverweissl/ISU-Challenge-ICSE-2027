"""Baseline: Z-Image-Turbo + PAI ControlNet-Union 2.1 (VideoX-Fun) as a Track A Translator.

Single control image (canny or depth) per call. This is the reference we try to beat
with a better control-fusion mechanism.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
from PIL import Image

VIDEOX_FUN = Path(os.environ.get("VIDEOX_FUN", Path.home() / "Projects" / "VideoX-Fun"))
sys.path.insert(0, str(VIDEOX_FUN))

from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from huggingface_hub import snapshot_download, hf_hub_download  # noqa: E402

from videox_fun.models import (  # noqa: E402
    AutoencoderKL, AutoTokenizer, Qwen3ForCausalLM, ZImageControlTransformer2DModel,
)
from videox_fun.pipeline import ZImageControlPipeline  # noqa: E402
from videox_fun.utils import apply_gpu_memory_mode, get_image_latent  # noqa: E402

BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
CN_REPO = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
CN_FILE = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"
CN_CONFIG = VIDEOX_FUN / "config/z_image/z_image_control_2.1.yaml"


class ZImageControlBaseline:
    def __init__(self, device: str = "cuda", dtype=torch.bfloat16, memory_mode: str = "model_full_load"):
        self.device = torch.device(device)
        self.dtype = dtype
        base = snapshot_download(BASE_REPO, local_files_only=True)
        cfg = OmegaConf.load(CN_CONFIG)

        transformer = ZImageControlTransformer2DModel.from_pretrained(
            base, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=dtype,
            transformer_additional_kwargs=OmegaConf.to_container(cfg["transformer_additional_kwargs"]),
        ).to(dtype)
        state = load_file(hf_hub_download(CN_REPO, CN_FILE, local_files_only=True))
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        print(f"controlnet load: missing={len(missing)} unexpected={len(unexpected)}")

        vae = AutoencoderKL.from_pretrained(base, subfolder="vae").to(dtype)
        tokenizer = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            base, subfolder="text_encoder", dtype=dtype, low_cpu_mem_usage=True,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler")
        self.pipe = ZImageControlPipeline(
            vae=vae, tokenizer=tokenizer, text_encoder=text_encoder,
            transformer=transformer, scheduler=scheduler,
        )
        apply_gpu_memory_mode(self.pipe, memory_mode, self.device, dtype,
                              exclude_module_name=["x_pad_token", "cap_pad_token"])

    @torch.no_grad()
    def translate(
        self,
        control: Image.Image,
        prompt: str,
        size: tuple[int, int],
        scale: float = 0.8,
        steps: int = 8,
        seed: int = 0,
    ) -> Image.Image:
        """Generate an image from one control map. `size` = original (W, H); output is resized back."""
        w, h = size
        sample = [round(h / 16) * 16, round(w / 16) * 16]  # [H, W], multiples of 16
        ctrl = get_image_latent(control, sample_size=sample)[:, :, 0]
        blank = torch.zeros([1, 3, *sample])
        mask = torch.ones([1, 1, *sample]) * 255
        out = self.pipe(
            prompt=prompt, negative_prompt=" ", height=sample[0], width=sample[1],
            generator=torch.Generator(device=self.device).manual_seed(seed),
            guidance_scale=0.0, image=blank, mask_image=mask, control_image=ctrl,
            num_inference_steps=steps, control_context_scale=scale,
        ).images[0]
        return out.resize((w, h), Image.LANCZOS)


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[2] / "data"
    stem = sys.argv[1] if len(sys.argv) > 1 else "sample_00000"
    kind = sys.argv[2] if len(sys.argv) > 2 else "canny"
    prompt = sys.argv[3] if len(sys.argv) > 3 else (
        "a photo of the interior of a car, a person sitting in the driver seat, realistic, natural daylight")
    sim = Image.open(data / "images" / f"{stem}_sim.png").convert("RGB")
    ctrl = Image.open(data / kind / f"{stem}_{kind}.png").convert("RGB")
    m = ZImageControlBaseline()
    import time
    t = time.perf_counter()
    res = m.translate(ctrl, prompt, sim.size)
    print(f"gen time {time.perf_counter() - t:.2f}s")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    row = Image.new("RGB", (sim.width * 3, sim.height))
    for i, im in enumerate([sim, ctrl, res]):
        row.paste(im, (i * sim.width, 0))
    row.save(out / f"smoke_{stem}_{kind}.png")
    print("saved", out / f"smoke_{stem}_{kind}.png")
