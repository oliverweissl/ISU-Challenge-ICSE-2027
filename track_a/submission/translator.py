"""Track A translator: FLUX.1-schnell img2img (strength 0.6, 4 steps) with a label-derived prompt.

The simulated image is VAE-encoded, noised to strength s and denoised by FLUX with a prompt restating the scene
label (people, belts, objects), so layout stays and appearance (materials, light, faces) is repainted.
Weights come from build_weights.py (transformer stored as fp8, upcast layer-wise to bf16): no network at run time.
"""
from __future__ import annotations

import os
import zlib
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from evaluator_cli import Translator
from prompts import label_prompt

HERE = Path(__file__).resolve().parent
WEIGHTS = Path(os.environ.get("ISU_FLUX_WEIGHTS", "/opt/flux-schnell"))  # written by build_weights.py during the image build
RES = (960, 544)  # multiples of 16
STRENGTH, STEPS = 0.6, 4


def _load_transformer(folder: Path, device: str):
    from accelerate import init_empty_weights
    from diffusers import FluxTransformer2DModel
    from safetensors.torch import load_file

    config = FluxTransformer2DModel.load_config(folder)
    with init_empty_weights():
        model = FluxTransformer2DModel.from_config(config)
    state = load_file(folder / "transformer_fp8.safetensors", device=device)  # keeps the stored fp8/bf16 dtypes
    model.load_state_dict(state, assign=True)
    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    return model.eval()


class FluxTranslator(Translator):
    def __init__(self, device: str, weights: Path = WEIGHTS):
        from diffusers import FluxImg2ImgPipeline

        self.device = device
        from transformers import CLIPTextModel, T5EncoderModel

        transformer = _load_transformer(weights, device)
        # Load straight to the GPU: the evaluation container only has 14 GB of RAM.
        clip = CLIPTextModel.from_pretrained(weights / "text_encoder", torch_dtype=torch.bfloat16, device_map={"": device})
        t5 = T5EncoderModel.from_pretrained(weights / "text_encoder_2", torch_dtype=torch.bfloat16, device_map={"": device})
        self.pipe = FluxImg2ImgPipeline.from_pretrained(weights, transformer=transformer, text_encoder=clip,
                                                        text_encoder_2=t5, torch_dtype=torch.bfloat16)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.to(device)
        # 24 GB cards cannot hold T5 next to the transformer plus activations: park T5 in RAM between scenes.
        total = torch.cuda.get_device_properties(device).total_memory if device.startswith("cuda") else 0
        self.park_t5 = device.startswith("cuda") and total < 30 * 2**30

    @torch.no_grad()
    def translate(self, image: Image.Image, auxiliary_data: Optional[dict] = None) -> Image.Image:
        aux = auxiliary_data or {}
        prompt = label_prompt(aux["label"]) if aux.get("label") else "A realistic photo taken inside a car."
        seed = zlib.crc32(str(aux.get("stem", "")).encode())
        size = image.size
        if self.park_t5:
            self.pipe.text_encoder_2.to(self.device)
        prompt_embeds, pooled, _ = self.pipe.encode_prompt(prompt, prompt_2=None, device=self.device, max_sequence_length=256)
        if self.park_t5:
            self.pipe.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()
        out = self.pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled,
                        image=image.resize(RES, Image.LANCZOS if size[0] > RES[0] else Image.BICUBIC),
                        strength=STRENGTH, num_inference_steps=STEPS, guidance_scale=0.0,
                        height=RES[1], width=RES[0], generator=torch.Generator("cpu").manual_seed(seed)).images[0]
        return out.convert("RGB").resize(size, Image.LANCZOS)


def create_translator(device: str) -> FluxTranslator:
    return FluxTranslator(device)
