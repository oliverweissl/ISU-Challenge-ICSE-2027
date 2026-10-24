"""Image-build step: download FLUX.1-schnell and store the transformer as fp8 so the image stays under 40 GB.

Usage (Dockerfile): python build_weights.py /opt/flux-schnell
Local test with an existing diffusers snapshot: python build_weights.py OUT --local SNAPSHOT_DIR
Output: <out>/{model_index.json,scheduler,text_encoder,text_encoder_2,tokenizer*,vae} (bf16, as published) and
<out>/transformer_fp8.safetensors + config.json (Linear weights fp8-e4m3, norms/embeddings bf16).
"""
import shutil
import sys
from pathlib import Path

import torch
from diffusers import FluxTransformer2DModel
from huggingface_hub import snapshot_download
from safetensors.torch import save_file

REPO = "black-forest-labs/FLUX.1-schnell"
out = Path(sys.argv[1])
local = len(sys.argv) > 3 and sys.argv[2] == "--local"
tmp = Path(sys.argv[3]) if local else out.parent / "flux-download"
if not local:
    snapshot_download(REPO, local_dir=tmp, allow_patterns=[
        "model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
        "vae/*", "transformer/*"], ignore_patterns=["*.md", "*.jpeg"])
model = FluxTransformer2DModel.from_pretrained(tmp / "transformer", torch_dtype=torch.bfloat16)
model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
out.mkdir(parents=True, exist_ok=True)
save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out / "transformer_fp8.safetensors")
shutil.copy(tmp / "transformer" / "config.json", out / "config.json")
for item in tmp.iterdir():
    if item.name in ("transformer", ".cache") or item.suffix in (".safetensors", ".md", ".jpeg"):
        continue  # top-level single-file checkpoints (ae/flux1-schnell) are not needed
    shutil.copytree(item, out / item.name, dirs_exist_ok=True) if item.is_dir() else shutil.copy(item, out / item.name)
if not local:
    shutil.rmtree(tmp)
(out / "transformer").mkdir(exist_ok=True)  # FluxImg2ImgPipeline.from_pretrained expects the folder (weights passed in)
(out / "transformer" / "config.json").write_bytes((out / "config.json").read_bytes())
for p in out.rglob("*"):
    if p.is_file():
        p.chmod(0o644)
print("done", sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**30, "GiB")
