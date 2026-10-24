"""Download the model weights used by family M (fp16 variants only where available)."""
import sys
from huggingface_hub import snapshot_download
J = {
 "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", ["*.json","*.txt","unet/*.fp16.safetensors","vae/*.fp16.safetensors","text_encoder*/*.fp16.safetensors","tokenizer*/*"]),
 "cn_canny": ("diffusers/controlnet-canny-sdxl-1.0", ["config.json","diffusion_pytorch_model.fp16.safetensors"]),
 "cn_depth": ("diffusers/controlnet-depth-sdxl-1.0", ["config.json","diffusion_pytorch_model.fp16.safetensors"]),
 "sdxl_vae": ("madebyollin/sdxl-vae-fp16-fix", None),
 "ip2p": ("timbrooks/instruct-pix2pix", ["*.json","*.txt","unet/*.safetensors","vae/*.safetensors","text_encoder/*.safetensors","tokenizer/*","safety_checker/*.json"]),
 "rv": ("SG161222/Realistic_Vision_V5.1_noVAE", ["*.json","*.txt","unet/*.safetensors","text_encoder/*.safetensors","tokenizer/*","vae/*.json"]),
 "cn15_canny": ("lllyasviel/control_v11p_sd15_canny", ["config.json","diffusion_pytorch_model.fp16.safetensors"]),
 "sd15vae": ("stabilityai/sd-vae-ft-mse", ["*.json","diffusion_pytorch_model.safetensors"]),
 "kontext": ("black-forest-labs/FLUX.1-Kontext-dev", ["model_index.json","transformer/*","scheduler/*"]),
 "qwen_edit": ("Qwen/Qwen-Image-Edit-2509", None),
 "qwen_lora": ("lightx2v/Qwen-Image-Lightning", ["Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"]),
 "flux_schnell": ("black-forest-labs/FLUX.1-schnell", None),
}
for k in sys.argv[1:]:
    repo, pat = J[k]
    print(k, snapshot_download(repo, allow_patterns=pat), flush=True)
