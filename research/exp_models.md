# Family M: other generative models (agent `models`, 2026-10-18)

Code: `track_a/solution/other_models/` (`run_m.py` variants table, `sdxl_gen.py`, `sd15_gen.py`, `ip2p_gen.py`, `flux_gen.py`, `kontext_gen.py`,
`qwen_gen.py`, `cascade_gen.py`, `m_common.py`, `download.py`). Every GPU process ran under `tools/gpu_run.sh`. Env `zimage` (torch 2.10, diffusers 0.41.0.dev0,
transformers 5.20.1, peft 0.18; no new pip installs). Weights in `/data/weissl/.cache/huggingface/hub`. Runs: `track_a/submissions/<variant>_60`,
reports in `track_a/track_a_evaluation_output/<variant>_60/report.json`. No real photos are used anywhere in family M (no `_R` variants).

## Common decisions
- Controls: canny from `data/canny/*.png` (RGB 8-bit, used as is); depth from `data/depth/png` is **16-bit (I;16)** and saturates under `convert("RGB")`
  (re-verified: dtype uint16, min 0 / max 65535). `m_common.load_depth` reproduces the `hint_fusion.method.load_control` recipe (exterior pixels from instance_seg colour
  (8,63,122) set to far; 1-99 percentile stretch; near = bright) without importing videox_fun.
- Prompt: `prompts.label_prompt(label)` (same as baseline) for text-conditioned models; instruction models get a fixed instruction (+ optional label).
  Negative prompt for SD-family: "cartoon, 3d render, cgi, painting, illustration, drawing, oversaturated, plastic skin, deformed, blurry".
- Seeds: `ctx.seed = 0 + scene index` (exp_common). Output resized to sim size 960x540 by the harness; timing = gen_fn only (warm-up on scene 0 untimed). GPU: L40S, shared.
- Resolution: SDXL 1024x576, SD1.5 768x432, FLUX/Kontext/Qwen 960x544, resized to 960x540 at the end.
