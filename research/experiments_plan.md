# Exploration sprint (2026-10-18): >= 20 very different Track A ideas

Goal: find a promising direction fast. Known so far (runs.csv, negative_results.md): text-to-image canny baseline FID ~258, SAM 0.76, failure ~0.83 (SD 0.05);
sim-copy FID ~130 (SAM 0.19 = near-copy exploit); init-from-sim s0.5 FID 116 / SAM 0.38 / failure 0.78 (realism unproven); training-free fusion = no-op; depth-only unrealistic.
Real photos are dim IR-like fisheye in-cabin images; sim renders look CGI. Metrics: SAM validity (lower better, <0.7 meaningful), FID/KID/VGG/SSIM vs paired real,
failure rate (noisy, saturated), CLIP/feature diversity, s/scene (<= 5 target).

## Shared rules (all agents)
1. EVERY GPU process (generation, evaluator, training, smoke test) runs through `track_a/solution/tools/gpu_run.sh [--min-free MiB] [--log FILE] -- cmd`
   (flock per GPU, one job/GPU, waits for free memory). Never call python on a GPU directly; never kill processes you did not start; no job > 40 GB.
   Training jobs: `--min-free 38000`, checkpoint to disk, set bf16 + grad checkpointing, cap batch so VRAM <= 38 GB. Only 2 GPUs are shared by ~5 agents: keep jobs short,
   smoke on 3 scenes first (n=3, evaluate=False), then n=60 + evaluate.
2. Harness: `track_a/solution/exp_common.py` (`run_submission(name, gen_fn, config, n=60, evaluate=True)`) builds submission + runs evaluator -> report.json. Existing models:
   `baseline_zimage_cn.ZImageControlBaseline`, `init_sim.ZImageInitSim` (`translate(..., sim=, strength=)`). Env `~/miniconda3/envs/zimage/bin/python`. Do not edit evaluator files, `data/`,
   VideoX-Fun, or other agents' method dirs; put code in `track_a/solution/<your_dir>/`. New pip installs go into the zimage env only if needed (pip install --no-deps preferred, note it).
3. Run naming `<expid>_<variant>_60`, unique, never overwrite. After each evaluated run: `python track_a/solution/tools/record_run.py <run> --note "<what>"`.
   Look at the images (Read tool on a 4-6 scene sheet via `tools/make_sheet.py`) before trusting numbers: low SAM/FID from near-copies of the sim is NOT a win.
4. Data-leakage rule (research/PROTOCOL.md): never put real photos/blends in `generated/`. If an idea uses real photos for anything (training, colour stats, style target) it is a `_R` variant:
   use scenes 0-39 only for fitting, report metrics on held-out scenes 40-59 as well (use `track_a/solution/analysis/per_feature.py`-style subsetting of report.json or run n=60 and
   re-aggregate), and say so. Always also give the best no-real variant of the idea if one exists.
5. Documentation is part of the deliverable: write `research/exp_<agent>.md` with, per experiment: id, hypothesis, DECISIONS taken (and why), methodology (what exactly is implemented),
   experimental setup (every parameter, seeds, steps, resolution, models + versions, GPU, time), results table (numbers from report.json only), image observations, verdict. Negative results too.
6. Budget: aim to finish all your experiments; if one is infeasible after a real attempt, record why and substitute a different idea in the same family. Report back with: table of runs
   (name, failure, SAM, FID, KID, VGG, SSIM, clip_div, s/scene), your top-2 ideas, visual quality verdicts.

## Experiments (ids are fixed; agents own families)
### Family Z - Z-Image-Turbo native tuning (agent zimage)
- Z1 clean sigma schedule baseline (backlog B1) canny s1.0, plus control-scale 0.6/1.2 sweep point.
- Z2 style-matched prompts (dim IR-like in-cabin fisheye camera, low saturation, grain) text-to-image + canny.
- Z3 init-from-sim with strengths 0.4/0.6 + style prompt (extends is_s0.5/0.7).
- Z4 timestep-scheduled control scale (strong early, decaying or off late) with init-from-sim.
- Z5 seg-region-gated init: per-region noise level (objects/belts/phones/people low strength, seats/windows/background higher) via latent mask blending during denoising.
- Z6 flow-inversion init (deterministic partial inversion of the sim latent with the Turbo model, instead of random noise) + ControlNet.
- Z7 resolution/upscale: generate at 1.5x/2x then downscale; and 2-pass (low-res structure -> high-res refine).
### Family M - other generative models (agent models)
- M1 SDXL (or SD1.5) + canny/depth ControlNet (+ img2img from sim at 0.5-0.7), label prompts.
- M2 FLUX.1-schnell/dev or SD3.5 (whatever fits 38 GB with offload) img2img from sim, optional control.
- M3 instruction/edit models ("make this a real photograph", e.g. InstructPix2Pix, FLUX Kontext, Qwen-Image-Edit, Z-Image-Edit if available) on the sim image.
- M4 sim->real via image-to-image translation of Stable-Diffusion-based real-world enhancers (e.g. "Enhancing Photorealism"-like, SDEdit w/ realistic-vision checkpoints, upscaler-refiner) at low strength.
- M5 two-model cascade: structure from model A (SDXL+ControlNet) then Z-Image-Turbo low-strength refinement (or reverse).
### Family C - classical / GAN / style (agent classical)
- C1 pure image processing sim->real: Reinhard/histogram colour matching to dataset style, IR-like desaturation, vignette, noise/grain, blur, JPEG (no network; the control for how far a trivial pipeline goes).
- C2 C1 + Z-Image img2img low strength (s0.3-0.5) as realism polish.
- C3 Neural photo-style transfer (AdaIN / WCT2 / PhotoWCT) with real photos as style targets (_R variant) and a no-real variant (style = generic photo prompts rendered by Z-Image).
- ~~C4 small unpaired CUT/CycleGAN-lite or pix2pix~~ REMOVED 2026-10-18 by the user: GANs are outdated. Never run, train or re-add GAN / CycleGAN / CUT / pix2pix-style methods. Code, checkpoints and outputs deleted.
### Family S - surrogates / verifiers / auxiliary nets (agent surrogate)
- S1 best-of-N sampling with a cheap surrogate score (CLIP realism prompt score + SAM-proxy validity + colour stats), N=3 within time budget.
- S2 VLM label verifier (Qwen2.5-VL-3B / Qwen3-VL-4B, local) in the loop: generate N candidates, keep the one whose VLM answers match the label (raises failure-metric agreement); report cost and gain.
- S3 prompt optimisation: VLM-written scene caption of the sim image (captioner as auxiliary net) fused with label facts, vs label prompt.
- S4 auxiliary depth/normal re-estimation (Depth-Anything-V2 or similar) of the sim RGB/I1 output to build a better control than the provided depth/canny (e.g. soft edges, HED/PidiNet) and test that as ControlNet input.
### Family T - trained adaptation (agent trained)
- T1' LoRA on Z-Image-Turbo (and/or the control branch) trained on self-distilled pairs (sim canny -> best-so-far refined output) = amortising a slow pipeline; no real photos.
- T2 same but with real photos 0-39 as targets (_R), held-out 40-59.
- T3 refresh the existing trunk (track_a/solution/trunk, checkpoint t1_smoke2) trained longer / with the new data; compare to the Z baseline.
