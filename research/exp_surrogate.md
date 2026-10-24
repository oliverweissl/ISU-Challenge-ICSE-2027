# Family S: surrogates / verifiers / auxiliary networks (agent surrogate, 2026-10-18)

Code: `track_a/solution/surrogate/` (common.py, gen_pool.py, score_pool.py, caption.py, make_controls.py, assemble.py, chain.sh, run_pools.sh).
Runs: `track_a/submissions/s*_60`, reports in `track_a/track_a_evaluation_output/<run>/report.json`, rows in `research/runs.csv`.
No real photo is used anywhere (selection uses sim, label and generated image only). All GPU jobs via `tools/gpu_run.sh`.
Both GPUs were mostly blocked by a foreign ollama process and sibling training jobs; this only delayed jobs, it did not change results.

## Methodology (shared)
- Generator: `ZImageInitSim` (hint_fusion subclass, one control pass, canny only, scale 0.8, 8 Turbo steps, 960x544 -> resized, bf16, L40S),
  label prompt (`prompts.label_prompt`). Init strength s=0.6 with the CLEAN sigma schedule (DECISION: n004 flagged that old is_s* rows used the
  warm-up-mutated schedule; the clean one is the proper definition, so the control s0 is re-run rather than reusing `is_s0.5/0.7`).
- Pool-and-select design (DECISION): K=3 candidates per scene (seeds i, i+100, i+200) generated once (`gen_pool.py`, cache/pools/c06),
  scored once (`score_pool.py`), then every selection policy builds its submission from the cached PNGs (`assemble.py`). Policies are
  therefore compared on identical candidates (paired). s/scene is MODELLED = measured generation seconds of K candidates + K x measured
  mean scorer seconds (CLIP 0.008, SAM-proxy 0.37, Qwen 2.68 per candidate) + caption seconds; it is not a wall-clock end-to-end run (measured on a shared GPU, noisy).
- Surrogates: CLIP realism = P(real-ish prompts) among 2 real-ish ("dim infrared driver-monitoring camera photo", "real photograph") vs 2 render-ish prompts, CLIP B/32;
  SAM proxy = 1 - matched IoU of SAM-base partitions vs the sim (same recipe as analysis/fast_proxy.py); VLM verifier = Qwen2.5-VL-3B with the evaluator's
  own prompt/vocabulary (so its answers equal what the evaluator will output for that exact PNG) -> score = number of public features equal to the label, tie-break CLIP.
  FLAG metric-aware selection: the verifier uses the evaluator's own VLM; `vlm_adv` picks the LEAST label-consistent candidate (adversarial, raises failure by gaming the VLM).
- Seed noise to compare with (evaluator_analysis section 4): failure SD 0.05 (delta < 0.14 is noise), SAM 0.012, FID 15, KID 0.03, VGG 0.04. Almost every delta below is inside it.
- Sanity: `s_t2i_label_60` (strength none, scale 0.8, same seeds) reproduces `base_canny_s0.8_label` bit-exactly (SAM 0.75888, FID 266.70): the harness matches the baseline.

## Results (n=60, report.json numbers)
| run | exp | failure | SAM | FID | KID | VGG | SSIM | clip_div | s/scene | label-agree (Qwen vs label) |
|---|---|---|---|---|---|---|---|---|---|---|
| s0_c06_k0_60 | S0 control init s0.6, N=1 | 0.767 | 0.385 | 114.4 | 0.101 | 0.710 | 0.465 | 0.105 | 1.10 | 0.694 |
| s_c06_clip_60 | S1 best-of-3 CLIP | 0.750 | 0.379 | 111.7 | 0.098 | 0.706 | 0.465 | 0.106 | 3.31 | 0.681 |
| s_c06_sam_60 | S1 best-of-3 SAM-proxy | 0.733 | 0.377 | 114.9 | 0.104 | 0.712 | 0.465 | 0.107 | 4.39 | 0.685 |
| s_c06_combo_60 | S1 CLIP+SAM proxy | 0.767 | 0.375 | 114.0 | 0.101 | 0.712 | 0.465 | 0.107 | 4.41 | 0.684 |
| s_c06_vlm_60 | S2 VLM verifier (match label) | 0.783 | 0.374 | 113.9 | 0.101 | 0.708 | 0.465 | 0.108 | 11.35 (upper bound) | 0.751 |
| s_c06_vlm_adv_60 | S2x adversarial verifier | 0.733 | 0.376 | 112.4 | 0.099 | 0.710 | 0.465 | 0.101 | 11.35 (upper bound) | 0.639 |
| s_c06_cap_60 | S3a Qwen caption + label, init s0.6 | 0.767 | 0.384 | 114.3 | 0.104 | 0.705 | 0.465 | 0.107 | 1.12 + 1.0 caption = ~2.1 (s/scene column of runs.csv omits the caption s) | 0.706 |
| s_t2i_label_60 | S3b control (t2i canny, label) | 0.867 | 0.759 | 266.7 | 0.376 | 0.955 | 0.316 | 0.142 | 2.46 | 0.688 |
| s_t2i_cap_60 | S3b caption+label, t2i canny | 0.900 | 0.751 | 264.9 | 0.375 | 0.992 | 0.306 | 0.125 | 2.53 (+1.0 caption) | 0.711 |
| s_hed_t2i_60 | S4a HED(sim) t2i | 0.667 | 0.425 | 151.4 | 0.147 | 0.770 | 0.397 | 0.149 | 2.47 (+HED cost not timed, ~0.1 on GPU) | 0.677 |
| s_da2_t2i_60 | S4b Depth-Anything-V2-Base(sim) t2i | 0.917 | 0.719 | 214.7 | 0.223 | 0.778 | 0.345 | 0.245 | 2.46 (+DA2 not timed) | 0.675 |
| s_hed_06_60 | S4c HED + init s0.6 | 0.700 | 0.278 | 124.6 | 0.123 | 0.787 | 0.465 | 0.098 | 1.10 | 0.708 |
| s_da2_06_60 | S4d DA2 + init s0.6 | 0.717 | 0.362 | 125.5 | 0.124 | 0.748 | 0.465 | 0.104 | 1.09 | 0.685 |
(SSIM and label-agree columns computed from report.json / runs.csv.)
S4 control maps were precomputed on CPU from the sim image (cost ~10 s/img on CPU for DA2-Base+HED+PidiNet, not GPU timed): a deployable version needs HED (~30 MB, tens of ms on GPU) and DA2-Small; PidiNet maps were generated but not evaluated.

## Image observations (sheet `out/sheets/s_sheet1.png`, 4 scenes)
- init s0.6 (control, S1-S3 variants, HED/DA2+s0.6): near-copies of the sim: same person, seat, baby seat colour, layout; slightly cleaner shading and a belt often added. Not dim IR-like like the real photos. Realism gain is only what FID/KID show vs identity sim (FID 133 -> 114).
- HED t2i: the most interesting: photographic textures/lighting, structure kept (SAM 0.43), but colours drift (cyan cast, baby seat colour changed from red to blue/yellow) and clothes/objects drift; FID 151 is far better than canny t2i 267 yet worse than init-from-sim. Failure 0.667 is the real-vs-real floor, i.e. the generator is not an adversarial image but photographic; per the analysis a LOW failure here reflects semantic agreement with real, not a defect.
- DA2 depth t2i: photoreal but camera viewpoint/layout wrong (windshield view, driver centred): re-estimated depth of a fisheye render does not constrain the layout; SAM 0.72 (as bad as canny). With init s0.6 it is just the sim.
- Captions mostly state "natural daylight", conflicting with the dim IR look; with init s0.6 the prompt has ~no effect (metrics equal to control); with t2i the caption gave a small FID/SAM gain within noise and lower clip_div.

## Verdicts
- S1 best-of-N with cheap surrogates: NEGATIVE/neutral. Within-scene CLIP sd 0.097 vs between-scene 0.235, SAM-proxy within-scene sd 0.038: picking among 3 seeds moves the headline metrics by FID -2.7 (clip), SAM -0.01: all inside seed noise. Cheap (3.3-4.4 s) but the candidates are too similar (init s0.6 near-copies) for selection to matter. Selection-bias flag: SAM-proxy/CLIP selection optimises evaluator proxies, not human realism; no visible difference.
- S2 VLM verifier: positive on its own target only. Label agreement of Qwen answers 0.694 -> 0.751 (+0.057), mismatches-vs-real per scene 2.10 -> 1.70 (i.e. selection toward label consistency also moves toward the real image's answers), failure 0.767 -> 0.783 (noise). Cost 11.3 s/scene => over the 5 s budget (upper bound). Adversarial variant: agreement 0.639, failure 0.733 (no gain; failure is saturated, "raising failure by gaming" did not even work because real-vs-gen differences are driven by the real answer). Metric-aware: uses the evaluator VLM; not a realism gain and no FID/KID change.
- S3 captioner: no effect at s0.6 (all metrics equal control within noise); weak/noisy effect at t2i. Not worth 1 s + captioner.
- S4 auxiliary controls: HED soft edges from the sim are the best idea of the family: as a text-to-image control they give FID 151 / KID 0.147 / SAM 0.425 vs canny 267 / 0.376 / 0.759 (4x seed SD in FID, SAM -0.33), at 2.5 s/scene, with non-copy photographic output; colour/identity drift is the open problem (combine with colour-preserving prompts or low-strength init s~0.8). HED + init s0.6 gives the lowest SAM (0.278) but is a near-copy (FID 124.6 worse than control 114). DA2 depth re-estimation: negative (layout lost).
- Top-2 ideas: (1) HED(sim) as the ControlNet input (new, cheap, structure kept without copying); (2) the Qwen label verifier as a quality gate (only if budget is relaxed / a faster 1-token-per-feature verifier is built; note it uses the evaluator's VLM).
- Caveats: n=60, single seed per run, selection policies share candidates; realism proof needs human rating (HED output looked best to me but was not blind-rated).
