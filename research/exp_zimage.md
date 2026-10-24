# Family Z: Z-Image-Turbo native tuning (agent zimage, 2026-10-18)

Code: `track_a/solution/zimage_tune/` (`model.py` = primitives, `run_z.py` = experiment registry + runner, `prompts_z.py`, `eval_z.sh`).
Nothing in `baseline_zimage_cn.py`, `init_sim/`, `hint_fusion/`, `run_experiment.py` was modified: `ZT` subclasses `ZImageControlBaseline`
and calls the pipeline directly (no hint-fusion wrapper, single canny control, no depth).

## Shared implementation and decisions (apply to every Z run)
- D1 Own denoise routine instead of reusing ZImageInitSim. Reason: ZImageInitSim runs through the hint-fusion class (default cfg = canny+depth,
  two control passes) and can only start on the 8-step Turbo grid, so strengths 0.4/0.6 would silently snap to 0.333/0.545.
  ZT injects an explicit sigma node list by patching `scheduler.set_timesteps` for the duration of one pipeline call.
  Start sigma s0 is arbitrary: nodes = [s0] + Turbo-grid nodes below s0 (s0=0.6 -> [0.6, 0.545, 0.333]; s0=0.4 -> [0.4, 0.333]).
- D2 Clean schedule everywhere (`shift_terminal=None`, reset after every call because the pipeline sets 2/3 for steps==2 and 1/2 for steps<=4).
  Verified numerically: clean 8-step sigmas `[1, .947, .882, .8, .692, .545, .333, 0, 0]`, non-clean (after the 2-step warm-up)
  `[1, .982, .961, .933, .897, .848, .778, .667, 0]`. Finding: with steps=8 the clean schedule has only 7 positive nodes, the 8th model call is at sigma=0
  with dt=0 (wasted NFE). ZT drops it (bit-identical output, ~1/7 faster). steps=9 gives 8 positive nodes (the official 8-NFE recipe) -> run `z1_clean9`.
- D3 Prompts: control = `prompts.label_prompt`. Style prompts replace the head/tail sentences only (label facts untouched), see `prompts_z.py`.
  Looking at real photos (data/real): ceiling-mounted wide-angle fisheye camera, barrel distortion, dark vignetted corners, muted warm-green colour cast,
  bright blown-out windows. They are NOT IR/dark, so `fisheye` is the main style prompt; `ir` (plan wording) is run as an alternative on Z2.
- D4 Runs: seed = 0 + scene index (exp_common), canny control, scale as stated, bf16, 960x544 latent grid (960x540 snapped to 16), L40S shared GPU, one job per GPU.
  Generation and evaluation are separate jobs (generator holds ~28 GB, evaluator child OOMs when nested, see measurement.md); evaluator = standard `evaluator_cli.py`, qwen failure model.
- D5 Seeds for sim init noise: `seed + 10000` (as ZImageInitSim). Single seed per run; thresholds from measurement.md (SAM 0.012, failure 0.14, FID 15, KID 0.03).
- D6 No real photos are used anywhere in Family Z (no `_R` variants, no fitting of any kind). `real` in smoke sheets is only for me to look at.

(Per-experiment sections and results follow below.)

## Experiments (all n=60, seed 0, one run each; thresholds: SAM 0.012, failure 0.14, FID 15, KID 0.03)
Registry and exact parameters: `zimage_tune/run_z.py` dict `E` (also saved in each run's config.json). Reference rows (runs.csv): base_canny_s1.0_label FID 254/SAM .761/fail .783/2.49 s; is_s0.5_60 FID 116/SAM .382/fail .783/1.46 s; is_s0.7_60 FID 135/SAM .565.
- **Z1** clean schedule canny, label prompt, scale 1.0 / 0.6 / 1.2, steps 9. Hypothesis: the non-clean sigma schedule in old baseline rows hurt. Decision: clean schedule + dropped no-op NFE. Smoke: 2.1 s/scene.
- **Z2** style prompts (`fisheye`, `ir`) text-to-image canny s1.0. Hypothesis: prompt wording moves colour/camera look toward real.
- **Z3** init-from-sim at start sigma 0.4 / 0.6 (off-grid, D1) + fisheye prompt; control `z3_s0.6_label` isolates the prompt.
- **Z4** control scale ramp scale(sigma)=1.2*clip((sigma-0.2)/(s0-0.2),0,1) (strong early, 0 at sigma<=0.2) at s0=0.6 and 0.75, fisheye prompt. Decision: applied with init-from-sim as the plan says; ramp is a transformer.forward wrapper.
- **Z5** region-gated init: all latents start at sigma 0.692; inside seg regions the sim latent is re-noised to the current sigma after every step until sigma falls to the region strength (people 0.545; belts/phones/cola/suitcase/baby seat/baby 0.333; background/seat/interior/exterior 0.692); soft mask (2x 3x3 blur, soft width 0.1). Fisheye prompt, scale 0.8.
- **Z6** flow inversion: 3-4 Euler steps of the Turbo model from clean sim latent up to s0 (velocity evaluated at the lower sigma, canny control, "3D render" prompt), then normal denoise with fisheye prompt. s0 0.6 and 0.75.
- **Z7** resolution: `up1.5`/`up2` = init-from-sim s0 0.6 fisheye at 1.5x/2x then Lanczos downscale; `2pass` = pass 1 at 0.75x (s0 0.6) -> Lanczos x2 to 1.5x -> pass 2 re-noised to sigma 0.4 and refined with canny at 1.5x.

### Results (numbers from report.json via runs.csv)
| run | failure | SAM | FID | KID | VGG | SSIM | clip_div | s/scene |
|---|---|---|---|---|---|---|---|---|
| z1_clean_s1.0_60 | 0.817 | 0.761 | 257.717 | 0.359 | 1.018 | 0.337 | 0.148 | 2.169 |
| z1_clean_s0.6_60 | 0.867 | 0.800 | 309.087 | 0.442 | 1.015 | 0.329 | 0.197 | 2.161 |
| z1_clean_s1.2_60 | 0.733 | 0.769 | 251.900 | 0.312 | 0.987 | 0.367 | 0.167 | 2.164 |
| z2_fisheye_t2i_60 | 0.883 | 0.755 | 265.553 | 0.380 | 0.950 | 0.351 | 0.143 | 2.189 |
| z3_s0.6_fisheye_60 | 0.817 | 0.435 | 113.243 | 0.102 | 0.693 | 0.461 | 0.104 | 1.098 |
| z1_clean9_s1.0_60 | 0.883 | 0.761 | 253.603 | 0.353 | 1.032 | 0.336 | 0.151 | 2.441 |
| z2_ir_t2i_60 | 0.850 | 0.754 | 249.421 | 0.344 | 0.936 | 0.368 | 0.139 | 2.181 |
| z3_s0.4_fisheye_60 | 0.833 | 0.290 | 125.438 | 0.126 | 0.757 | 0.464 | 0.096 | 0.824 |
| z3_s0.6_label_60 | 0.800 | 0.431 | 115.509 | 0.105 | 0.705 | 0.461 | 0.108 | 1.100 |
| z4_ramp_s0.6_60 | 0.767 | 0.450 | 112.632 | 0.096 | 0.665 | 0.465 | 0.110 | 1.111 |
| z4_ramp_s0.75_60 | 0.850 | 0.633 | 142.082 | 0.127 | 0.670 | 0.456 | 0.121 | 1.403 |
| z5_gate_60 | 0.717 | 0.515 | 127.099 | 0.116 | 0.689 | 0.460 | 0.119 | 1.235 |
| z6_inv_s0.6_60 | 0.733 | 0.360 | 192.600 | 0.241 | 0.677 | 0.501 | 0.092 | 2.056 |
| z6_inv_s0.75_60 | 0.700 | 0.420 | 231.844 | 0.306 | 0.580 | 0.516 | 0.097 | 2.629 |
| z7_up1.5_60 | 0.733 | 0.361 | 116.530 | 0.106 | 0.705 | 0.464 | 0.115 | 2.765 |
| z7_up2_60 | 0.783 | 0.345 | 116.858 | 0.110 | 0.725 | 0.466 | 0.111 | 5.765 |
| z7_2pass_60 | 0.783 | 0.503 | 134.707 | 0.130 | 0.683 | 0.472 | 0.110 | 2.748 |

### Image observations (blind sheet `out/sheets/zA.png`, 5 scenes, key `zA.key.json`; plus 3-scene smokes in `zimage_tune/out/`)
- Every sim-initialised variant (Z3-Z7) is a near-copy of the sim: same geometry, same colours (real photos are warm-green, dark corners, blown windows; outputs stay neutral grey). Gain over the sim is slightly more photographic skin/fabric texture and cleaner faces; no colour/camera-look transfer. The `fisheye` prompt did not change the look visibly (z3_s0.6_label vs fisheye identical within noise).
- Z6 (inversion) is the smoothest, painterly/blurred (FID 193-232 despite SAM .36): worse realism; the sim-render prompt in the inversion did not turn into a style change.
- Z5 shows noise-like artefacts in windows/background (background released at 0.692) and no FID gain.
- Z4 ramp s0.6 is visually equal to Z3 s0.6 (best FID 112.6, VGG 0.665, failure .767: all within noise of Z3).
- Z7 up1.5/up2: sharper, no realism gain (FID 117), 2x costs 5.8 s/scene (over budget); 2pass blurrier textures and FID 135.
- Text-to-image variants (Z1/Z2) look like clean stock photos of a different car; far from real (FID 250-310). Style prompts did not help (Z2 FID 266/249 vs Z1 258: noise).
### Verdicts
- Z1: clean schedule = baseline within noise (FID 258 vs 254); scale 1.2 slightly better on all (FID 252, failure .73, within seed SD); scale 0.6 clearly worse (FID 309, SAM .80). steps 9 no change. Dropping the no-op NFE is a free speed-up but not a quality change. Negative result for B1.
- Z2: negative. Prompt text does not move the generator toward the real camera look.
- Z3: s0.6 FID 113 / SAM .435 / 1.1 s is the best "no-real" trade seen (equal to is_s0.5 FID 116 within noise, 25% faster). s0.4 is more sim-like (SAM .29, FID 125). Near-copy flagged: not a realism win by human standards.
- Z4: no measurable gain over Z3 (differences inside noise); s0.75 worse (FID 142, SAM .63).
- Z5: negative at this setting (FID 127, SAM .515): worse than uniform s0.5-0.6; failure .717 is within noise.
- Z6: negative (FID 193/232) though failure .70-.73 and best VGG (.58) / SSIM (.52) = blur-like copy of sim, not a realism gain.
- Z7: negative/neutral: resolution does not change FID; 2x infeasible on time; 2pass worse.
- Top-2 ideas: (1) Z3 init-from-sim s0.6 off-grid (+Z4 ramp optional, 1.1 s); (2) Z1 control scale 1.2 for text-to-image (marginal). Neither is leaderboard-grade on human realism: the sim-init family is limited by the sim colour/camera domain gap, which only an appearance model (colour/grain/vignette or trained adaptation, families C/T) can close.
- Failure rate: all runs 0.70-0.88, inside seed SD 0.05 x2; not interpretable. No real photos used anywhere (no _R variants).
