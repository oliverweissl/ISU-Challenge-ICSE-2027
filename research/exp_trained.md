# Family T: trained adaptation of Z-Image-Turbo (agent `trained`, 2026-10-18)

Code: `track_a/solution/trained_adapt/` (+ `--init` flag added to `trunk/train.py`). Checkpoints: `/data/weissl/isu_ckpt/<run>/` (LoRA 57M params ~ 230 MB each).
Logs: `track_a/solution/logs/trained_*.jsonl`, `t1_t3*.jsonl`. Sheets: `out/sheets/t_sheet1.png`, `t_sheet2.png` (keys alongside). All numbers from report.json / runs.csv.
Goal clarification (human): feature-level perturbation (colours/structure of items) with quantity/presence constant. Canny-conditioned students keep layout and counts by construction; colours come from the label prompt (not tested as perturbation here, see open items).

## Decisions
- D1 Backbone for LoRA = Z-Image-Turbo + PAI ControlNet-Union 2.1 (canny only; control scale 1.0), LoRA r=16/alpha 16 on all attention (q,k,v,out) and FFN (w1,w2,w3) linears of main layers, control layers and refiners (343 linears, 57.2M trainable), bf16 base, LoRA fp32, grad checkpointing, peak 22 GB. Why: amortise the slow init-from-sim pipeline into the control path; keeps the validated ControlNet.
- D2 Teacher (T1', no real photos): `ZImageInitSim`, canny 1.0 + depth 0.2 sum, scale 0.8, init strength 0.6, label prompt + style suffix (dim fisheye in-cabin driver-monitoring camera, muted colours, mild noise), seed 1000+i, scenes 0-599, 1.9 s/img (19 min). Student sees ONLY canny + prompt, not the sim image.
- D3 Training sigmas drawn from the clean 8-step Turbo grid (+N(0,0.03)), not uniform, to keep the distilled few-step behaviour. No flips (prompts name seats). Resolution 960x544 (inference size), latents/text cached.
- D4 Inference uses the CLEAN sigma schedule (shift_terminal=None) in all my rows (baseline rows used the warm-up schedule); `t0_noadapter_canny_60` is the adapter-free control on the same code path (matches baseline: FID 257.7 vs 254.3).
- D5 Leakage: T1' and T3b use no real photos (scenes 0-59 evaluation is clean). T2 and T3a use real photos of scenes 0-39 (_R): only scenes 40-59 (`*_hold20`: all 60 refs, 20 generated, like t1v2_*) are valid evidence. The smoke2 trunk (init of T3a) was itself trained on real 0-39.
- D6 GPU: 8.5 GB of another job was permanently resident, so `--min-free 38000` could never be satisfied; T3 jobs used `--min-free 32000` (measured peak 26 GB).
- Substitution: none needed; T3 split into T3a_R (continue smoke2 + more data) and T3b (no-real, fresh trunk) for a leakage-free comparison.

## Setup
| Exp | Data | Init | Params | Steps | Time |
|---|---|---|---|---|---|
| T1' `t1p_a` | 600 teacher pairs (scenes 0-599, style prompt) | zero LoRA | r16 all, lr 1e-4, warmup 20, batch 4 (accum), seed 0 | 439 (45 min cap; 500 planned) | ~6 s/step on shared GPU |
| T2 `t2_a` | 40 real photos scenes 0-39 (label prompt, no style) | zero LoRA | same | 300 (=30 epochs) | 30 min |
| T3a_R | 600 teacher + 40 real x3 (style prompts) | trunk t1_smoke2 (K=4, res 768) | lr 1e-4, accum 4, 700 steps | 700 | 30 min |
| T3b | 600 teacher only | fresh trunk, same cfg | same | 700 | 30 min |
Eval: `trained_adapt/run_eval.py` / `run_eval_trunk.py` via `exp_common.run_submission`, n=60, seed 0 (seed+idx), 8 steps, qwen failure model, L40S.

## Loss curves (mean per fifth of training; flow-matching MSE at random Turbo sigma, noisy)
- T1' loss: 0.438, 0.344, 0.322, 0.296, 0.296 (final 0.327 noise); T2: 0.507, 0.423, 0.375, 0.359, 0.342 (final 0.297; falls steadily = memorising 40 images).
- T3a mse_gt: 0.537, 0.452, 0.496, 0.498, 0.519 (flat); T3b: 0.601, 0.450, 0.488, 0.493, 0.525 (flat, includes knob/dropout mixing, so not comparable to LoRA).

## Results (n=60 | held-out 40-59 where applicable)
| run | scenes | failure | SAM | FID | KID | VGG | SSIM | clip_div | s/scene |
|---|---|---|---|---|---|---|---|---|---|
| t0_noadapter_canny_60 (control) | 60 | 0.817 | 0.761 | 257.7 | 0.362 | 1.018 | 0.337 | 0.148 | 2.49 |
|  | 40-59 | 1.00 | 0.765 | 278.0 | 0.381 | 1.061 | 0.328 | 0.137 | |
| t1p_a_s1.0_60 (T1', no real) | 60 | 0.883 | 0.669 | 156.5 | 0.161 | 0.644 | 0.480 | 0.113 | 2.52 |
|  | 40-59 | 0.95 | 0.671 | 188.9 | 0.183 | 0.678 | 0.465 | 0.108 | |
| t1p_a_is0.5_60 (T1' + sim init 0.5) | 60 | 0.650 | 0.366 | 149.4 | 0.163 | 0.692 | 0.481 | 0.099 | 1.11 |
|  | 40-59 | 0.70 | 0.356 | 161.9 | 0.170 | 0.730 | 0.469 | 0.088 | |
| t2_a_s1.0_60 (T2_R) | 60 (0-39 TRAIN) | 0.883 | 0.729 | 137.2 | 0.106 | 0.511 | 0.575 | 0.116 | 2.49 |
|  | 40-59 | 0.90 | 0.728 | 167.4 | 0.122 | 0.566 | 0.561 | 0.114 | |
| t3a_R_k1.0_60 (T3a_R) | 60 (0-39 TRAIN) | 0.833 | 0.672 | 155.4 | 0.142 | 0.619 | 0.493 | 0.119 | 3.17 |
|  | 40-59 | 0.85 | 0.678 | 175.8 | 0.138 | 0.676 | 0.481 | 0.133 | |
| t3b_noreal_k1.0_60 (T3b, no real) | 60 | 0.850 | 0.649 | 200.0 | 0.234 | 0.620 | 0.427 | 0.113 | 3.14 |
|  | 40-59 | 0.95 | 0.646 | 205.7 | 0.224 | 0.651 | 0.415 | 0.112 | |
References: is_s0.5_60 (training-free init-from-sim, 1.46 s): failure 0.783, SAM 0.382, FID 116.0, KID 0.105, VGG 0.700, SSIM 0.463. t1v2_k1.0_20 (old trunk, 400 steps, hold-out): SAM 0.712, FID 168.8, failure 0.85. Failure rate noise SD 0.05 (n=60), ~0.11 at n=20; FID subset n=20 is biased high (compare hold20 rows only with each other: control 278).

## Image observations
- Control (no adapter, label prompt): bright daylight cabins, often different geometry from the sim, extra/missing items; not an IR-like in-cabin camera.
- T1' (no real): matches sim layout (canny), dim cool/grey grading, fisheye look, persons/seat/baby seat/phone preserved (counts constant); textures slightly plastic, colours of objects partly lost (red baby seat OK, yellow bag often missing/dull). Visibly closer to real photos than control. Not a sim copy (SAM 0.67, vs copy 0.19).
- T2_R: reproduces real photo look (dark, greenish, motion blur), but blurry/ghosted subjects and baby seats; train-scene FID 137 is memorisation, hold-out 167 vs T1' 189 hold-out: modest generalisation gain from real targets, with quality artefacts (ghosting).
- T3a/T3b trunks: right layout, softer textures than LoRA students; T3b without real data is the worst of the trained rows.
- T1'+init0.5 ~ is_s0.5 (near sim with better grading); realism not clearly better than the training-free version.

## Verdicts
1. T1' is the best no-real result of this family: the canny-only student amortises most of the teacher gain with no sim input, FID 254 -> 157 (hold-out subset 278 -> 189), SAM 0.76 -> 0.67, VGG 1.02 -> 0.64, SSIM 0.34 -> 0.48, same 2.5 s/scene. It does NOT reach the training-free init-from-sim (FID 116, SAM 0.38), which is also faster (1.5 s); its value is that realism comes without feeding sim pixels (less copy-risk). Student+init0.5 gives the best failure (0.65, noisy) but equals is_s0.5 otherwise.
2. T2_R (real targets, 40 images) gives the best hold-out FID/KID/VGG/SSIM of the LoRA rows (167/0.122/0.57/0.56) but only vs 189 for no-real T1': real-photo supervision helps a bit but overfits (loss falling steadily, ghosting); needs organiser permission and cannot be the no-real final.
3. T3: continuing the old trunk with 600 teacher + real pairs (T3a_R) improves it slightly over t1v2 (hold-out FID 176 vs 169 on the old run is within noise; SAM 0.678 vs 0.712) but is no better than the simple LoRA (T1') and is slower (3.2 s). A no-real trunk (T3b) is clearly worse (FID 200). The trunk route is not competitive with LoRA on the ControlNet-Union; do not pursue.
4. Leakage flags: t2_a and t3a_R (and the smoke2 init) saw real photos 0-39: only hold20 rows count. No run puts real images in generated/. Seed: single seed (0); differences below FID ~15 / SAM 0.012 / failure 0.14 are noise (measurement.md).
## Top ideas
1. T1' LoRA distilled from a better/longer teacher (init-from-sim at several strengths + the Z family's best style prompt), more steps (439 steps only, loss still flat-noisy), and prompt-conditioned colour variation for feature-level perturbation (train with label colour edits, check feature_hamming).
2. T1' LoRA + low-strength init (0.5-0.6) with feature-aware prompts: best failure rate and SAM trade-off (0.65 / 0.37) but needs human realism check vs is_s0.5.
## Open / not done
Held-out numbers are single-seed n=20; no human realism rating; colour-edit prompts (feature-level perturbation test) and larger-rank/longer LoRA not run.
