# methodology.md — Track A

Last updated: 2026-10-24 (ninth pass: `m5_sdxl_zimage_60`, `c3_real_s60_60_hold20` reports, see Findings 2026-10-24; no experiment files newer than 2026-10-18; eighth pass: `t3b_noreal_*`, `c5_seat_recolor_60`, `c5_babyseat_hue_60` reports, see Findings "eighth pass"; seventh pass: `m1_sdxl_*`, `m2_flux s0.5/s0.7`, `c3_*`, `t3a_R` reports, see Findings "seventh pass"; sixth pass: `m3_ip2p_60`, `m4_rv_i2i_s0.6_60` reports, see Findings "non-Z models"; fifth pass: 5 more n=60 Z-family reports + `t2_a_s1.0_60_hold20`, see Findings "Z family, fifth pass"; earlier passes added hold-out and sprint reports). Experiments and parameters: `setup.md`. Idea backlog / novelty map: `research/backlog.md`, `research/related_work.md`.

## Problem & constraints
- Task: Track A translator, simulated in-car image (+ control maps) -> realistic image. Rules text: `track_a/README.md` (not summarised here yet).
- Evaluator (`track_a/evaluator_cli.py`, `track_a_metrics.py`) reports:

| Axis | Metric(s) |
|---|---|
| Validity (structure) | SAM class-agnostic partitions, sim vs generated: ARI, NMI, matched IoU, region covering, boundary F1. `semantic_segmentation_score` = 1 - matched IoU, **lower is better** (verified: identical images -> 0.0) |
| Realism | generated vs real (and sim vs real) pixel/texture metrics, FID, KID, IS |
| Failure rate | VLM reads labelled features (driver phone, belt, passengers, objects, baby, suitcase) from the generated image and from the paired REAL photo of the same scene; any mismatch between the two readings = failure (`track_a_metrics.py`, checked by `research/verdicts.md`; earlier text of this file said "vs label", corrected 2026-10-17). We have treated lower as better; `research/backlog.md` H1 instead pushes mismatches up, so the intended optimisation direction is **unresolved** |
| Diversity | feature-level mean pairwise Hamming and CLIP cosine distance, over failing scenes only |
| Efficiency | seconds per scene (generation only) |

- Only 60 scenes have real refs: n=60 is the comparable unit; n=10 is smoke.
- Caveat: reports list failure VLM `vikhyatk/moondream2` while configs say `qwen` (unresolved).

## Backbone
- Z-Image-Turbo (`Tongyi-MAI/Z-Image-Turbo`), 8-step distilled flow-matching DiT, guidance-free.
- Rationale: few-step sampling gives low s/scene (efficiency axis); existing control ecosystem (VideoX-Fun). Comparison to other backbones not run: n/a.

## Baseline
- Z-Image-Turbo + PAI ControlNet-Union 2.1 (`baseline_zimage_cn.py`), single control (canny or depth) per call, 8 steps, label-derived prompt.
- Measured (n=60, label prompt, runs in `setup.md`):

| Run | Failure | FID gen-real | Validity (lower better) | s/scene |
|---|---|---|---|---|
| `base_canny_s0.8_label` | 0.867 | 266.7 | 0.7589 | 2.47 |
| `base_canny_s1.0_label` | 0.783 | 254.3 | 0.7612 | 2.49 |
| `base_depth_s0.8_label` | 0.850 | 447.5 | 0.5918 | 2.51 |

- Measured weaknesses:
  - Failure rate 0.78-0.87: prompt-stated attributes (phone, belt, baby seat, suitcase) are often not realised or hallucinated (qualitative, `dbg`).
  - Realism: gen-vs-real FID 254-267 is worse than raw sim-vs-real FID 133.3, i.e. under this metric the baseline output is further from real than the unmodified sim image. Caveat: FID at n=60 is high-variance.
  - Structure: matched IoU vs sim only ~0.24 (validity score ~0.76); partition agreement with sim is low.
  - Depth baseline preserves sim structure much better (validity 0.592 vs 0.759) but is far from real under FID (447.5, KID 0.705), failure 0.85. Depth-loader used is unverified (see Findings).
- Seed repeats of canny s1.0 (`base_canny_s1.0_label`, `_seed1`, `_seed2`; seeds 0 / 1000 / 2000): failure 0.783 / 0.833 / 0.883, FID 254.3 / 264.3 / 255.7, validity 0.761 / 0.757 / 0.766. Failure SD over seeds 0.05 (`research/measurement.md`).
- Not yet run/recorded: depth s1.0 failure evaluation (report has no failure section), generic prompt, scale values other than 0.8/1.0.

## Proposed method
Goal: fuse several control signals (canny, depth, later seg) in Z-Image-Turbo without the single-control weaknesses. Backlog ids from `research/backlog.md`.

| Status | Item |
|---|---|
| Implemented, partly evaluated | **F1 `hint_fusion`** (`solution/hint_fusion/`), training-free: the Union control branch runs once per modality; the 17 per-layer hints (2 noise-refiner + 15 main-layer) are merged before being added to the base transformer. Modes: `sum` (raw weights; default canny 1.0, depth 0.2), `mean` (weights normalised), `region_gated` (F2: per-token canny/depth weights from `instance_seg` classes via `gate_table.json`, averaged over each 16x16 token), `phase_gated` (F3: weights interpolate linearly or by step from `phase_start` at noise to `phase_end` at clean timestep), `conflict_aware` (F4: per token, cosine of token-mean-centred hints < threshold -> project each hint off the other, PCGrad-style, then sum/mean; ~0.7% tokens conflict on average, up to ~8% in some layers). With weights (1,0) it reproduces baseline output exactly (max diff 0, smoke test, 2 scenes). Cost: control branch runs 2x -> 5.86 s/scene measured in `hf_sum_60` (vs 2.47-2.49 baseline). Evaluated n=60: `hf_sum_60`, `hf_region_60`, `hf_conflict_60`, `hf_phase_canny_full_60` (phase_gated, depth 0.1 -> 0.3), `hf_sum_w83_60` (sum, 0.8/0.3). Cost 5.9-9.1 s/scene measured (shared GPU, causes unverified). No variant separates from plain `sum` beyond what single-seed noise allows (Findings). |
| Implemented, evaluated (n=60) | **I1 `init_sim`** (`solution/init_sim/`): SDEdit-style start. Sim image is VAE-encoded, noised to the sigma of the first used Turbo step (`x_t = (1-sigma) x0 + sigma eps`), and only the remaining steps run, with the `hint_fusion` hints (canny 1.0 / depth 0.2, `sum`) active. Strength -> sigma / steps run: 0.85 -> 0.882 / 5, 0.7 -> 0.692 / 4, 0.5 -> 0.545 / 3; 2.61 / 1.83 / 1.46 s/scene. Motivation: evaluator rewards keeping sim structure (see Findings, noisy-sim copy). Results `is_s0.85_60`, `is_s0.7_60`, `is_s0.5_60`: see Findings; the metrics trend toward those of a sim copy as strength drops, so the method trades structure change for metric score. |
| In progress (n=10 pilot, no n=60 run) | **H1 `hynea` per-scene optimisation** (code `solution/hynea/optimize.py`, `qwen_diff.py`; **dir removed from the tree by 2026-10-18, archived in `research/archive/hynea_code.tar.gz`**; backlog H1): per scene, a zero-init trunk hypernet (K blocks, run `h1_k2_10`: K=2, res 256, 150 steps max) is optimised with AdamW/OneCycle through the truncated 8-step Turbo sampler + VAE decode, starting from the VAE-encoded **`is_s0.7_60` output as anchor**. Loss = `lam_fid` x (L1 + LPIPS to anchor, optional LPIPS to sim) + `lam_st` x steering (differentiable Qwen2.5-VL teacher-forced log-prob of the **real-photo reading**, hinge at `tau`, on features not yet mismatching). Early stop when the evaluator-style Qwen reading shows `anchor mismatches + extra` mismatching features vs the real-photo reading. **It reads `data/real/` photos at generation time** and is tuned on scenes 40-49, which are inside the n=60 eval set; whether this is allowed by the Track A rules is not checked. The objective raises failure rate, opposite to the "lower is better" reading used elsewhere (see Problem). Pilot log `logs/h1_k2_10.log`: stopped at 5 of 10 scenes at last check (stop reasons: reached 3, reached_at_init 2; 20-157 s/scene optimising), no evaluation report yet. H2 (T1 trunk + H1 adaptation) queued, needs T1 checkpoint. |
| Smoke-trained only (two 400-step checkpoints; n=20 evaluation of the 768 px one) | **T1 trained control trunk** (`solution/trunk/`): frozen Z-Image-Turbo + trainable shared trunk of 8 deep-copied transformer blocks (fp32), per-modality token embeddings (canny, depth, seg) + modality embedding, null-control token, fidelity-knob embedder (copy of timestep embedder, zero-init last layer), zero-initialised output projections (output == frozen backbone at init), hints added to outputs of evenly spread base layers. Training recipe (`TrunkConfig`): modality dropout (p_all_off 0.15, keep prob 0.6/modality), knob 0 w.p. 0.10 / 1 w.p. 0.40 else U(0,1); loss = flow-matching data loss + control-off distillation to the frozen backbone (lambda 1.0, knob 0 must reproduce backbone); sigma sampling mix of uniform and Turbo 8-step grid; lr 5e-5, batch 1 x accum 8, 512 px, 10k steps, grad checkpointing. Code exists (`train.py`, `inference.py`, `dry_run.py`, `tiny_check.py`, `tools/t1_infer.py`). Smoke run `checkpoints/t1_smoke` (17:3x-17:47): 400 steps, 4 trunk blocks, 512 px, batch 1 x accum 4, lr 1e-4, 40 pairs from scenes 0-39 (`tools/build_t1_pairs.py`: **real photo as target**, sim-derived canny/depth/instance_seg controls, label prompt); last logged step 400: mse_gt 0.367, mse_distill 0.534 (noisy per-step values). Second smoke run `checkpoints/t1_smoke2`: same recipe at 768 px, lr 1e-4, 400 steps, 4 trunk blocks, 40 pairs; evaluated on scenes 40-59 at knobs 1.0 / 0.5 / 0.0 (`t1v2_*`, Findings). No n=60 generation; full 10k-step training not run. Scenes 0-39 are training scenes and overlap the n=60 eval set (the n=20 eval on 40-59 is outside the 40 training pairs). |
| Implemented, evaluated (n=60, five variants) | **C1 classical** (`solution/classical/c1.py`, `run_c1.py`; sprint family C, `research/experiments_plan.md`): pure image processing of the sim image, no network. Variants (module docstring): `hand` (no real data; hand-set warm tint, saturation boost, S-curve; `hand_fx` adds vignette, blur, grain, JPEG), `reinhard` (global Lab mean/std matched to pooled real photos of scenes 0-39), `hist` (per-channel quantile mapping to pooled real CDF of scenes 0-39), `*_fx` = variant + vignette/blur/grain/JPEG post chain. Exact constants not recorded here. Evaluated: `c1_hand_fx_60`, `c1_hand_60` (no real), `c1_hist_60`, `c1_hist_fx_60`, `c1_reinhard_60` (real photos 0-39 used for fitting; `config.json` `real_used: true`, `fit_scenes: "0-39"`). 0.02-0.18 s/scene. Results in Findings. Control for how far a trivial pipeline goes. |
| Implemented, evaluated (n=60, one run) | **C2 classical + Z-Image polish** (`solution/classical/c2.py`): C1 pre-processing, then the `init_sim` img2img path (`ZImageInitSim`, strength 0.5, `hint_fusion` `sum` canny 1.0 / depth 0.2, scale 0.8, clean schedule). Run `c2_hand_s0.5_60` uses the no-real `hand` pre-processing. 1.50 s/scene. Result in Findings. |
| Implemented, evaluated (n=60, 13 runs) | **S family surrogate pools** (`solution/surrogate/`): Z-Image-Turbo + control (canny / hed / Depth-Anything-2), text-to-image or img2img strength 0.6, prompt `label` or cached captions `cap`; candidate-pool (K=3) selection policies `vlm`, `vlm_adv`, `clip`, `sam`, `combo` on the `c06` pool, plus `k0` (no selection). Scale 0.8, seed 0. The package states no real photos are used. Results in Findings / `setup.md`. Policy definitions were not read for this document (names only); `vlm`/`vlm_adv` cost 11.3 s/scene (modelled, incl. scoring). |
| Evaluated (n=60) | **C2 variants, M2 FLUX, M3 Kontext, T1' / T2 LoRA, Z native tuning** (sprint code in `classical/`, `other_models/`, `trained_adapt/`, `zimage_tune/`): `c2_hand_s0.35_60`, `c2_reinhard_s0.5_60`, `m2_flux_i2i_s0.6/0.8_60` (FLUX img2img, 4 steps), `m3_kontext_60` (instruction editing, 20 steps), `t1p_a_*` / `t2_a_s1.0_60` / `t0_noadapter_canny_60` (`trained_adapt` LoRA on Z-Image-Turbo + canny; per `runs.csv` notes, not code: `t1p_a` trained on Z-Image teacher s0.6 outputs, no real photos; `t2_a` on real photos 0-39; `t0` = same path without LoRA), `z1_*`, `z2_fisheye_t2i_60`, `z3_s0.6_fisheye_60` (scale / step / prompt variants of Z-Image-Turbo; `fisheye` prompt written after looking at real photos). Parameters and numbers in `setup.md`. |
| Implemented, evaluated (n=60) | **C3** (`classical/c3.py`): VGG16 feature-statistics optimisation of the sim image toward a style set, 60 steps; `c3_real_s60_60` (style = real photos 0-39, leak) and `c3_zgen_s60_60` (style = 12 Z-Image t2i renders, no real). **T3a_R** (`t3a_R_k1.0_60`): T1 smoke2 trunk continued 700 steps (per `runs.csv` note: 600 teacher outputs + real photos 0-39 x3, style prompts). **M1** SDXL t2i/img2img + canny. Results in Findings. |
| Implemented, evaluated (n=60; `interior_dark` no report) | **C5 single-feature perturbation** (`classical/c5.py`; pure image processing, no network, no real data, `real_used` false): one instance-seg class of the sim image is edited, everything else pixel-identical (2 px feathered mask). Variants: `seat_recolor` (class `seat`, seeded hue, saturation 0.45, luminance kept), `babyseat_hue` (class `baby_seat`, hue rotation 60-300 deg, no-op when absent), `interior_dark` (class `car_interior` x0.6). Presence/quantity of items cannot change. Evaluated: `c5_seat_recolor_60`, `c5_babyseat_hue_60`; `c5_interior_dark_60` has a submission but no report. Purpose in the plan is not stated in the files read (n/a). Results in Findings. |
| Evaluated (n=60 + n=20 hold-out) | **T3b** (`t3b_noreal_k1.0_60`): fresh T1 trunk (K=4, 768 px), 700 steps on 600 teacher targets only (no real photos; recipe per `runs.csv` note), as no-real control for `t3a_R`. Result in Findings. |
| Code only / smoke only (no report) | (`classical/c3.py` now evaluated, see row above), `m5_sdxl_zimage_smoke3` (SDXL -> Z-Image cascade), `m3_qwen_edit_smoke3` (Qwen edit, 4 steps, LoRA), all `zimage_tune` runs now have n=60 reports (see Findings), other-model smoke runs on 3 scenes (`m1_sdxl_*`, `m3_kontext_smoke3`, `m2_flux_smoke3`; `m3_ip2p` and `m4_rv` now have n=60 reports), `smoke_t2_3`, `c2smoke_hand`, `c3smoke_real`). An earlier SDXL smoke log ends in `FileNotFoundError` (sdxl-vae-fp16-fix HF cache ref). No result claimed. |
| Planned | D1 self-generated training pairs (Z-Image-Turbo renders from label prompts -> extract canny/depth/seg) + 60 real photos; P1 label-aware prompting with per-feature diversity sampling. No code on disk. |

- Critic verdict 2026-10-17 (`research/verdicts.md`, batch 1): PIVOT away from hint-space fusion F1-F4 (no-op vs canny, 2.4-3.6x cost); iterate on I1 and measurement. This file's F1 status is unchanged (implemented, null result).
- Novelty status (`research/related_work.md`, abstract-level reads only): Cross-ControlNet and MaxFusion overlap strongly with F1/F4; AtteConDA with F2. No verdict recorded; contribution claim is not established.

## Findings
- 2026-10-17 (`base_canny_s0.8_label`, `base_canny_s1.0_label`): raising control scale 0.8 -> 1.0 lowers failure rate 0.867 -> 0.783, FID 266.7 -> 254.3, KID 0.370 -> 0.334; validity unchanged (0.7589 vs 0.7612), diversity similar (Hamming 0.265 / 0.261). Single seed, n=60: no noise estimate, so differences are not shown to be significant.
- 2026-10-17 (same runs): negative result for baseline realism. gen-vs-real FID (254-267) exceeds sim-vs-real FID (133.3); the baseline does not move images toward real under FID.
- 2026-10-17 (`ref_id_sim`): evaluator sanity check; identical images give validity 0.0, confirming lower = better. Other `ref_*` perturbation runs (noise, blur, black, blend, shuffled real, ...) are built but have no reports; evaluator gaming potential not yet quantified.
- 2026-10-17 (data): `data/depth/png` is 16-bit; `convert("RGB")` saturates it to white; depth also has noisy exterior. Fixed in `hint_fusion.method.load_control`. Which depth input `base_depth_s0.8_label` used is unverified.
- 2026-10-17 (`hint_fusion` smoke, README of the module): depth hint weight above ~0.3 next to canny 1.0 collapses output to a pastel depth render; depth-only works at scale ~0.5. Hence default depth weight 0.2. Observation from smoke tests on few scenes, not from evaluated runs.
- 2026-10-17 (`hf_sum_60`, `hf_region_60` vs `base_canny_s0.8_label`, all scale 0.8, seed 0, n=60): fusion gives small gains within likely noise. Failure 0.867 -> 0.817 / 0.783; FID 266.7 -> 255.4 / 256.0; KID 0.370 -> 0.323 / 0.334; validity 0.7589 -> 0.7571 / 0.7548 (unchanged). Failure differences are 3-5 scenes of 60; no seed repeats, so not shown significant. Fusion does not fix the FID gap to real (sim-vs-real 133.3).
- 2026-10-17 (cost): `hf_sum_60` 5.86 s/scene vs 2.47-2.49 baseline (2.4x); `hf_region_60` 9.05; `hf_conflict_60` 7.53 (shared GPU, cause of extra time unverified).
- 2026-10-17 (`ref_real_rs`, negative/diagnostic): feeding the real photos themselves yields failure rate 0.667 (40/60), validity 0.702, diversity Hamming 0.288. So the moondream2 failure reader has a high floor; baseline 0.78-0.87 is only 0.12-0.20 above it, and validity of an ideal real image is ~0.70 (generated 0.755-0.76). Metric headroom is small; gains in failure rate may be limited by the reader. `ref_real_jpeg` validity 0.713 (SAM only).
- 2026-10-17 (`hf_conflict_60`, `hf_phase_canny_full_60`, `hf_sum_w83_60`, all scale 0.8, seed 0, n=60): fusion variants are indistinguishable from each other and from baseline s0.8 on failure/FID. Failure 0.817 / 0.783 / 0.833 (baseline 0.867, `hf_sum_60` 0.817); FID 256.1 / 258.4 / 275.7 (baseline 266.7); validity 0.7561 / 0.7549 / 0.7374 (baseline 0.7589). Only `hf_sum_w83_60` (canny 0.8, depth 0.3) moves validity (0.737, depth share higher) while FID and failure get worse than `hf_sum_60`; consistent with the depth-weight smoke observation. Conflict-aware projection and phase gating show no gain over `sum`: negative result at this sample size. Best failure 0.783 (3 runs) = 47/60; no run beats `base_canny_s1.0_label` (0.783).
- 2026-10-17 (`ref_sim_noise3_s1`, `ref_sim_noise3_s0`, `ref_sim_jitter`, `ref_sim_blur`, evaluator gaming, important): sim image + Gaussian noise sigma 3 scores failure 0.783 (47/60, seed 1; seed 0: 0.733) (equal or better than best generated runs), FID 129.3 (below sim-vs-real 133.3, far below generated 254-276), validity 0.193 (generated ~0.75). Jitter FID 131.9, validity 0.166 (failure n/a). Blur gives failure 0.967, FID 328, validity 0.452. So a near-copy of sim beats all generative runs on FID and validity, and equals them on failure; the metrics reward not changing the image. A paper claim must be framed against this (e.g. the failure metric with the moondream2 reader, floor 0.667 with real photos) rather than absolute scores.
- 2026-10-17 (`ref_shuf_real`, `ref_real_jpeg`, `ref_sim_noise30`, `ref_sim_jitter`): wrong-scene real photo gives failure 0.933 (56/60) vs 0.667 for the right real photo, so the failure reader does respond to label mismatch. Real photos at JPEG q30 give 0.583 (35/60): reader noise of about 5 scenes from compression alone, so failure differences of 3-5 scenes between our runs are within reader noise. Noise sigma 30 on sim: failure 0.917, FID 208, validity 0.449. Jitter on sim: failure 0.833, FID 131.9, validity 0.166 (the sigma-3 noise copy still best: failure 0.783).
- 2026-10-17 (`base_canny_s1.0_label`, `_seed1`, `_seed2`, seed noise): canny s1.0 failure 0.783 vs 0.833 (47 vs 50 of 60) across seeds 0 / 1000; seed 2000: 0.883 (53/60). FID 254.3 / 264.3 / 255.7, validity 0.761 / 0.757 / 0.766. Seed SD (n-1): failure 0.050, FID 5.4, validity 0.004 (`research/measurement.md`). Failure range 0.78-0.88 spans every baseline/fusion/init_sim difference measured (0.78-0.87); FID range ~10 exceeds the s0.8 -> s1.0 and fusion-variant FID differences. The earlier "scale 1.0 is better" and "fusion gives small gains" statements are therefore not supported (one s0.8 seed only for those runs). Differences in validity > ~0.01 and FID gaps to depth (447) / sim copies (~130) are real.
- 2026-10-17 (`ref_id_sim`, `ref_sim_noise3_s0`, `ref_blend`, `ref_sim_noise3_s1`, failure reader on near-copies): an unmodified sim copy has failure 0.800 (48/60), sim + noise sigma 3 0.733 (seed 0, 44/60) / 0.783 (seed 1), 0.5 sim/real blend 0.700 (42/60), FID 98.2. Same-image-class variation of 0.73-0.78 between two noise seeds again equals reader noise. The best generated seed-0 run (0.783) is no better than a sim copy (0.800) on failure; mean over canny s1.0 seeds is 0.833. Failure rate does not separate generation from copying.
- 2026-10-17 (hold-out subsets; `*_hold10` = scenes 40-49, `*_hold20` = scenes 40-59, copies of `base_canny_s1.0_label` / `is_s0.7_60` images; n=10/20, so FID/KID are unreliable):

  | Run | n | Failure | FID | Validity |
  |---|---|---|---|---|
  | `base_canny_s1.0_label_hold10` | 10 | 0.9 | 283.1 | 0.770 |
  | `is_s0.7_60_hold10` (H1 anchor) | 10 | 0.7 | 163.7 | 0.548 |
  | `h1_k2_10` | 10 | 1.0 | 237.9 | 0.692 |
  | `h1ctl_k2_10` (lam_st 0) | 10 | 1.0 | 198.2 | 0.631 |
  | `h2_k4_10` | 10 | 1.0 | 300.4 | 0.647 |
  | `base_canny_s1.0_label_hold20` | 20 | 0.9 | 272.2 | 0.767 |
  | `is_s0.7_60_hold20` | 20 | 0.75 | 145.0 | 0.544 |
  | `t1s_k0.5_20` = `t1s_k1.0_20` | 20 | 0.95 | 183.6 | 0.728 |

- 2026-10-17 (`h1_k2_10`, `h1ctl_k2_10`, `h2_k4_10`, negative/inconclusive): H1/H2 reach failure 1.0 vs 0.7 for the anchor, but the control without steering loss also reaches 1.0, so the rise is not shown to come from steering; fidelity optimisation at 256 px plus re-decoding alone moves images (validity 0.548 -> 0.63-0.69). Failure saturates at 1.0 on 10 scenes, so the three runs cannot be separated. Stop reasons in logs: `reached`/`reached_at_init` in most scenes, `max_steps` in 1-4 scenes per run. Whether higher failure is the goal is unresolved (Problem); real-photo access in H1 unchecked against rules.
- 2026-10-17 (`t1s_k0.5_20`, `t1s_k1.0_20`, T1 smoke checkpoint, negative): fidelity knob 0.5 vs 1.0 gives identical validity, FID and failure (KID differs at the 7th digit): no visible knob effect at this checkpoint (400 steps, 40 pairs). Vs baseline on the same 20 scenes: failure 0.95 vs 0.9 (worse), FID 183.6 between baseline 272.2 and I1 145.0, validity 0.728 between 0.767 and 0.544. n=20, one seed: no claim.
- 2026-10-17 (`t1v2_k1.0_20`, `t1v2_k0.5_20`, `t1v2_k0.0_20`, T1 `t1_smoke2`, 768 px, scenes 40-59, n=20, one seed): the fidelity knob now has an effect.

  | Knob | Failure | FID | KID | Validity |
  |---|---|---|---|---|
  | 1.0 | 0.85 | 168.79 | 0.141 | 0.712 |
  | 0.5 | 1.0 | 263.16 | 0.244 | 0.799 |
  | 0.0 | 0.95 | 291.09 | 0.361 | 0.797 |

  Reference, same scenes: baseline canny s1.0 0.9 / 272.2 / 0.767; `is_s0.7_60` 0.75 / 145.0 / 0.544; sim-vs-real FID 148.85. Knob 1.0 beats the baseline on FID (168.8 vs 272.2) and validity (0.712 vs 0.767) after 400 training steps, but FID is still above I1 and sim-vs-real; knob 0.0 is near baseline FID. Failure differences (0.85-1.0 = 17-20 of 20) lie within reader noise. This contradicts the earlier `t1s_*` "no knob effect" result, which used the 512 px `t1_smoke` checkpoint; the two checkpoints differ in resolution (and possibly training run variance), cause of the change not isolated. n=20, FID/KID at n=20 unreliable: no claim.
- 2026-10-17 (`is_s0.85_60`, `is_s0.7_60`, `is_s0.5_60`, init_sim, scale 0.8, hints canny 1.0 / depth 0.2): 

  | Strength | Failure | FID | KID | Validity | s/scene |
  |---|---|---|---|---|---|
  | 0.85 | 0.850 (51/60) | 232.73 | 0.289 | 0.733 | 2.61 |
  | 0.7 | 0.800 (48/60) | 134.55 | 0.125 | 0.565 | 1.83 |
  | 0.5 | 0.783 (47/60) | 116.02 | 0.105 | 0.382 | 1.46 |

  FID falls to or below sim-vs-real (133.3) at strength <= 0.7 while validity moves toward 0 (copy). Failure is not better than best noise-start runs (0.783). Strength 0.85 is close to noise-start runs on validity (0.733 vs ~0.76) but FID 232.7 is better than all noise-start canny runs (254-276). Single seed each. Negative aspect: no setting improves failure rate; gains in FID come with structure copying, the same direction as the noisy-sim copy artefact below.
- 2026-10-17 (`ref_blend`, `ref_sim_gray`, `ref_black`, evaluator probes): 0.5 sim/real blend FID 98.2 (below sim-vs-real 133.3 and every generated run; failure n/a), validity 0.576. Grayscale sim: failure 0.75 (45/60, below all generated runs), FID 210.6, validity 0.350. Black images: failure 1.0, FID 738.4. Failure reader again scores trivially modified sim at or below generated runs (grayscale 0.75 vs best generated 0.783).
- 2026-10-18 (C1 classical, n=60, seed 0, no generative model; sim-vs-real FID 133.35):

  | Run | Real used | Failure | FID | KID | Validity | Diversity (Ham / CLIP) | s/scene |
  |---|---|---|---|---|---|---|---|
  | `c1_hand_60` | no | 0.75 (45/60) | 137.63 | 0.1416 | 0.1794 | 0.2820 / 0.1007 | 0.022 |
  | `c1_hand_fx_60` | no | 0.7333 (44/60) | 139.03 | 0.1419 | 0.3720 | 0.2548 / 0.1087 | 0.096 |
  | `c1_reinhard_60` | scenes 0-39 | 0.75 (45/60) | 140.69 | 0.1416 | 0.1370 | 0.2694 / 0.0946 | 0.099 |
  | `c1_hist_60` | scenes 0-39 | 0.7333 (44/60) | 126.06 | 0.1221 | 0.2304 | 0.2620 / 0.0868 | 0.087 |
  | `c1_hist_fx_60` | scenes 0-39 | 0.7333 (44/60) | 122.59 | 0.1121 | 0.3242 | 0.2661 / 0.1032 | 0.180 |
  | `c2_hand_s0.5_60` (C1 `hand` + Z-Image img2img, strength 0.5) | no | 0.7667 (46/60) | 127.70 | 0.1202 | 0.3904 | 0.2823 / 0.1058 | 1.498 |

  Hand-set and Reinhard processing leave FID at the sim level (137.6-140.7 vs 133.35); `hist` / `hist_fx` are below sim-vs-real (126.06 / 122.59), but they fit on real photos 0-39, which are inside the n=60 eval set (leak; hold-out 40-59 not evaluated separately). C2 (`c2_hand_s0.5_60`, no real data) reaches FID 127.70 and failure 0.7667, in the same range as `init_sim` strength 0.5 (116.02 / 0.783, `is_s0.5_60`; the difference is the hand pre-processing) with validity 0.390 vs 0.382: no visible gain over plain I1, 1.50 vs 1.46 s/scene. Single seed each; differences are within seed/reader noise. Failure 0.733-0.767 equals the sim copy (0.800) and noise copies (0.733-0.783) within reader noise, and is at or below all generative runs (best 0.783). `c1_hand_fx_60` and `c1_hand_60` earlier had SAM-only reports; failure/FID sections arrived later. Validity with the `_fx` post chain is higher (worse structure match) than without: `hand_fx` / `hand` 0.37 / 0.18, `hist_fx` / `hist` 0.32 / 0.23; `reinhard` 0.14. The `hand` vs `hand_fx` definitions overlap in the docstring. All sit far below generative noise-start runs (~0.76). Cost 0.02-0.18 s/scene vs 2.47 baseline. Reading: trivial pipelines match or beat every generative run on the three scored axes, the same evaluator weakness as the noisy-sim copy; a paper claim must not rest on these absolute scores. Single seed 0; seed effect not tested.
- 2026-10-18 (sprint reports, n=60, seed 0, single seed each; sim-vs-real FID 133.35; full table in `setup.md` "Sprint runs"):

  | Group | Runs | Failure | FID | Validity |
  |---|---|---|---|---|
  | Noise-start Z-Image + canny (t2i) | `z1_clean_s0.6/1.0/1.2`, `z1_clean9`, `z2_fisheye_t2i`, `z2_ir_t2i`, `t0_noadapter_canny`, `s_t2i_*` | 0.73-0.90 | 249.4-309.1 | 0.75-0.80 |
  | Noise start, other control | `s_hed_t2i_60` / `s_da2_t2i_60` | 0.667 / 0.917 | 151.35 / 214.72 | 0.425 / 0.719 |
  | img2img 0.6 Z-Image (S, Z3) | `s0_c06_k0`, `s_c06_cap`, `s_hed_06`, `s_da2_06`, `z3_s0.6_fisheye` | 0.70-0.82 | 113.2-125.5 | 0.28-0.44 |
  | Selection policies on `c06` | `s_c06_{vlm,vlm_adv,clip,sam,combo}` | 0.733-0.783 | 111.7-114.9 | 0.375-0.379 |
  | FLUX | `m2_flux_i2i_s0.6` / `s0.8`; `m3_kontext` | 0.90 / 0.933; 0.833 | 139.7 / 289.7; 140.6 | 0.527 / 0.798; 0.474 |
  | LoRA (`trained_adapt`) | `t1p_a_is0.5`, `t2_a_s1.0`, `t1p_a_s1.0` | 0.65, 0.883, n/a | 149.4, 137.2, 156.5 | 0.366, 0.729, 0.669 |
  | C2 | `c2_hand_s0.35`, `c2_reinhard_s0.5` | 0.75, 0.633 | 143.1, 121.5 | 0.289, 0.392 |

  Readings: (1) The structure of the result is unchanged: every setting that keeps validity near the sim image (< 0.5) reaches FID 111-150; every noise-start run sits at FID 250-310 and validity ~0.75, as for the baseline. (2) Selection policies (K=3) do not separate from `k0`: failure 0.733-0.783 vs 0.767, FID/validity within 3 units; failure spread equals reader noise (SD 0.05 across seeds, 5-scene compression noise). VLM selection costs 11.3 s/scene vs 1.10. Negative result for selection. (3) hed control with noise start (`s_hed_t2i_60`) has FID 151 / validity 0.425, far from other noise-start runs: it carries more sim structure than canny, so it moves metrics toward a sim copy again; failure 0.667 is the lowest of any noise-start run but within reader noise of 0.78 (+/-0.05 SD, n=60). (4) FLUX img2img strength 0.8 is as far from real as noise-start Z-Image (FID 289.7); 0.6 is in the near-copy regime but with failure 0.90 (worse than the sim copy 0.80). (5) Lowest failure among non-reference runs: `c2_reinhard_s0.5_60` 0.633 (real photos 0-39 used for fit, scored on all 60: leak), `t1p_a_is0.5_60` 0.65 (training data not checked), `s_hed_t2i_60` 0.667. These sit near the real-photo floor 0.667, but the same reader gives 0.73-0.80 to sim copies, so differences of 0.1 are ~2 SD of seed noise; not claimed as gains. (6) Control scale for noise start: scale 0.6 / 1.0 / 1.2 gives FID 309.1 / 257.7 / 251.9 and validity 0.800 / 0.761 / 0.769 (`z1_clean_*`; failure 0.867 / 0.817 / 0.733); monotone FID trend but steps are within ~2 FID seed SD only between 1.0 and 1.2; the failure drop at 1.2 (0.733, 44/60) is within the seed range 0.78-0.88 plus reader noise, not claimed. 2026-10-18: `z1_clean9_s1.0_60` (9 steps) 0.883 / 253.6 / 0.761 and `z2_ir_t2i_60` (prompt `ir`) 0.85 / 249.4 / 0.754: no gain over `z1_clean_s1.0_60`. `t0_noadapter_canny_60` (LoRA code path, no LoRA) = `z1_clean_s1.0_60` on all three metrics (0.8167 / 257.72 / 0.7612). `z1_clean_s1.0_60` (clean schedule) vs `base_canny_s1.0_label`: FID 257.7 vs 254.3, failure 0.817 vs 0.783, validity 0.761 vs 0.761; no difference beyond seed noise. (7) `fisheye` prompt (written from real-photo look) gives no gain: `z2_fisheye_t2i_60` failure 0.883, FID 265.6 vs `z1_clean_s1.0_60` 0.817 / 257.7; `z3_s0.6_fisheye_60` 0.817 / 113.2 / 0.436 vs `s0_c06_k0_60` (label) 0.767 / 114.4 / 0.385: negative result. Captions (`cap`) vs `label`: failure 0.9 vs 0.867 (t2i), 0.767 vs 0.767 (img2img): no effect. (8) C2 `hand` strength 0.35 / 0.5: FID 143.1 / 127.7, validity 0.289 / 0.390: lower strength copies sim more, same trade-off as `init_sim`.- 2026-10-18 (hold-out scenes 40-59, n=20, sim-vs-real FID 148.85; `setup.md` "Hold-out runs (sprint)"): the methods fitted/trained on real photos 0-39 were re-scored outside the fit set.

  | Run | n=60 failure / FID | Hold-out failure / FID / validity |
  |---|---|---|
  | `c1_hist_fx` | 0.733 / 122.6 | 0.70 / 134.1 / 0.310 |
  | `c1_reinhard` | 0.75 / 140.7 | 0.80 / 159.3 / 0.134 |
  | `c2_reinhard_s0.5` | 0.633 / 121.5 | 0.60 / 139.8 / 0.389 |
  | `t1p_a_is0.5` (no real photos in training) | 0.65 / 149.4 | 0.70 / 161.9 / 0.356 |
  | `t1p_a_s1.0` | 0.883 / 156.5 | 0.95 / 188.9 / 0.671 |

  Readings: (1) FID rises on hold-out for every row (+11 to +32), but n=20 FID is not comparable to n=60 FID (the sim-vs-real FID itself rises 133.35 -> 148.85); relative to that reference, `c1_hist_fx` (134.1) stays at or below sim-vs-real, `c1_reinhard` (159.3) and `t1p_a_is0.5` (161.9) do not. (2) Failure does not collapse out of sample: `c2_reinhard_s0.5` 0.60 (12/20) vs 0.633, so the low n=60 failure is not explained by the fit leak alone, but 1 scene = 0.05 and reader noise is ~0.05 at n=60, so no claim. (3) Validity is stable between n=60 and hold-out (differences <= 0.02). (4) `t1p_a_is0.5` (trained on teacher outputs, not real photos) has the lowest failure of the non-leaky sprint runs (0.65 / 0.70), still within reader noise of the sim copy (0.80 at n=60). Single seed each.
- 2026-10-18 (Z family, 5 new n=60 runs, seed 0, `fisheye` prompt, single seed; table in `setup.md` "Z family, fourth pass"; sim-vs-real FID 133.35):

  | Run | Change vs `z3_s0.6_fisheye_60` (0.8167 / 113.24 / 0.4355) | Failure | FID | Validity |
  |---|---|---|---|---|
  | `z3_s0.4_fisheye_60` | s0 0.4 | n/a | 125.44 | 0.290 |
  | `z4_ramp_s0.6_60` | scale 1.2, ramp [0.2, 0.6] | 0.7667 | 112.63 | 0.450 |
  | `z5_gate_60` | mode `gate`, per-class gate table | 0.7167 | 127.10 | 0.515 |
  | `z6_inv_s0.6_60` | mode `inv`, inv_prompt `render` | 0.7333 | 192.60 | 0.360 |
  | `z7_up1.5_60` | `mult` 1.5 | n/a | 116.53 | 0.361 |

  Readings: (1) None leaves the sim-copy regime: validity 0.29-0.52 with FID 113-127, the same trade-off as `init_sim`; no run moves failure outside reader noise (0.72-0.77 vs 0.77-0.82 for neighbours; 1 scene = 0.017, reader noise ~0.05). (2) `z6_inv` is the exception on FID (192.6, KID 0.241) at validity 0.360: worse realism than the other img2img-type runs; inversion variant not shown to help. (3) `z3_s0.4` has lowest validity here (0.290) but FID 125.4 is above s0 0.6 (113.2): lower strength copies sim structure without a FID gain. (4) Semantics of `ramp`, `mult`, `gate`, `inv` are from run names/config only; code not read. Negative/inconclusive; no claim.
- 2026-10-18 (Z family, fifth pass: 5 new n=60 runs + failure for `z3_s0.4_fisheye_60` 0.8333 and `z7_up1.5_60` 0.7333; hold-out `t2_a_s1.0_60_hold20`; seed 0, single seed; details `setup.md` "Z family, fifth pass"):

  | Run | Change | Failure | FID | Validity |
  |---|---|---|---|---|
  | `z3_s0.6_label_60` | label prompt (vs `fisheye` 0.8167 / 113.24 / 0.4355) | 0.80 | 115.51 | 0.4307 |
  | `z4_ramp_s0.75_60` | s0 0.75, scale 1.2, ramp [0.2, 0.75] | 0.85 | 142.08 | 0.6332 |
  | `z6_inv_s0.75_60` | mode `inv`, s0 0.75 | 0.70 | 231.84 | 0.4196 |
  | `z7_up2_60` | `mult` 2.0 | 0.7833 | 116.86 | 0.3453 |
  | `z7_2pass_60` | mode `pass2` | 0.7833 | 134.71 | 0.5034 |

  Readings: (1) `label` vs `fisheye` prompt at s0 0.6 again no difference (0.80 vs 0.8167, FID 115.5 vs 113.2): confirms the earlier negative result for the `fisheye` prompt. (2) Higher start strength moves toward the noise-start regime: `z4_ramp_s0.75` validity 0.633, FID 142; `z6_inv_s0.75` FID 231.8 (inv worse on FID at both 0.6 and 0.75). (3) `z6_inv_s0.75` has the lowest failure of the batch (0.70) at the worst FID; within reader noise (SD 0.05), no claim. (4) `z7_up1.5` / `z7_up2` / `z7_2pass` give validity 0.36 / 0.35 / 0.50 with FID 116.5 / 116.9 / 134.7: no gain over plain s0 0.6 (0.385 / 114.4); `z7_2pass` costs 2.7 s/scene vs 1.1. (5) `t2_a_s1.0_60_hold20` (LoRA on real photos 0-39): hold-out failure 0.90, FID 167.4 (sim-vs-real 148.85), validity 0.729; its n=60 FID 137.2 was inflated by training on scenes 0-39, and the hold-out FID is above sim-vs-real. Overall: negative/inconclusive; no Z variant leaves the copy-vs-noise-start trade-off.
- 2026-10-18 (non-Z models, n=60, seed 0, single seed; `setup.md` "Sprint runs with reports"):

  | Run | Failure | FID | Validity | s/scene |
  |---|---|---|---|---|
  | `m3_ip2p_60` (InstructPix2Pix, 20 steps) | 0.80 | 128.91 | 0.2703 | 0.767 |
  | `m4_rv_i2i_s0.6_60` (SD1.5-family img2img 0.6 + ControlNet 0.6) | 0.85 | 229.96 | 0.7072 | 0.748 |

  Readings: `m3_ip2p` is in the sim-copy regime (validity 0.27, FID near sim-vs-real 133.35, failure equal to sim copy 0.800): it barely edits the image. `m4_rv` is in the noise-start regime (validity 0.71, FID 230), failure 0.85 within reader noise. Same trade-off as every other run; no non-Z backbone leaves it. Negative result; no claim.
- 2026-10-18 (seventh pass, n=60, seed 0, single seed; `setup.md` "Seventh pass runs"; sim-vs-real FID 133.35):

  | Run | Failure | FID | Validity |
  |---|---|---|---|
  | `m2_flux_i2i_s0.5_60` | 0.75 | 131.96 | 0.3552 |
  | `m2_flux_i2i_s0.7_60` (suspect, see below) | 0.90 | 139.66 | 0.5273 |
  | `m1_sdxl_canny_60` (t2i) | 0.7833 | 307.35 | 0.7579 |
  | `m1_sdxl_i2i_s0.6_60` | 0.9333 | 209.12 | 0.6621 |
  | `c3_real_s60_60` (real 0-39 style) | 0.7167 | 147.35 | 0.3366 |
  | `c3_zgen_s60_60` (no real) | 0.6833 | 144.60 | 0.3405 |
  | `t3a_R_k1.0_60` | 0.8333 | 155.45 | 0.6721 |
  | `t3a_R_k1.0_60_hold20` (n=20; sim-vs-real 148.85) | 0.85 | 175.79 | 0.6788 |

  Readings: (1) Same trade-off as before: SDXL t2i (FID 307, validity 0.76) sits with the Z noise-start runs; FLUX 0.5 and C3 sit in the sim-copy regime (validity 0.34-0.36, FID 132-147). No new run leaves the trade-off. (2) `m2_flux_i2i_s0.7_60` has FID, validity and failure identical to `m2_flux_i2i_s0.6_60` (KID and time differ slightly); likely the same images, not verified, so the strength-0.7 point is not used. (3) C3 no-real (`c3_zgen`, failure 0.6833) is not worse than C3 with real style (0.7167); both are below the sim copy (0.80) by 0.08-0.12 but this is ~2 SD of reader noise, near-copy regime, no claim. Cheap: 0.75 s/scene. (4) `t3a_R` (trunk continued on teacher + real data) is noise-start-like (validity 0.67-0.68) with FID 155 at n=60; hold-out FID 175.8 is above sim-vs-real 148.85 and failure 0.85 is not better than baseline: no gain over `t1v2_k1.0_20` (0.85 / 168.8 / 0.712). Real photos 0-39 in its training set make the n=60 FID leaky. (5) `t0_noadapter_canny_60_hold20` (failure 1.0, FID 278.0) is the noise-start control on scenes 40-59, consistent with `base_canny_s1.0_label_hold20` (0.9 / 272.2). Negative/inconclusive.
- 2026-10-18 (eighth pass, n=60 unless noted, seed 0, single seed; `setup.md` "Eighth pass runs"; sim-vs-real FID 133.35 / 148.85 on 40-59):

  | Run | Failure | FID | Validity | s/scene |
  |---|---|---|---|---|
  | `t3b_noreal_k1.0_60` | 0.85 | 199.997 | 0.6489 | 3.144 |
  | `t3b_noreal_k1.0_60_hold20` (n=20) | 0.95 | 205.75 | 0.6457 | 3.144 |
  | `c5_seat_recolor_60` | 0.7667 | 226.22 | 0.2048 | 0.164 |
  | `c5_babyseat_hue_60` | 0.85 | 139.06 | 0.0469 | 0.149 |

  Readings: (1) T3b (no real photos) vs `t3a_R` (real 0-39 in training): FID 200.0 vs 155.4 (n=60), hold-out 205.7 vs 175.8; validity equal (0.65 vs 0.67). Real photos in training help FID by ~30-45 here, but the n=60 gap is partly leak; hold-out gap is 30 at n=20. Both stay above sim-vs-real 148.85 on hold-out and failure is not better than baseline: no trunk variant beats the copy-vs-noise-start trade-off. `research/disk_policy.md` lists the T3b checkpoint as dropped. (2) C5 edits one class and leaves the rest pixel-identical: `babyseat_hue` is almost a sim copy (validity 0.047, FID 139.1, failure 0.85 = 51/60) and `seat_recolor` (validity 0.205) has FID 226.2, worse than the sim copy 133.3, i.e. recolouring the seat alone moves FID +93; failure 0.7667 is within reader noise of the sim copy 0.80. A one-class colour edit changes failure by no more than reader noise (SD 0.05). Single seed; motivation of C5 n/a; `c5_interior_dark_60` has no report.
- 2026-10-24 (ninth pass, single seed; `setup.md` "Ninth pass runs"): `m5_sdxl_zimage_60` (SDXL -> Z-Image cascade, n=60) FID 318.74, validity 0.7545, no failure section: noise-start regime, FID worse than every Z noise-start run (249-309); no gain from the cascade. `c3_real_s60_60_hold20` (n=20, scenes 40-59) failure 0.80, FID 170.81, validity 0.3237 vs sim-vs-real 148.85: C3 with real style has n=60 FID 147.35 (above sim-vs-real 133.35) and hold-out FID 170.81 is 22 above the hold-out reference 148.85: no out-of-sample realism gain over a sim copy (n=20, FID unreliable). `c5_interior_dark_60` still has no report. Negative/inconclusive. Housekeeping: `track_a/submissions/` pruned to 19 folders; older runs are reproducible from `report.json` only, not from images.
- 2026-10-18 (`s_t2i_label_60` = `base_canny_s0.8_label`): identical failure, FID, validity, Hamming; the S-family t2i/label/canny/0.8 pool reproduces the baseline run (consistent with the same code path; not diffed at image level).
- 2026-10-17 (`base_depth_s1.0_label`, report 18:13): failure 0.85 (same as s0.8), FID 451.4 / KID 0.717 / validity 0.580 vs s0.8 447.5 / 0.705 / 0.592. Depth scale change does not fix realism.
- 2026-10-17 (`dbg`): smoke test, 9/10 failures, FID unreliable; superseded by n=60 runs.

## Open questions / next steps
- Which failure model does the official evaluator use; why does `--failure-model qwen` give moondream2 in reports?
- H1/H2: failure saturates at 1.0 on n=10 and the steering-off control matches; decide whether to extend to more scenes with a non-saturated measure (per-feature mismatches) and settle whether using real photos at generation time is permitted.
- T1: knob effect appears at `t1_smoke2` (768 px) but not `t1_smoke` (512 px); isolate cause, evaluate on n=60 (needs scenes not in training pairs, only 60 real refs exist) and decide on longer training.
- Decide the failure-rate target direction (lower vs higher) from the official rules; H1 and the earlier tables assume opposite directions.
- Repeat s0.8 baseline / fusion / init_sim with more seeds; seed SD (failure 0.05, FID 5.4) means most variant differences so far are within noise.
- init_sim: decide if a strength between 0.7 and 0.85 or single-canny hints is worth testing; so far it only moves metrics toward a sim copy.
- Decide how to handle the evaluator weakness (noisy sim copy FID 129 / validity 0.19 / failure 0.783): which score is the target, and does the challenge penalise near-copies?
- Why does depth give much better validity (0.59) but much worse FID (447)? Check depth loader used (saturated vs fixed) before concluding.
- Explain 5.9-9.1 s/scene spread of fusion runs (GPU sharing vs gating cost).
- Decide on T1 training (needs data from D1 and sign-off); fusion cost (2x control branch) vs <=5 s/scene target (`research/PROTOCOL.md`).
- Novelty verdict versus Cross-ControlNet, MaxFusion, AtteConDA.
- Commit `track_a/solution/` so runs map to a commit hash (HEAD still `2ea8759` on 2026-10-18; solution untracked).
- C1/C2: `_R` variants (`reinhard`, `hist`, `hist_fx`) were run on scenes 0-39 stats and scored on all 60 (leak), so a hold-out (40-59) score is needed; decide whether real-photo fitting is allowed (`research/experiments_plan.md` rule 4 restricts real-photo use; H1 uses real photos at generation on scenes 40-49, which conflicts with that rule as written).
- Sprint results (2026-10-18): all evaluated variants either stay at baseline-like FID ~250+ or move toward a sim copy; none shows a gain that survives seed/reader noise. Decide which axis to optimise given the evaluator weakness before more sweeps; multi-seed repeats of the best 2-3 candidates (`s_hed_t2i_60`, `t1p_a_is0.5_60`, `c2_reinhard_s0.5_60`) are needed to check failure 0.63-0.67.
- Verify the `t1p_a` / `t2_a` training data from code (current statement is from `runs.csv` notes only) and read the selection-policy definitions before using those runs in the paper; all Z-family configs now have reports (with failure); `c3` now evaluated (seventh pass; code not read, only configs); check `m2_flux_i2i_s0.7_60` vs `s0.6` for duplicate images; `runs.csv` lacks m2 s0.5/s0.7 rows; `runs.csv` has Z rows through `z7_2pass_60` (92 data rows). Z variants (`ramp`, `mult`, `inv`, `pass2`) semantics still unread from code. `runs.csv` lists n=60 for the `*_hold20` c1/c2 rows although the reports have n=20 (fix at source). `checkpoints/t1_smoke` is no longer on disk (only `t1_smoke2`).
