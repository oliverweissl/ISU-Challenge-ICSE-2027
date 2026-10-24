# setup.md — Track A experiments

Last updated: 2026-10-24 (ninth pass). New since eighth pass (state on disk 2026-10-24; no experiment files newer than 2026-10-18 13:04): report `m5_sdxl_zimage_60` (n=60, SAM + realism only, no failure/efficiency section), n=20 hold-out `c3_real_s60_60_hold20`; `c5_interior_dark_60` has a report folder with SAM cache files but no `report.json` (still no metrics); see "Ninth pass runs". `track_a/submissions/` now holds 19 folders (was 89; most image folders gone, 18 have `config.json`; likely disk clean-up, cause not checked), report folders 108 (unchanged count). HEAD `2ea8759`, `track_a/solution/` still untracked. No GAN-based method is part of this project (removed 2026-10-18, one line in `research/negative_results.md`). Everything below this sentence is the eighth-pass and older header, kept as history; where it says "no report" for runs listed in "Ninth pass runs", the latter wins. Eighth pass header: New since seventh pass: n=60 reports `t3b_noreal_k1.0_60`, `c5_seat_recolor_60`, `c5_babyseat_hue_60`, n=20 hold-out `t3b_noreal_k1.0_60_hold20` (see "Eighth pass runs"); submissions without report: `c5_interior_dark_60`, smokes `m6_hybrid_flux_rv_smoke3`, `m7_flux_seat_s0.6_smoke3`, `m7_flux_seat_s0.7_smoke3`, `m7_kontext_seat_smoke3`, `m3_kontext_keep_smoke3`. Report folders on disk: 108; `track_a/submissions/` folders: 89. Values from `report.json` (failure, FID, validity, s/scene) and `research/runs.csv` (t3b rows). HEAD still `2ea8759` (`git log` this pass); `track_a/solution/` still untracked. Seventh-pass header follows. New since sixth pass: 8 n=60 reports (`m2_flux_i2i_s0.5_60`, `m2_flux_i2i_s0.7_60`, `m1_sdxl_canny_60`, `m1_sdxl_i2i_s0.6_60`, `c3_real_s60_60`, `c3_zgen_s60_60`, `t3a_R_k1.0_60`) plus n=20 hold-outs `t3a_R_k1.0_60_hold20`, `t0_noadapter_canny_60_hold20`; see "Seventh pass runs". Report folders on disk: 104; `track_a/submissions/` folders: 82. New smoke submissions without report: `m5_sdxl_zimage_smoke3`, `m3_qwen_edit_smoke3`. Metrics from `report.json` (m2 s0.5/s0.7, c3_zgen, t3a_R spot-checked) and `research/runs.csv` (others). `git log -1` unchanged since sixth pass (`2ea8759`). Sixth-pass header follows. New since fifth pass: two n=60 reports `m3_ip2p_60` and `m4_rv_i2i_s0.6_60` (rows added to "Sprint runs with reports"; values from `report.json`, agree with `research/runs.csv`, which now has 94+ rows through these two; configs seed 0, torch 2.10.0+cu128). Report folders on disk: 95; `track_a/submissions/` folders: 71. HEAD still `2ea8759`; git status unchanged in kind. No other new runs. Fifth-pass header follows. New since fourth pass: 5 Z-family n=60 reports (`z3_s0.6_label_60`, `z4_ramp_s0.75_60`, `z6_inv_s0.75_60`, `z7_up2_60`, `z7_2pass_60`; see "Z family, fifth pass"), failure sections now present for `z3_s0.4_fisheye_60` and `z7_up1.5_60`, and one n=20 hold-out report `t2_a_s1.0_60_hold20`. Report folders on disk: 94; `track_a/submissions/` folders: 70. HEAD still `2ea8759`. New-row metrics: `report.json` (spot-checked) and `research/runs.csv` (92 data rows, agrees with reports where checked). Statements in the history below that say a Z run has "no report" are superseded.

History (fourth pass header, kept for reference): New since third pass: 5 n=60 Z-family reports read from `report.json` / `config.json` (not in `research/runs.csv`, 81 data rows): `z3_s0.4_fisheye_60`, `z4_ramp_s0.6_60`, `z5_gate_60`, `z6_inv_s0.6_60`, `z7_up1.5_60` (see "Z family, fourth pass"). Submissions without report: `z3_s0.6_label_60`, `z4_ramp_s0.75_60`, `z6_inv_s0.75_60`, `z7_2pass_60`, `z7_up2_60` (configs only). Report folders on disk: 87. HEAD still `2ea8759`; git status unchanged in kind. Third pass summary: New since second pass: 10 reports: full n=60 `t0_noadapter_canny_60`, `z1_clean9_s1.0_60`, `z2_ir_t2i_60`, plus failure sections now present for `z1_clean_s1.2_60` and `t1p_a_s1.0_60`; and six **n=20 hold-out (scenes 40-59) reports** `*_hold20` for `c1_hist_fx`, `c1_reinhard`, `c2_reinhard_s0.5`, `t1p_a_is0.5`, `t1p_a_s1.0` (see "Hold-out runs (sprint)"). Report folders on disk: 82; `track_a/submissions/` holds 65 folders (new, no report: `z6_inv_s0.6_60`, `z7_up1.5_60`). HEAD still `2ea8759`. Earlier (second pass) summary: reports for all family-S runs plus 5 selection-policy runs, C2 variants, M2 FLUX (2), M3 Kontext, T1' / T2 LoRA, Z family. See "Sprint runs" tables below. Statements below that two submissions have no report (`z1_clean9_s1.0_60`, `z2_ir_t2i_60`) are superseded: both now have reports. S-family metrics are taken from `research/runs.csv` (58 data rows, not re-checked against each report.json); all other new numbers read directly from `report.json` / `config.json`. `track_a/solution/hynea/` is no longer on disk (archived as `research/archive/hynea_code.tar.gz`). Statements below that say "no report yet" for family S, or "19 folders", are superseded by the sprint tables. Sections for H1/H2 and `base_depth_s1.0_label` that say "running"/"no failure section" are stale where contradicted by the tables (H1/H2 reports exist; depth s1.0 report has failure 0.85). Numbers copied from `track_a/track_a_evaluation_output/*/report.json`, `track_a/submissions/*/{config,generation_times}.json`, `research/{runs.csv,env.md}`. `n/a` = not on disk. "derived" = arithmetic on file values.

## Environment

Source: `research/env.md` (snapshot 2026-10-17 15:01).

| Item | Value |
|---|---|
| Conda env | `zimage` (`~/miniconda3/envs/zimage`), Python 3.12.2 |
| torch / torchvision | 2.10.0 / 0.25.0 |
| diffusers | git `b800f828f3951c4d206e154411bcbfa693cd322e` |
| transformers | 5.20.1 (`track_a_metrics.py` patched for it, uncommitted: `get_image_features` ModelOutput -> `.pooler_output`) |
| other | peft 0.18.0, xformers 0.0.34, safetensors 0.8.0, torch_fidelity 0.4.0, torchmetrics 1.9.0, scikit-image 0.26.0, numpy 2.3.1 |
| GPU | 2x NVIDIA L40S 46 GB, driver 580.178.04 (shared with other users; 1 GPU per job). Timing hardware string in `generation_times.json`: `NVIDIA L40S` |
| Inference code | VideoX-Fun (`~/Projects/VideoX-Fun`, not in repo); commit n/a |
| Base model | `Tongyi-MAI/Z-Image-Turbo` rev `f332072aa78be7aecdf3ee76d5c247082da564a6` (HF cache, `local_files_only`) |
| ControlNet | `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` rev `5155fc56d17821007d6f62ac192c09e0f0e72016`, file `Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors`, config `VideoX-Fun/config/z_image/z_image_control_2.1.yaml` |
| dtype / memory | bfloat16 / `model_full_load`; fusion peak 26.9 GiB (`out/hint_fusion_smoke.json`) |
| Sampler | FlowMatchEulerDiscreteScheduler, guidance_scale 0.0, negative prompt `" "`, 8 steps (default in `translate`, not stored in config.json) |
| Evaluator | `track_a/evaluator_cli.py evaluate`; SAM `facebook/sam-vit-base` (rev 70c1a07f); CLIP `openai/clip-vit-base-patch32`; failure VLM see "Failure-model discrepancy" |
| Repo commit | HEAD `2ea8759` for all runs (unchanged 2026-10-18); `track_a/solution/`, `research/`, `paper/`, `methodology.md`, `setup.md` untracked (`research/experiments_plan.md` staged), `track_a_metrics.py`, `README.md`, `.gitignore` modified, `track_b/` staged for deletion |

## Data

| Item | Value |
|---|---|
| Source | `data/` (`data/README.md`); dataset version n/a |
| Sim images | `data/images/*_sim.png` (960x540 assumed from warmup call; not verified) |
| Controls | `data/canny/{stem}_canny.png`, `data/depth/png/{stem}_depth.png` (16-bit), `data/instance_seg/{stem}_instance_seg.png` (used by region gating and depth cleanup) |
| Labels | `data/labels/{stem}.json` -> text prompt via `solution/prompts.py` |
| Real refs | `data/real/` 60 files (`sample_00000..00059`); only these 60 scenes have real refs, so n=60 is the comparable unit. n=10 = smoke |
| Split | first `n` stems sorted by name; no train/val split (no training run yet) |
| Preprocessing | size rounded to multiples of 16, output resized back (LANCZOS) to sim size; blank `image` + all-255 mask. Fusion/`depth` runs: 16-bit depth -> 8-bit, near=bright, exterior (instance_seg) = far, 1-99 percentile stretch (`hint_fusion.method.load_control`). `Image.convert("RGB")` on the raw 16-bit PNG saturates to white |
| Seeds | per scene `seed = args.seed + i` (i from 1), args.seed 0 |

## Prompt modes (`solution/prompts.py`)

| Mode | Text |
|---|---|
| `generic` | "a photo of the interior of a car, realistic, natural light" |
| `label` | From label JSON: driver/co-driver/rear passengers (emotion, gender, t-shirt color, seat belt), driver phone, objects on front passenger seat (phone, cola bottle/can, suitcase, baby seat) |

Additional modes in the sprint runs: `cap` (cached captions, `solution/surrogate/cache/captions.json`), `fisheye` / `ir` (`solution/zimage_tune/prompts_z.py`), `label+style` (label + style suffix, `t1p_*` runs). All runs in the first experiments tables use `label`; the sprint tables state the mode per run.

## Metric conventions (verified on disk)
- `semantic_segmentation_score` = 1 - matched_iou (sim vs generated SAM partitions): `ref_id_sim` (generated = sim copy) gives 0.0; `base_canny_s0.8_label` 1 - 0.24112 = 0.75888. **Lower = structure better preserved.**
- Failure rate: fraction of scenes where VLM-read features mismatch labels (lower better). FID/KID lower better; IS higher better.
- Diversity = mean pairwise feature Hamming and CLIP cosine distance, over failing scenes only.

## Experiments table

All: Z-Image-Turbo + Union 2.1, 8 steps, seed 0 (+i), prompt `label`, single GPU L40S.

| Run | Date | Control | Method | Scale | n | Failure rate | FID gen-vs-real | KID gen-vs-real | Validity (sem. seg. score, lower better) | Diversity (Hamming / CLIP cos) | s/scene |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dbg` | 2026-10-17 | canny | baseline | 0.8 | 10 | 0.9 | 304.01 (unreliable, n=10) | 0.389 | 0.7445 | 0.2463 / 0.1010 | 2.423 |
| `base_canny_s0.8_label` | 2026-10-17 | canny | baseline | 0.8 | 60 | 0.8667 | 266.70 | 0.3704 | 0.7589 | 0.2655 / 0.1418 | 2.472 |
| `base_canny_s1.0_label` | 2026-10-17 | canny | baseline | 1.0 | 60 | 0.7833 (47/60) | 254.26 | 0.3337 | 0.7612 | 0.2611 / 0.1514 (47 failing scenes) | 2.490 |
| `base_depth_s0.8_label` | 2026-10-17 | depth | baseline | 0.8 | 60 | 0.85 (51/60) | 447.47 | 0.7052 | 0.5918 | 0.2813 / 0.1004 (51 failing) | 2.510 |
| `base_depth_s1.0_label` | 2026-10-17 | depth | baseline | 1.0 | 60 | 0.85 | 451.35 | 0.7172 | 0.5802 | 0.2778 / 0.0793 | 2.480 (report 18:13) |
| `hf_sum_60` | 2026-10-17 | canny+depth | hint_fusion `sum`, w canny 1.0 / depth 0.2 | 0.8 | 60 | 0.8167 (49/60) | 255.37 | 0.3227 | 0.7571 | 0.2697 / 0.1702 (49 failing) | 5.861 |
| `hf_region_60` | 2026-10-17 | canny+depth | hint_fusion `region_gated`, w canny 1.0 / depth 0.2, `gate_table.json` | 0.8 | 60 | 0.7833 (47/60) | 256.00 | 0.3338 | 0.7548 | 0.2602 / 0.1642 (47 failing) | 9.054 (543.25 s / 60; likely shared-GPU load, see details) |
| `hf_conflict_60` | 2026-10-17 | canny+depth | hint_fusion `conflict_aware` (conflict_base sum, thresh 0.0, centred), w 1.0 / 0.2 | 0.8 | 60 | 0.8167 (49/60) | 256.07 | 0.3203 | 0.7561 | 0.2663 / 0.1707 (49 failing) | 7.529 |
| `hf_phase_canny_full_60` | 2026-10-17 | canny+depth | hint_fusion `phase_gated`, linear; canny 1.0 -> 1.0, depth 0.1 (noise) -> 0.3 (clean) | 0.8 | 60 | 0.7833 (47/60) | 258.43 | 0.3450 | 0.7549 | 0.2834 / 0.1544 (47 failing) | 6.274 |
| `hf_sum_w83_60` | 2026-10-17 | canny+depth | hint_fusion `sum`, w canny 0.8 / depth 0.3 | 0.8 | 60 | 0.8333 (50/60) | 275.68 | 0.3411 | 0.7374 | 0.2850 / 0.1737 (50 failing) | 6.772 (406.34 s / 60) |
| `base_canny_s1.0_label_seed1` | 2026-10-17 | canny | baseline (seed repeat of `base_canny_s1.0_label`, config seed 1000) | 1.0 | 60 | 0.8333 (50/60) | 264.29 | 0.3553 | 0.7573 | 0.2738 / 0.1496 (50 failing) | 4.076 (244.58 s / 60) |
| `base_canny_s1.0_label_seed2` | 2026-10-17 | canny | baseline (seed repeat, config seed 2000) | 1.0 | 60 | 0.8833 (53/60) | 255.71 | 0.3455 | 0.7656 | 0.2847 / n/a (53 failing; CLIP value not extracted) | 7.063 (report `seconds_per_scene`) |
| `is_s0.85_60` | 2026-10-17 | canny+depth (hints as `hf` `sum`, canny 1.0 / depth 0.2) | `init_sim` strength 0.85 (first Turbo sigma 0.882, 5 of 8 steps) | 0.8 | 60 | 0.85 (51/60) | 232.73 | 0.2886 | 0.7328 | 0.2744 / 0.1271 (51 failing) | 2.609 (156.52 s / 60) |
| `is_s0.7_60` | 2026-10-17 | canny+depth (as above) | `init_sim` strength 0.7 (sigma 0.692, 4 of 8 steps) | 0.8 | 60 | 0.8 (48/60) | 134.55 | 0.1248 | 0.5647 | 0.2581 / 0.1163 (48 failing) | 1.827 (109.59 s / 60) |
| `is_s0.5_60` | 2026-10-17 | canny+depth (as above) | `init_sim` strength 0.5 (sigma 0.545, 3 of 8 steps) | 0.8 | 60 | 0.7833 (47/60) | 116.02 | 0.1055 | 0.3824 | 0.2616 / 0.1047 (47 failing) | 1.464 (87.86 s / 60) |

Subset runs (n=10/20; FID/KID unreliable at this n; sim-vs-real FID on the 20 scenes 40-59 = 148.85, KID 0.154). Scenes 40-49 / 40-59. Failure model as above.

| Run | Date | Method | n | Failure rate | FID | KID | Validity | Diversity (Hamming / CLIP cos) | s/scene |
|---|---|---|---|---|---|---|---|---|---|
| `base_canny_s1.0_label_hold10` | 2026-10-17 | copy of `base_canny_s1.0_label` images | 10 | 0.9 | 283.1 | n/a | 0.770 | n/a | n/a |
| `is_s0.7_60_hold10` | 2026-10-17 | copy of `is_s0.7_60` (H1 anchor) | 10 | 0.7 | 163.7 | n/a | 0.548 | n/a | n/a |
| `h1_k2_10` | 2026-10-17 | H1 hynea, K=2 | 10 | 1.0 | 237.9 | n/a | 0.692 | n/a | n/a |
| `h1ctl_k2_10` | 2026-10-17 | H1 control, lam_st 0 | 10 | 1.0 | 198.2 | n/a | 0.631 | n/a | n/a |
| `h2_k4_10` | 2026-10-17 | H2 (T1 trunk + H1) | 10 | 1.0 | 300.4 | n/a | 0.647 | n/a | n/a |
| `base_canny_s1.0_label_hold20` | 2026-10-17 | copy | 20 | 0.9 | 272.2 | n/a | 0.767 | n/a | n/a |
| `is_s0.7_60_hold20` | 2026-10-17 | copy | 20 | 0.75 | 145.0 | n/a | 0.544 | n/a | n/a |
| `t1s_k0.5_20` = `t1s_k1.0_20` | 2026-10-17 | T1 `t1_smoke` (512 px), knob 0.5 / 1.0 | 20 | 0.95 | 183.6 | n/a | 0.728 | n/a | n/a |
| `t1v2_k1.0_20` | 2026-10-17 | T1 `t1_smoke2` (768 px), knob 1.0 | 20 | 0.85 (17 failing) | 168.79 | 0.1414 | 0.7121 | 0.1780 / 0.1182 | 2.856 |
| `t1v2_k0.5_20` | 2026-10-17 | same, knob 0.5 | 20 | 1.0 | 263.16 | 0.2435 | 0.7993 | 0.2013 / 0.2252 | 2.840 |
| `t1v2_k0.0_20` | 2026-10-17 | same, knob 0.0 | 20 | 0.95 | 291.09 | 0.3611 | 0.7974 | 0.2084 / 0.1864 | 2.799 |

Classical runs (C1: no diffusion model; C2 row: C1 pre-processing + Z-Image-Turbo img2img; not covered by the "All:" line above). FID/KID are gen-vs-real:

| Run | Date | Method | n | Failure rate | FID | KID | Validity | Diversity | s/scene |
|---|---|---|---|---|---|---|---|---|---|
| `c1_hand_fx_60` | 2026-10-18 | C1 `hand_fx`: hand-set tint/saturation/S-curve/vignette/blur/grain/JPEG on sim, seed 0, no real data | 60 | 0.7333 (44/60) | 139.03 | 0.1419 | 0.3720 | 0.2548 / 0.1087 (44 failing) | 0.096 (5.764 s / 60) |
| `c1_hand_60` | 2026-10-18 | C1 `hand`, no real data | 60 | 0.75 (45/60) | 137.63 | 0.1416 | 0.1794 | 0.2820 / 0.1007 (45 failing) | 0.022 (`runs.csv` 0.02185) |
| `c1_reinhard_60` | 2026-10-18 | C1 `reinhard`: Lab mean/std match to pooled real 0-39 (`real_used` true) | 60 | 0.75 (45/60) | 140.69 | 0.1416 | 0.1370 | 0.2694 / 0.0946 (45 failing) | 0.099 |
| `c1_hist_60` | 2026-10-18 | C1 `hist`: per-channel quantile map to pooled real CDF 0-39, no post chain (`real_used` true) | 60 | 0.7333 (44/60) | 126.06 | 0.1221 | 0.2304 | 0.2620 / 0.0868 (44 failing) | 0.087 |
| `c1_hist_fx_60` | 2026-10-18 | C1 `hist_fx`: per-channel quantile map to pooled real CDF 0-39 + post chain (`real_used` true) | 60 | 0.7333 (44/60) | 122.59 | 0.1121 | 0.3242 | 0.2661 / 0.1032 (44 failing) | 0.180 |
| `c2_hand_s0.5_60` | 2026-10-18 | C2: C1 `hand` pre-processing, then `ZImageInitSim` img2img polish, strength 0.5, hints `sum` canny 1.0 + depth 0.2, scale 0.8, clean schedule, seed 0 (`real_used` false; `fit_scenes` "0-39" in config but unused) | 60 | 0.7667 (46/60) | 127.70 | 0.1202 | 0.3904 | 0.2823 / 0.1058 (46 failing) | 1.498 (89.86 s / 60) |

C1/C2 reports: sim-vs-real FID in each is 133.35 (KID 0.1363-0.1378). C1 SAM details: `c1_hand_fx_60` matched IoU 0.6280, ARI 0.4580, NMI 0.7535, boundary F1 0.7618; `c1_hand_60` 0.8206 / 0.7222 / 0.8871 / 0.8925; `c1_hist_fx_60` 0.6758 / 0.5081 / 0.7820 / 0.7888; `c1_reinhard_60` and `c1_hist_60` matched IoU etc. n/a (not extracted; validity only). `c2_hand_s0.5_60`: matched IoU 0.6096, ARI 0.4546, NMI 0.7510, region covering 0.6190, boundary F1 0.7266. Gen-vs-real PSNR / SSIM: `c1_hist_fx_60` 13.01 / 0.4576; `c1_hist_60` 12.84 / 0.4555 (`runs.csv`); `c2_hand_s0.5_60` 12.19 / 0.3621 (report); others n/a (not extracted). IS gen: `c1_hand_fx_60` 1.282, `c1_hand_60` 1.262, `c1_reinhard_60` 1.277, `c1_hist_fx_60` 1.263, `c1_hist_60` 1.232, `c2_hand_s0.5_60` 1.370. Hist/Reinhard fit on real photos 0-39 and are scored on all 60 scenes (leak, no hold-out report).

Values for the first eight rows are copied from the earlier version of this file/methodology (reports on disk; KID, diversity, s/scene not re-extracted this update). `t1v2_*`: scenes sample_00040..00059 (n=20), `config.json`: method `t1_trunk`, scale 1.0, `ckpt` `checkpoints/t1_smoke2/trunk_000400.pt`, 8 steps, clean schedule, seed idx+1, prompt label; reports 18:55 / 18:59 / 19:02.

Reference for realism: sim-vs-real FID = 133.35 (all n=60 reports; KID 0.136-0.138; dbg n=10: 158.26). Generated canny-based noise-start images are further from real (FID 254-276) than raw sim images under this metric; depth baseline is worse (447-451). `init_sim` at strength 0.7 / 0.5 gets FID 134.55 / 116.02 (0.85: 232.73), i.e. at or below sim-vs-real, but validity 0.565 / 0.382 shows these images stay close to sim. IS gen: 1.478 (canny s0.8), 1.471 (canny s1.0), 1.566 (depth s0.8), 1.556 (`hf_sum_60`), 1.474 (`hf_region_60`), 1.544 (`hf_conflict_60`), 1.518 (`hf_phase_canny_full_60`), 1.632 (`hf_sum_w83_60`); IS sim-vs-real side 1.247-1.271 (varies per report).

Other metrics s0.8 / s1.0 (gen vs real): PSNR 8.540 / n/a, SSIM 0.3161 / n/a (s1.0 not read). SAM partition (s0.8): ARI 0.0610, NMI 0.4681, matched IoU 0.2411, boundary F1 0.2290. (s1.0): ARI 0.0615, NMI 0.4783, matched IoU 0.2388, boundary F1 0.2271.

### Evaluator sanity runs (`ref_*`, not generative methods)
Built by `solution/analysis/build_ref_submissions.py` from the 60 sim/real pairs. Purpose: probe evaluator behaviour/gaming.

| Run | Generated image |
|---|---|
| `ref_id_sim` | sim copy |
| `ref_real_rs` | real resized to sim size |
| `ref_real_jpeg` | real, JPEG q30 |
| `ref_sim_blur` | sim, Gaussian blur r=6 |
| `ref_sim_jitter` | sim, random brightness/contrast/colour/hue |
| `ref_sim_noise30` / `ref_sim_noise3_s0` / `ref_sim_noise3_s1` | sim + Gaussian noise sigma 30 / 3 (seed 0 / 1) |
| `ref_shuf_real` | real of next scene (resized) |
| `ref_black`, `ref_blend`, `ref_sim_gray` | black / 0.5 sim-real blend / grayscale sim |

Results on disk (n=60; efficiency not applicable):

| Run | Validity | Failure rate | FID gen-vs-real | KID | Diversity (Hamming / CLIP cos) | Note |
|---|---|---|---|---|---|---|
| `ref_id_sim` | 0.0 | 0.8 (48/60) | 133.35 (equals sim-vs-real) | 0.1375 | 0.2775 / n/a (48 failing) | report 17:34, full; sim copy |
| `ref_black` | 0.6211 | 1.0 (60/60) | 738.36 | 1.2385 | 0.2806 / -1.2e-7 (60 failing; identical images) | black images; generation time recorded as 0.0 (not applicable) |
| `ref_blend` | 0.5759 | 0.7 (42/60) | 98.20 | 0.0847 | 0.2748 / n/a (42 failing) | 0.5 sim + 0.5 real; full report 17:54 (KID 0.0840 in the earlier partial report) |
| `ref_sim_gray` | 0.3504 | 0.75 (45/60) | 210.55 | 0.2840 | 0.2932 / 0.0774 (45 failing) | grayscale sim |
| `ref_real_rs` | 0.7023 | 0.6667 (40/60) | 0.417 | -0.0053 | 0.2877 / 0.1135 | full report; generated = real photos |
| `ref_real_jpeg` | 0.7133 | 0.5833 (35/60) | 9.918 | -0.0002 | n/a (not extracted) | full report (15:42); real photos JPEG q30 |
| `ref_shuf_real` | 0.7183 (1 - matched IoU 0.2817) | 0.9333 (56/60) | 0.417 | -0.0054 | 0.2744 / 0.1154 (56 failing) | real of next scene; FID equals `ref_real_rs` since the image set is the same |
| `ref_sim_blur` | 0.4516 | 0.9667 (58/60) | 328.12 | 0.4916 | 0.2721 / 0.0580 (58 failing) | full report |
| `ref_sim_noise30` | 0.4489 | 0.9167 (55/60) | 207.999 | 0.2708 | 0.2809 / 0.1020 (55 failing) | sim + Gaussian noise sigma 30 |
| `ref_sim_noise3_s1` | 0.1932 | 0.7833 (47/60) | 129.33 | 0.1320 | 0.2844 / 0.0890 (47 failing) | sim + Gaussian noise sigma 3 |
| `ref_sim_noise3_s0` | 0.1987 | 0.7333 (44/60) | 129.17 | 0.1317 | 0.2679 / n/a (44 failing) | full report 17:44; lowest failure rate of any run on disk besides `ref_real_*` and `ref_blend` |
| `ref_sim_jitter` | 0.1656 | 0.8333 (50/60) | 131.89 | 0.1331 | 0.2875 / 0.1071 (50 failing) | report refreshed 16:24, now full |

All `ref_*` runs and `base_depth_s1.0_label` have a full report on disk. A VLM-free `fast_proxy.py` (SAM proxy, KID/FD on ResNet50, HF ratio, colour EMD) exists; no proxy outputs recorded here.

## Per-experiment details

### Failure-model discrepancy (all runs)
`config.json` has `failure_model: "qwen"` (CLI default of `run_experiment.py`), but every `report.json` lists inference model `vikhyatk/moondream2` (revision `main`, temp 0, max_tokens 512). Cause unknown; check `evaluator_cli.py`. Failure rates are comparable across runs here (same model), but not necessarily to the intended Qwen evaluator.

### dbg
- `python run_experiment.py --name dbg --control canny --prompt label --n 10 --eval`. Smoke only; FID/KID on 10 vs 10 images. IS 1.0. SSIM 0.3187, PSNR 8.61, VGG perceptual 0.980. Partition: ARI 0.073, NMI 0.474, matched IoU 0.255, boundary F1 0.240.
- Warmup 1 call (2 steps) before timing; s/scene = generation only (excl. load, eval). Total 24.23 s / 10.
- Failures seen: driver phone and belt hallucinated (sample_00002-4), baby/baby seat hallucinated (_00003), suitcase attributes missed (_00001); only sample_00000 matched.

### base_canny_s0.8_label / base_canny_s1.0_label / base_depth_s0.8_label
- `run_experiment.py --name <run> --control {canny|depth} --scale {0.8|1.0} --prompt label --n 60 --eval`. Total gen time 148.30 s / 149.43 s / 150.63 s.
- `base_canny_s0.8_label` config.json predates `--method` flags (no `method` field), others record `method: baseline`.
- `base_depth_s0.8_label`: report.json now present (see table). Which depth loader was used is not recorded: `hint_fusion/README.md` says the grid used the saturated-white 16-bit depth; current `run_experiment.py` uses the fixed loader. Treat as unverified.
- Records in `research/runs.csv`: `dbg`, `base_canny_s0.8_label` only (s1.0 missing).

### hf_sum_60
- `--method hint_fusion --fusion-mode sum --n 60 --eval`; effective weights canny 1.0, depth 0.2 (`FusionConfig` defaults), control branch run twice, hints summed (see methodology). Full config in `submissions/hf_sum_60/config.json`.
- Generation 351.63 s (60 images); evaluation finished OK (`logs/eval_queue.log`). Results in table. `chain_a.status` records "hf_sum_60 exit 1" for the generation step but images, generation_times and report exist; cause of exit 1 not checked.
- Smoke test (`out/hint_fusion_smoke.json`, samples 00000 and 00003, shared GPU): all modes 4.1-4.9 s/img, peak 26.88 GiB; `canny_only` weights (1,0) reproduce baseline output with max diff 0. These timings are not comparable to the table (different load).

### hf_region_60, hf_conflict_60
- Rerun after the `load_control` fix. Both generated 60 images (543.25 s / 451.76 s) with `run_experiment.py --method hint_fusion --fusion-mode {region_gated|conflict_aware} --n 60 --eval`; weights canny 1.0 / depth 0.2, scale 0.8, seed 0 (`config.json`).
- `hf_region_60`: evaluation OK. Gen time 9.05 s/scene vs 5.86 for `hf_sum_60`; log shows ~1.1 s/it denoising (similar to others), so extra time is probably gate computation or a shared GPU; not verified.
- `hf_conflict_60`: first evaluation crashed with CUDA OOM; re-evaluated OK later (report 15:49, see update below).
- `chain_a.status` lists exit 1 for generation of all hf runs although outputs exist; cause not checked.

### hf_conflict_60 (update)
- Earlier OOM in evaluation resolved: `logs/eval_queue.log` lists `hf_conflict_60 eval ok`; report 15:49.

### hf_phase_canny_full_60, hf_sum_w83_60
- Both rerun and evaluated (`eval_queue.log`: eval ok; reports 16:10 / 16:00). `chain_a.status` again lists "exit 1" for all five hf generation steps despite outputs; cause not checked.
- `hf_phase_canny_full_60`: `--fusion-mode phase_gated`, `--fusion-config {"phase_start":{"canny":1.0,"depth":0.1},"phase_end":{"canny":1.0,"depth":0.3}}`, shape linear, scale 0.8. Generation 376.47 s. Meaning of "full" in name n/a.
- `hf_sum_w83_60`: `--fusion-mode sum --fusion-config {"weights":{"canny":0.8,"depth":0.3}}`, scale 0.8. Generation 406.34 s.
- Per-run s/scene differences (5.9-9.1) are not explained; GPU shared with other users.

### base_canny_s1.0_label_seed1 / _seed2 (seed repeats)
- Repeats of `base_canny_s1.0_label`: `config.json` seeds 1000 / 2000 (original 0), method baseline, scale 1.0, n 60. (config also holds unused `init_strength` 0.7 and `fusion_mode` mean fields; irrelevant for baseline.) Generation 244.58 s / 423.81 s total (differences likely shared GPU; not verified). `logs/seeds.status` lists "exit 1" for both generation steps although outputs exist (cause not checked).
- Spread over three seeds (seed 0 / 1000 / 2000): failure 0.7833 / 0.8333 / 0.8833; FID 254.26 / 264.29 / 255.71; KID 0.3337 / 0.3553 / 0.3455; validity 0.7612 / 0.7573 / 0.7656. `_seed2` report complete at 17:30. Sample SD over 3 seeds (`research/measurement.md`): failure 0.050, FID 5.42, validity 0.0041.

### is_s0.7_60 / is_s0.85_60 / is_s0.5_60 (init_sim)
- `run_experiment.py --name is_s0.7_60 --method init_sim --init-strength 0.7 --fusion-mode sum --prompt label --n 60 --eval` (same with 0.85, 0.5); scale 0.8, seed 0, hints canny 1.0 / depth 0.2 (hint_fusion `sum`). Strength -> first Turbo sigma / steps run: 0.85 -> 0.882 / 5, 0.7 -> 0.692 / 4, 0.5 -> 0.545 / 3 (`init_sim/README.md`). Launched via `logs/chain_is.sh` on GPU 1; `chain_is.status`: all three exit 0. Generation 109.59 s / 156.52 s / 87.86 s. All three evaluated (reports 16:58 / 17:10 / 17:22). `init_sim` overrides the scheduler to the clean schedule; baseline/hf runs use sigmas 1..0.667 then 0 because of the warm-up `shift_terminal` mutation (`init_sim/README.md`). Smoke log `logs/init_sim_smoke.log`.

### h1_k2_10 (H1 pilot; reports now exist for h1_k2_10, h1ctl_k2_10, h2_k4_10, see subset table; the "State at 17:56" text below is historical)
- 2026-10-18 note: `track_a/solution/hynea/` no longer exists; code archived in `research/archive/hynea_code.tar.gz` (contents not inspected). Commands below cannot be rerun from the tree as is.
- Command (from docstring, `solution/hynea/optimize.py`): `python -m hynea.optimize --name h1_k2_10 --K 2 --res 256 --steps 150 --device cuda:0`, run from `track_a/solution`. Scenes `sample_00040..00049`; anchor = `submissions/is_s0.7_60/generated`; seed `1000 + idx`; strength default and loss weights `lam_*`, `tau`, `lr`, `extra`, `check_every`: n/a (CLI defaults not read).
- Reasoning model: Qwen2.5-VL via `hynea/qwen_diff.py` (differentiable); readings cache `out/hynea/h1_k2_10/readings.json`. Uses real photos for the steering target (see methodology).
- State at 17:56: 5 images in `submissions/h1_k2_10/generated`; log lines: sample_00040 init 0 -> final 3 mismatches (target 3, 101 s); _00041 3 -> 6 (target 6, 157 s); _00042 reached at init (4, target 3); _00043 reached at init (8, target 2); _00044 1 -> 3 (target 3, 20 s); _00045 running. Peak VRAM 26.1 GB. No evaluator report, so no failure/FID/validity values (n/a).
- `submissions/` also has `base_canny_s1.0_label_hold10`, `_hold20`, `is_s0.7_60_hold10`, `_hold20`: 10 / 20 image subsets copied from `base_canny_s1.0_label` / `is_s0.7_60` (`generation_times.json`: "copy of ..."; hold20 not opened), no `config.json`, no report. Purpose not documented.

### t1_smoke2 (T1 trunk, 768 px) and t1v2_* evaluation
- `checkpoints/t1_smoke2/cfg.json`: n_trunk 4, resolution 768, batch 1, grad_accum 4, max_steps 400, warmup 30, lr 1e-4, ckpt_every 400; `trunk_000400.pt`; `pairs.jsonl` 40 lines (identity with `t1_smoke` pairs not diffed). Log `logs/t1_smoke2.jsonl`: step 400 at 1002 s, mem 26.2 GB, mse_gt 0.531, mse_distill 0.427 (single noisy step); an earlier attempt crashed at step 200 (`logs/t1_smoke2_crashed_at200.jsonl`, cause not checked).
- Inference `logs/t1v2_infer.log`: knobs 1.0 / 0.5 / 0.0 on scenes 40-59 (also 0-59 timing lines logged up to sample_00059; only 20 evaluated). Evaluated via `logs/eval_t1v2_*.log`.
- Same 20 scenes, other runs: baseline copy `_hold20` 0.9 / 272.2 / 0.767, `is_s0.7_60_hold20` 0.75 / 145.0 / 0.544 (failure / FID / validity).

### t1_smoke (T1 trunk smoke training, no evaluation)
- `checkpoints/t1_smoke/cfg.json`: n_trunk 4, resolution 512, batch 1, grad_accum 4, max_steps 400, warmup 30, lr 1e-4, ckpt_every 100. Checkpoints `trunk_000100..000400.pt`, `last.pt`. Data: `pairs.jsonl`, 40 pairs from scenes 0-39, real photo as target, sim controls (canny, depth, instance_seg), label prompt. Logs `logs/t1_smoke.out/.jsonl`; last step 400 sec 779, mem 26.2 GB. Loss values in methodology. Dry-run `logs/trunk_dry_run.json`, `trunk_dry_translate.png` (17:32). `tools/t1_infer.py` exists; no submission folder generated from this checkpoint.

### c1_* (classical C1) and c2_hand_s0.5_60
- `config.json` per run: family C1, n 60, seed 0; variant / `real_used` / `fit_scenes`: `hand_fx` / false / null; `hand` / false / null; `reinhard` / true / "0-39"; `hist_fx` / true / "0-39". Code `solution/classical/{c1.py,run_c1.py,chain_c1.sh}`; real stats cache `real_stats_0_39.npz`. Logs `logs/c1_{hand_fx,hand,reinhard,hist_fx}.log`; `logs/classical.status`: `c1_hand_fx_60`, `c1_hist_fx_60`, `c1_reinhard_60` exit 0 (`c1_hand_60` not listed there). Hardware `NVIDIA L40S`. Exact filter constants not recorded here (n/a).
- Reports: `c1_hand_fx_60` (03:08 SAM-only at first check, now full with failure/FID/diversity/efficiency); `c1_hist_fx_60` 03:18; `c1_reinhard_60` 03:27; `c1_hand_60` 03:37 first version was SAM + realism only, later rewritten with failure 0.75 / diversity / efficiency (`seconds_per_scene` 0.0218); `c1_hist_60` full report (`classical.status`: exit 0). `c1_hand_fx_60` SAM region covering 0.6487.
- Timing: `seconds_per_scene` `c1_hand_fx_60` 0.0961, `c1_reinhard_60` 0.0993, `c1_hist_fx_60` 0.1801, `c1_hist_60` 0.0872, `c1_hand_60` 0.0218.
- `c1_hist_60` / `c1_hand_60`: variants without the `_fx` post chain. Note: with no post chain, validity is 0.23 (`hist`) vs 0.32 (`hist_fx`), 0.18 (`hand`) vs 0.37 (`hand_fx`); the post chain raises the score (worse structure match) as expected.
- `c2_hand_s0.5_60`: `config.json` family C2, pre `hand`, strength 0.5, fusion "sum canny1.0+depth0.2", scale 0.8, schedule clean, seed 0, n 60; `classical.status` exit 0; log `logs/c2_hand_s0.5_60.log`. The log prints "Qwen25VLAnalyzer" for the failure step while the report lists `vikhyatk/moondream2` (same discrepancy as above, see Failure-model discrepancy). Smoke `c2smoke_hand` (n 3, same config) has no report. Report `seconds_per_scene` 1.4977.
- Superseded: `c3` (`c3_real_s60_60`, `c3_zgen_s60_60`) now have configs and reports (see sprint tables and "Seventh pass runs").

### Sprint runs with reports (n=60, seed 0, recorded 2026-10-18)
Sim-vs-real FID 133.35 in every report. Failure model in reports: `vikhyatk/moondream2`. Validity = `semantic_segmentation_score` (lower better). `real` = real photos 0-39 used at fit/train time per `config.json` (`real_used`); `n/a` where config has no such field.

| Run | Method (from `config.json`) | Real used | Failure | FID | KID | Validity | Hamming | s/scene |
|---|---|---|---|---|---|---|---|---|
| `c2_hand_s0.35_60` | C2 `hand` pre + `init_sim` strength 0.35, sum canny 1.0/depth 0.2, scale 0.8 | no | 0.75 | 143.09 | 0.1499 | 0.2885 | 0.2726 | 1.112 |
| `c2_reinhard_s0.5_60` | C2 `reinhard` pre + `init_sim` 0.5 | yes (0-39) | 0.6333 | 121.51 | 0.1021 | 0.3922 | 0.2522 | 1.567 |
| `m2_flux_i2i_s0.6_60` | FLUX img2img, strength 0.6, 4 steps | n/a | 0.9 | 139.66 | 0.1291 | 0.5273 | 0.2686 | 0.862 |
| `m2_flux_i2i_s0.8_60` | FLUX img2img, strength 0.8, 4 steps | n/a | 0.9333 | 289.73 | 0.3736 | 0.7978 | 0.2710 | 1.091 |
| `m3_kontext_60` | FLUX Kontext instruction edit, 20 steps, cfg 2.5 | n/a | 0.8333 | 140.64 | 0.1373 | 0.4740 | 0.2599 | 10.528 |
| `t1p_a_is0.5_60` | `trained_adapt` LoRA `t1p_a/lora_000439.pt` (step 439), canny scale 1.0, `init_sim` 0.5, prompt `label+style`, 8 steps | n/a | 0.65 | 149.40 | 0.1629 | 0.3665 | 0.2657 | 1.110 |
| `t1p_a_s1.0_60` | same LoRA, noise start | n/a | 0.8833 | 156.53 | 0.1607 | 0.6687 | 0.2763 (`runs.csv`) | 2.522 |
| `t0_noadapter_canny_60` | `trained_adapt` code path with `lora: none`, canny scale 1.0, noise start, prompt `label`, 8 steps, clean schedule (control for the LoRA runs; 2026-10-18) | no | 0.8167 | 257.72 | 0.3622 | 0.7612 | n/a (not extracted) | 2.486 |
| `z1_clean9_s1.0_60` | Z t2i, scale 1.0, 9 steps (per run name; config not read) | no | 0.8833 | 253.60 | 0.3533 | 0.7611 | 0.2745 (`runs.csv`) | 2.441 (`runs.csv`) |
| `z2_ir_t2i_60` | Z t2i, scale 1.0 (assumed as `z2_fisheye`), prompt `ir` (config not read) | no | 0.85 | 249.42 | 0.3436 | 0.7540 | 0.2790 (`runs.csv`) | 2.181 (`runs.csv`) |
| `m3_ip2p_60` (added 2026-10-18 sixth pass) | InstructPix2Pix (`kind` ip2p), 20 steps, cfg 7.5, image cfg 1.5, instruction "make this a real photograph taken inside a car" | n/a | 0.80 | 128.91 | 0.1209 | 0.2703 | 0.2681 | 0.767 |
| `m4_rv_i2i_s0.6_60` (added sixth pass) | SD1.5-family (`kind` sd15) img2img strength 0.6, 30 steps, cfg 6.0, ControlNet scale 0.6 (checkpoint n/a) | n/a | 0.85 | 229.96 | 0.2454 | 0.7072 | 0.2572 | 0.748 |
| `t2_a_s1.0_60` | `trained_adapt` LoRA `t2_a/lora_000300.pt` (step 300), canny 1.0, noise start, prompt `label` | n/a | 0.8833 | 137.20 | 0.1056 | 0.7290 | 0.2661 | 2.494 |
| `z1_clean_s0.6_60` | Z t2i, canny scale 0.6, 8 steps, `label` | no | 0.8667 | 309.09 | 0.4417 | 0.7999 | 0.2694 | 2.161 |
| `z1_clean_s1.0_60` | Z t2i, scale 1.0 | no | 0.8167 | 257.72 | 0.3590 | 0.7612 | 0.2686 | 2.169 |
| `z1_clean_s1.2_60` | Z t2i, scale 1.2 | no | 0.7333 | 251.90 | 0.3119 | 0.7688 | 0.2619 (`runs.csv`) | 2.164 |
| `z2_fisheye_t2i_60` | Z t2i, scale 1.0, prompt `fisheye` | no | 0.8833 | 265.55 | 0.3803 | 0.7549 | 0.2755 | 2.189 |
| `z3_s0.6_fisheye_60` | Z `sim` mode, s0 0.6, scale 0.8, prompt `fisheye` | no | 0.8167 | 113.24 | 0.1022 | 0.4355 | 0.2687 | 1.098 |

Hold-out runs (sprint; recorded 2026-10-18). Scenes 40-59, **n=20** (reports show `paired: 20`; `runs.csv` wrongly lists n=60 for the c1/c2 rows). Real fit/train photos were scenes 0-39 for the `_R` methods, so these scenes are outside the fit set. Reference in each report: sim-vs-real FID 148.85 (KID 0.1542) on these 20 scenes. FID/KID at n=20 are unreliable; failure step = 1/20 = 0.05. Values from `report.json` (failure, FID) and `runs.csv` (validity, KID, where marked):

| Run | Full n=60 counterpart (failure / FID / validity) | Failure (n=20) | FID | Validity | s/scene |
|---|---|---|---|---|---|
| `c1_hist_fx_60_hold20` | 0.7333 / 122.59 / 0.3242 | 0.70 | 134.08 | 0.3102 | 0.540 (`runs.csv`) |
| `c1_reinhard_60_hold20` | 0.75 / 140.69 / 0.1370 | 0.80 | 159.33 | 0.1338 | 0.298 (`runs.csv`) |
| `c2_reinhard_s0.5_60_hold20` | 0.6333 / 121.51 / 0.3922 | 0.60 | 139.79 | 0.3885 | 4.700 (`runs.csv`) |
| `t1p_a_is0.5_60_hold20` | 0.65 / 149.40 / 0.3665 | 0.70 | 161.92 (KID 0.1697) | 0.3563 | 1.110 |
| `t1p_a_s1.0_60_hold20` | 0.8833 / 156.53 / 0.6687 | 0.95 | 188.86 (KID 0.1829) | 0.6710 | 2.522 |
| `t2_a_s1.0_60_hold20` (added fifth pass; LoRA trained on real photos 0-39, so scenes 40-59 are out of sample) | 0.8833 / 137.20 / 0.7290 | 0.90 | 167.44 (KID 0.1225) | 0.7288 | 2.494 (copied from full run) |

For comparison on the same 20 scenes (earlier table): baseline canny s1.0 0.9 / 272.2 / 0.767; `is_s0.7_60` 0.75 / 145.0 / 0.544. The s/scene of the hold20 rows is copied from the full run's timing or the subset timing (`t1p_a_is0.5_60_hold20` `generation_times.json`: 22.19 s / 20 scenes = 1.110); for the c1/c2 rows it is from `runs.csv` and does not match the full-run values (e.g. c2_reinhard 4.70 vs 1.567), so treat as unreliable (cause not checked).

### Seventh pass runs (2026-10-18; n=60 unless noted, seed 0, sim-vs-real FID 133.35)
Params from `submissions/*/config.json`; all torch 2.10.0+cu128, failure reader `vikhyatk/moondream2` (reports). Prompt mode n/a for M/C3 rows (not in config).

| Run | Method / params | Real used | Failure | FID | KID | Validity | Hamming | s/scene |
|---|---|---|---|---|---|---|---|---|
| `m2_flux_i2i_s0.5_60` | FLUX img2img, strength 0.5, 4 steps | n/a | 0.75 | 131.96 | 0.1357 | 0.3552 | n/a | 0.635 |
| `m2_flux_i2i_s0.7_60` | FLUX img2img, strength 0.7, 4 steps | n/a | 0.90 | 139.66 | 0.1285 | 0.5273 | n/a | 0.873 |
| `m1_sdxl_canny_60` | SDXL t2i + canny 0.8, 30 steps, cfg 6.0 | n/a | 0.7833 | 307.35 | 0.3637 | 0.7579 | 0.2708 | 2.932 |
| `m1_sdxl_i2i_s0.6_60` | SDXL img2img strength 0.6 + canny 0.5, 30 steps, cfg 6.0 | n/a | 0.9333 | 209.12 | 0.2298 | 0.6621 | 0.2788 | 1.870 |
| `c3_real_s60_60` | C3 VGG16 feature-statistics optimisation, 60 steps, style = pooled real photos 0-39 (style_n 40) | yes (0-39) | 0.7167 | 147.35 | 0.1569 | 0.3366 | 0.2672 | 0.751 |
| `c3_zgen_s60_60` | C3, 60 steps, style = 12 Z-Image t2i renders (style_n 12) | no | 0.6833 | 144.60 | 0.1478 | 0.3405 | 0.2549 | 0.754 |
| `t3a_R_k1.0_60` | T1 trunk (`t1_trunk`), ckpt `/data/weissl/isu_ckpt/t3a_R/trunk_000700.pt`, knob 1.0, style suffix, 8 steps clean schedule. Per `runs.csv` note: smoke2 trunk continued 700 steps on 600 teacher outputs + real photos 0-39 x3 | yes (0-39, per note) | 0.8333 | 155.45 | 0.1423 | 0.6721 | 0.2851 | 3.169 |

Hold-out (scenes 40-59, n=20, sim-vs-real FID 148.85; FID/KID unreliable at n=20):

| Run | Failure | FID | KID | Validity |
|---|---|---|---|---|
| `t3a_R_k1.0_60_hold20` | 0.85 | 175.79 | 0.1390 | 0.6788 |
| `t0_noadapter_canny_60_hold20` (copy of 40-59 subset of `t0_noadapter_canny_60`) | 1.0 | 277.98 | 0.3815 | 0.7658 |

Caveats:
- `m2_flux_i2i_s0.7_60` has FID 139.6631 and validity 0.5273, identical to `m2_flux_i2i_s0.6_60` (139.6631 / 0.5273), failure 0.9 also identical; KID (0.1285 vs 0.1291) and s/scene (0.873 vs 0.862) differ. Identical deterministic metrics suggest the same generated images; not verified (images not diffed). Treat the s0.7 row as suspect.
- `runs.csv` has no row for `m2_flux_i2i_s0.5_60` / `s0.7_60`; values here are from `report.json`.
- C3 configs state `real_used` per run; Hamming for `c3_*` from `runs.csv`, validity from `sam_semantic_score`. Training script/loss details of C3 not read here.
- Smoke only, no report: `m5_sdxl_zimage_smoke3` (kind `cascade`, SDXL `m1_sdxl_canny` then Z-Image, strength 0.5, n 3), `m3_qwen_edit_smoke3` (kind `qwen`, 4 steps, cfg 1.0, LoRA true, same instruction as Kontext, n 3).
- `t3a_R` training data/recipe is from `runs.csv` notes, not from code.

### Eighth pass runs (2026-10-18; seed 0; sim-vs-real FID 133.35 at n=60, 148.85 at n=20)
Failure reader `vikhyatk/moondream2` (reports). Config fields from `submissions/*/config.json`.

| Run | n | Method / params | Real used | Failure | FID | KID | Validity | Hamming | s/scene |
|---|---|---|---|---|---|---|---|---|---|
| `t3b_noreal_k1.0_60` | 60 | `t1_trunk`, ckpt `/data/weissl/isu_ckpt/t3b_noreal/trunk_000700.pt`, knob 1.0, style suffix, 8 steps, clean schedule. Per `runs.csv` note: fresh trunk K=4, 768 px, 700 steps on 600 teacher targets only | no (per note) | 0.85 | 199.997 | 0.2339 | 0.6489 | 0.2702 | 3.144 |
| `t3b_noreal_k1.0_60_hold20` | 20 | scenes 40-59 of the above | no | 0.95 | 205.75 | 0.2242 | 0.6457 | 0.2030 | 3.144 (copied) |
| `c5_seat_recolor_60` | 60 | C5 `seat_recolor`: instance class `seat` recoloured (seeded hue, saturation 0.45, luminance kept), classical, no network (`classical/c5.py` docstring) | no (`real_used` false) | 0.7667 | 226.22 | n/a | 0.2048 | 0.2752 | 0.164 |
| `c5_babyseat_hue_60` | 60 | C5 `babyseat_hue`: class `baby_seat` hue-rotated 60-300 deg, no-op when absent | no | 0.85 | 139.06 | n/a | 0.0469 | 0.2741 | 0.149 |

- `c5_interior_dark_60` (class `car_interior` x0.6): submission with `config.json` only; report folder exists (SAM cache files, `logs/c5_interior_dark_eval.log`) but no `report.json` on 2026-10-24; no metrics (n/a). `classical/chain_c5.sh` runs the three variants.
- C5 config.json holds only name, n, seed, family, variant, `real_used`; the edit parameters are from the module docstring (code not read beyond the header). KID for C5 not extracted (n/a).
- T3b: t3b rows in `runs.csv` (FID, KID, validity, Hamming, s/scene) agree with the values above; checkpoint dropped per `research/disk_policy.md` (15 GB), so it cannot be rerun. Comparison: `t3a_R_k1.0_60` 0.8333 / 155.45 / 0.6721; hold-out 0.85 / 175.79 / 0.6788.
- Smoke only, no report (3 scenes): `m6_hybrid_flux_rv_smoke3`, `m7_flux_seat_s0.6_smoke3`, `m7_flux_seat_s0.7_smoke3`, `m7_kontext_seat_smoke3`, `m3_kontext_keep_smoke3`; configs not read (n/a).

### Ninth pass runs (2026-10-24; values from `report.json`; sim-vs-real FID 133.35 at n=60, 148.85 at n=20)
| Run | n | Method / params | Real used | Failure | FID | KID | Validity | s/scene |
|---|---|---|---|---|---|---|---|---|
| `m5_sdxl_zimage_60` | 60 | SDXL -> Z-Image cascade (smoke config: SDXL `m1_sdxl_canny` then Z-Image strength 0.5; 60-scene config not on disk) | n/a | n/a (no failure section) | 318.74 | 0.3774 | 0.7545 | n/a |
| `c3_real_s60_60_hold20` | 20 | scenes 40-59 of `c3_real_s60_60` (style fit on real 0-39) | yes (0-39) | 0.80 | 170.81 | 0.1763 | 0.3237 | 2.254 (report; differs from full run 0.751, cause not checked) |

- `m5_sdxl_zimage_60`: run log `logs/m_chainG.log`; FID is above every other noise-start run (249-309) and equals SDXL t2i-like regime (validity 0.75, `m1_sdxl_canny_60` 307.35 / 0.7579). Failure step not in report.
- `c3_real_s60_60_hold20`: PSNR 12.91, SSIM 0.2955 (report); `runs.csv` row agrees. Full run `c3_real_s60_60` 0.7167 / 147.35 / 0.3366; hold-out FID 170.8 is above sim-vs-real on the same 20 scenes (148.85), as is the n=60 FID (147.35 vs 133.35); no out-of-sample FID advantage over a sim copy.

### Z family, fourth pass (reports read 2026-10-18; n=60, seed 0, sim-vs-real FID 133.35)
Values from `report.json`; params from `submissions/*/config.json` (`zcfg`). Prompt `fisheye` in all. Semantics of `mode` values (`sim`, `inv`, `gate`, `pass2`), `ramp`, `mult` not read from code (n/a).

| Run | zcfg | Failure | FID | KID | Validity | Hamming | s/scene |
|---|---|---|---|---|---|---|---|
| `z3_s0.4_fisheye_60` | mode `sim`, s0 0.4, scale 0.8 | 0.8333 (fifth pass) | 125.44 | 0.1258 | 0.2900 | n/a | n/a |
| `z4_ramp_s0.6_60` | mode `sim`, s0 0.6, scale 1.2, ramp [0.2, 0.6] | 0.7667 | 112.63 | 0.0959 | 0.4496 | 0.2622 | 1.111 |
| `z5_gate_60` | mode `gate`, scale 0.8, `gate_table` (instance classes: persons 0.545, other labelled items 0.333) | 0.7167 | 127.10 | 0.1158 | 0.5152 | 0.2646 | 1.235 |
| `z6_inv_s0.6_60` | mode `inv`, s0 0.6, scale 0.8, `inv_prompt` `render` | 0.7333 | 192.60 | 0.2414 | 0.3602 | 0.2586 | 2.056 |
| `z7_up1.5_60` | mode `sim`, s0 0.6, scale 0.8, `mult` 1.5 | 0.7333 (fifth pass) | 116.53 | 0.1062 | 0.3611 | n/a | n/a |

Update fifth pass: failure now present for `z3_s0.4_fisheye_60` (0.8333) and `z7_up1.5_60` (0.7333, both from `report.json`; Hamming / s/scene from `runs.csv`: 0.2639 / 0.8239 s and 0.2747 / 2.765 s).

### Z family, fifth pass (n=60, seed 0, sim-vs-real FID 133.35; reports read 2026-10-18)
Config from `submissions/*/config.json` (`zcfg`); metrics from `runs.csv` rows, `z7_2pass_60` and `z6_inv_s0.75_60` cross-checked in `report.json`. Mode semantics (`sim`, `inv`, `pass2`, `ramp`, `mult`) not read from code (n/a).

| Run | zcfg | Failure | FID | KID | Validity | Hamming | s/scene |
|---|---|---|---|---|---|---|---|
| `z3_s0.6_label_60` | mode `sim`, s0 0.6, scale 0.8, prompt `label` | 0.80 | 115.51 | 0.1047 | 0.4307 | 0.2715 | 1.100 |
| `z4_ramp_s0.75_60` | mode `sim`, s0 0.75, scale 1.2, ramp [0.2, 0.75], `fisheye` | 0.85 | 142.08 | 0.1273 | 0.6332 | 0.2581 | 1.403 |
| `z6_inv_s0.75_60` | mode `inv`, s0 0.75, scale 0.8, inv_prompt `render`, `fisheye` | 0.70 | 231.84 | 0.3062 | 0.4196 | 0.2527 | 2.629 |
| `z7_up2_60` | mode `sim`, s0 0.6, scale 0.8, `mult` 2.0, `fisheye` | 0.7833 | 116.86 | 0.1097 | 0.3453 | 0.2710 | 5.765 |
| `z7_2pass_60` | mode `pass2`, s0 0.6, s1 0.4, m1 0.75, m2 1.5, scale 0.8, `fisheye` | 0.7833 | 134.71 | 0.1300 | 0.5034 | 0.2752 | 2.748 |

Reference: `z3_s0.6_fisheye_60` 0.8167 / 113.24 / 0.4355; `s0_c06_k0_60` 0.7667 / 114.36 / 0.3849. All 8 Z-family configs with runs now have reports. Validity column here is `sam_semantic_score` in `runs.csv` (= `semantic_segmentation_score`; z7_2pass 0.5034 confirmed in report). s/scene of `z7_up*` and `z6_inv` is higher than plain sim mode; cause (extra passes vs shared GPU) not checked.

(History) Config-only note, superseded: `z3_s0.6_label_60` ... `z7_2pass_60` had configs only in the fourth pass. Reference same family: `z3_s0.6_fisheye_60` 0.8167 / 113.24 / 0.4355; `s0_c06_k0_60` 0.7667 / 114.36 / 0.3849.

Notes: `z6_inv_s0.6_60` and `z7_up1.5_60` previously had no report (superseded by the table above). The `fisheye` / `ir` prompt texts are in `solution/zimage_tune/prompts_z.py` ("A frame from a wide-angle fisheye in-cabin driver-monitoring camera inside a car." plus label text; remainder not read). `z*` configs have no `real_used` field; the `fisheye` prompt was written after looking at `data/real` (comment in `prompts_z.py`: real photos are a ceiling-mounted wide-angle fisheye in-cabin camera), so real photos informed the prompt (no pixel use). `run_z.py` defines further variants (`z3_s0.4`, `z4_ramp`, `z5_gate`, `z6_inv`, `z7_up`, `z7_2pass`); as of the fourth pass 5 have reports (table above), `z7_2pass` has a config only. Configs of the `t1p_*` / `t2_*` LoRA runs name the training set only by checkpoint path (`/data/weissl/isu_ckpt/...`). From `research/runs.csv` notes (not from the code): `t1p_a_*` = T1' LoRA (rank 16, all attention + FF, 439 steps) trained on **teacher s0.6 targets, no real photos** (teacher = Z-Image-Turbo img2img); `t2_a_s1.0_60` = T2_R LoRA rank 16, 300 steps, **real photos 0-39 as targets** (scenes 0-39 are train, so its n=60 score is leaky; no hold20 report). `t0_noadapter_canny_60` equals `z1_clean_s1.0_60` on failure/FID/validity (0.8167 / 257.72 / 0.7612), so the `trained_adapt` code path without LoRA reproduces the plain noise-start run.

S family (`solution/surrogate/`, all n=60, scale 0.8, seed 0, policy as named; failure/FID/KID/validity/Hamming from `research/runs.csv`):

| Run | Pool / policy | Control | Init | Prompt | Failure | FID | KID | Validity | Hamming | s/scene |
|---|---|---|---|---|---|---|---|---|---|---|
| `s_t2i_label_60` | `t2i_label` k0 | canny | t2i | label | 0.8667 | 266.70 | 0.3758 | 0.7589 | 0.2655 | 2.458 |
| `s_t2i_cap_60` | `t2i_cap` k0 | canny | t2i | cap | 0.9 | 264.91 | 0.3746 | 0.7511 | 0.2636 | 2.527 |
| `s_hed_t2i_60` | `hed_t2i` k0 | hed | t2i | label | 0.6667 | 151.35 | 0.1475 | 0.4246 | 0.2593 | 2.466 |
| `s_da2_t2i_60` | `da2_t2i` k0 | da2 | t2i | label | 0.9167 | 214.72 | 0.2232 | 0.7190 | 0.2687 | 2.464 |
| `s0_c06_k0_60` | `c06` k0 | canny | img2i 0.6 | label | 0.7667 | 114.36 | 0.1014 | 0.3849 | 0.2745 | 1.097 |
| `s_c06_cap_60` | `c06_cap` k0 | canny | img2i 0.6 | cap | 0.7667 | 114.25 | 0.1036 | 0.3840 | 0.2582 | 1.125 |
| `s_hed_06_60` | `hed_06` k0 | hed | img2i 0.6 | label | 0.7 | 124.58 | 0.1230 | 0.2781 | 0.2701 | 1.100 |
| `s_da2_06_60` | `da2_06` k0 | da2 | img2i 0.6 | label | 0.7167 | 125.54 | 0.1239 | 0.3619 | 0.2581 | 1.095 |
| `s_c06_vlm_60` | `c06` K=3, VLM-agreement selection | canny | img2i 0.6 | label | 0.7833 | 113.91 | 0.1015 | 0.3745 | 0.2822 | 11.349 |
| `s_c06_vlm_adv_60` | `c06`, VLM adversarial selection (name only) | canny | img2i 0.6 | label | 0.7333 | 112.40 | 0.0993 | 0.3760 | 0.2546 | 11.349 |
| `s_c06_clip_60` | `c06`, CLIP selection | canny | img2i 0.6 | label | 0.75 | 111.74 | 0.0981 | 0.3789 | 0.2714 | 3.313 |
| `s_c06_sam_60` | `c06`, SAM selection | canny | img2i 0.6 | label | 0.7333 | 114.91 | 0.1036 | 0.3771 | 0.2701 | 4.387 |
| `s_c06_combo_60` | `c06`, combined selection | canny | img2i 0.6 | label | 0.7667 | 114.03 | 0.1013 | 0.3751 | 0.2661 | 4.410 |

`s_t2i_label_60` reproduces `base_canny_s0.8_label` exactly on failure, FID (266.704), validity and Hamming (same images presumably; KID differs in the 3rd digit: 0.3758 vs 0.3704). s/scene of selection runs includes candidate scoring (modelled, per `generation_times.json`); selection-policy definitions (`score_pool.py`, `assemble.py`) were not read here, so the policy descriptions above come from run names only.

### Family S surrogate-pool submissions, config details (recorded 2026-10-18; reports now exist, see tables above)
`config.json`: method `surrogate`, policy `k0` (first candidate, K=1, so no selection happens), seed 0, scale 0.8, clean schedule, canny/hed/da2 control as listed, captions cache `solution/surrogate/cache/captions.json`. Code `solution/surrogate/{gen_pool,score_pool,assemble,caption,make_controls}.py`; `common.py` states no real photos are used. `generation_times.json` note: "modelled from measured generation + scorer seconds"; s/scene below = `total_seconds` / 60 (derived). All 60 `generated/` images exist for `s_hed_06_60` (others not counted). Results: see "Sprint runs with reports" (eight pool runs had no report at the first check, reports arrived later the same day).

| Run | Pool | Control | Init | Prompt | total s (60 scenes) | s/scene (derived) |
|---|---|---|---|---|---|---|
| `s_t2i_label_60` | `t2i_label` | canny | text-to-image (strength "none") | label | 147.48 | 2.458 |
| `s_t2i_cap_60` | `t2i_cap` | canny | t2i | `cap` (cached captions) | 151.63 | 2.527 |
| `s_hed_t2i_60` | `hed_t2i` | hed | t2i | label | 147.95 | 2.466 |
| `s_da2_t2i_60` | `da2_t2i` | da2 (Depth-Anything-2 cache) | t2i | label | 147.83 | 2.464 |
| `s0_c06_k0_60` | `c06` (pool K=3, policy K=1) | canny | img2i strength 0.6 | label | 65.80 | 1.097 |
| `s_c06_cap_60` | `c06_cap` | canny | img2i 0.6 | `cap` | 67.49 | 1.125 |
| `s_hed_06_60` | `hed_06` | hed | img2i 0.6 | label | 66.01 | 1.100 |
| `s_da2_06_60` | `da2_06` | da2 | img2i 0.6 | label | 65.67 | 1.095 |

Whether the img2i strength 0.6 pools use the `init_sim` path or another loader, the caption model, and the hed/da2 extractors are not recorded here (n/a; not read). `logs/s_chain1.log` shows a VLM-agreement scoring loop (`sample_00000_k0 8 / 11`, K=3 candidates) for the `c06` pool; no selection-policy run is on disk.

### Smoke submissions without reports (2026-10-18)
- Note sixth pass: `m3_ip2p_smoke3` and `m4_rv_i2i_s0.6_smoke3` now have n=60 counterparts with reports (`m3_ip2p_60`, `m4_rv_i2i_s0.6_60`, same params, see "Sprint runs with reports"); other smokes below remain without report.
- Added after the previous update (3 scenes, seed 0, torch 2.10.0+cu128, no report): `m2_flux_i2i_s0.6_smoke3` (FLUX img2img strength 0.6, 4 steps), `m3_ip2p_smoke3` (InstructPix2Pix, 20 steps, cfg 7.5, image cfg 1.5, "make this a real photograph taken inside a car"), `m3_kontext_smoke3` (FLUX Kontext, 20 steps, cfg 2.5, instruction "Turn this 3D render into a real photograph taken by an in-car camera. Keep the exact same scene, people, objects and layout."), `m4_rv_i2i_s0.6_smoke3` (SD1.5-family `sd15` kind, i2i strength 0.6, 30 steps, cfg 6.0, ControlNet scale 0.6). Checkpoint identifiers n/a.
- `m1_sdxl_canny_smoke3`: method `m1_sdxl_canny`, SDXL t2i, canny 0.8, 30 steps, cfg 6.0, n 3, seed 0, torch 2.10.0+cu128. `m1_sdxl_i2i_s0.6_smoke3`: SDXL i2i strength 0.6, canny 0.5, 30 steps, cfg 6.0, n 3. `smoke_t2_3`: method `trained_adapt`, LoRA `/data/weissl/isu_ckpt/smoke_t2/lora_000005.pt` (step 5), scale 1.0, canny, label prompt, 8 steps, clean schedule, n 3. No report.json; no metrics (n/a).

### 2026-10-18 sprint code without reports (smoke only)
- Plan: `research/experiments_plan.md` (families Z, M, C, S, T; run naming `<expid>_<variant>_60`; every GPU job through `tools/gpu_run.sh`; real-photo ideas are `_R` variants fit on scenes 0-39). Code present: `zimage_tune/`, `other_models/` (sd15, sdxl, flux, ip2p, cascade), `surrogate/`, `trained_adapt/`, `exp_common.py`. Logs only (`logs/m_*`, `s_pool_smoke.log`, `trained_*`); no `submissions/` folder, no report, no result. `m_smoke_sdxl2.log` ends with `FileNotFoundError` (missing HF cache ref for `madebyollin/sdxl-vae-fp16-fix`). `checkpoints/` holds only `t1_smoke2` (the `t1_smoke` checkpoint described below is no longer on disk).

### Proxy metrics (`solution/analysis/fast_proxy.py`)
- `run_proxy_all.sh` writes `analysis/proxy_out/{run}.json`; present at 17:57: `base_canny_s0.8_label`, `base_canny_s1.0_label`. Values not recorded here (n/a).

### base_depth_s1.0_label, ref_real_*, ref_sim_*
- `base_depth_s1.0_label`: generation 148.80 s; report.json rewritten 18:13 with failure 0.85, Hamming 0.2778, CLIP cos 0.0793.
- `ref_sim_noise3_s1`: sim + noise sigma 3 reaches failure 0.7833 (47/60), same as best method runs, FID 129.3 (below sim-vs-real 133.3), validity 0.193. Seed-0 variant (`ref_sim_noise3_s0`): failure 0.7333, FID 129.2, validity 0.199. Evaluator is gameable by near-copies of sim; see methodology Findings.
- `ref_real_rs` evaluated with the full evaluator: real photos obtain failure rate 0.667 (40/60) with the moondream2 reader, i.e. the failure metric has a floor far above 0 on this data; real-vs-sim validity 0.702. The same photos JPEG q30 (`ref_real_jpeg`) give 0.583 (35/60), so the reader is sensitive to compression (5 scenes).
- `ref_shuf_real` (wrong scene's real photo): failure 0.933 (56/60), validity 0.718; shows failure metric does detect label mismatch (0.667 -> 0.933).
- `ref_sim_noise30`: failure 0.917, FID 208, validity 0.449; `ref_sim_jitter` failure 0.833, FID 131.9, validity 0.166.
