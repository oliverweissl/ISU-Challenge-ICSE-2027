# Measurement hygiene (experiment-runner, 2026-10-17)

## Seed SD: baseline `--control canny --scale 1.0 --prompt label`, n=60
Seeds 0 / 1000 / 2000 (per-scene seed = base seed + index). Runs: base_canny_s1.0_label, _seed1, _seed2. SD is the sample SD (n-1) over 3 runs.

| metric | seed0 | seed1000 | seed2000 | mean | SD |
|---|---|---|---|---|---|
| failure_rate | 0.7833 | 0.8333 | 0.8833 | 0.8333 | 0.0500 |
| SAM score (lower better) | 0.7612 | 0.7573 | 0.7656 | 0.7614 | 0.0041 |
| FID | 254.26 | 264.29 | 255.71 | 258.09 | 5.42 |
| KID | 0.3337 | 0.3553 | 0.3455 | 0.3448 | 0.0108 |
| VGG | 0.9295 | 0.9321 | 0.9048 | 0.9222 | 0.0151 |
| SSIM | 0.3490 | 0.3508 | 0.3573 | 0.3523 | 0.0044 |
| feature_hamming | 0.2611 | 0.2738 | 0.2847 | 0.2732 | 0.0118 |
| clip_div | 0.1514 | 0.1496 | 0.1530 | 0.1513 | 0.0017 |

Reading: failure_rate seed SD 0.05 (range 0.78-0.88) swamps every failure difference between existing runs (0.78-0.87). FID SD ~5 (range 10): canny-vs-fusion FID gaps (<=2) are noise; depth (447) and sim-copy (130) gaps are real. SAM SD 0.004: SAM differences >~0.01 are real (fusion 0.755-0.757 vs 0.761 is noise; depth 0.59 and hf_sum_w83 0.737 are real).

## Per-feature agreement, paired per-scene bootstrap 95% CI
Script: `track_a/solution/analysis/per_feature.py RUN_A RUN_B` (10000 resamples of scenes, same scene indices for both runs; diff = B - A, agreement of generated vs real-photo answer from `final.failure.images[*].feature_matches`; * = CI excludes 0). Seed-0 runs only, so seed variance is not included in the CI; for scale see base vs seed1 (all CIs include 0).

### base_canny_s1.0_label (A) vs hf_sum_60 (B)
| feature | A agree | B agree | diff | 95% CI | sig |
|---|---|---|---|---|---|
| baby | 0.517 | 0.567 | +0.050 | [-0.067, +0.167] |  |
| baby_seat | 0.717 | 0.700 | -0.017 | [-0.100, +0.067] |  |
| colabottle_codriver_seat | 1.000 | 0.950 | -0.050 | [-0.117, +0.000] |  |
| colacan_codriver_seat | 1.000 | 0.950 | -0.050 | [-0.117, +0.000] |  |
| driver_phone | 0.733 | 0.717 | -0.017 | [-0.100, +0.067] |  |
| driver_safety_belt | 0.783 | 0.750 | -0.033 | [-0.150, +0.083] |  |
| passenger_codriver | 1.000 | 0.950 | -0.050 | [-0.117, +0.000] |  |
| passenger_rear_seat_left | 1.000 | 0.950 | -0.050 | [-0.117, +0.000] |  |
| passenger_rear_seat_right | 0.983 | 0.933 | -0.050 | [-0.133, +0.017] |  |
| phone_codriver_seat | 1.000 | 0.950 | -0.050 | [-0.117, +0.000] |  |
| suitcase | 0.900 | 0.850 | -0.050 | [-0.117, +0.000] |  |
| ALL_MATCH(1-failure) | 0.250 | 0.250 | +0.000 | [-0.100, +0.100] |  |
| MEAN_FEATURE | 0.876 | 0.842 | -0.033 | [-0.086, +0.009] |  |

### base_canny_s1.0_label (A) vs base_depth_s0.8_label (B)
| feature | A agree | B agree | diff | 95% CI | sig |
|---|---|---|---|---|---|
| baby | 0.517 | 0.400 | -0.117 | [-0.283, +0.050] |  |
| baby_seat | 0.717 | 0.650 | -0.067 | [-0.217, +0.083] |  |
| colabottle_codriver_seat | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] |  |
| colacan_codriver_seat | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] |  |
| driver_phone | 0.733 | 0.750 | +0.017 | [-0.083, +0.133] |  |
| driver_safety_belt | 0.783 | 0.633 | -0.150 | [-0.317, +0.017] |  |
| passenger_codriver | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] |  |
| passenger_rear_seat_left | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] |  |
| passenger_rear_seat_right | 0.983 | 1.000 | +0.017 | [+0.000, +0.050] |  |
| phone_codriver_seat | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] |  |
| suitcase | 0.900 | 0.967 | +0.067 | [+0.017, +0.133] | * |
| ALL_MATCH(1-failure) | 0.250 | 0.150 | -0.100 | [-0.233, +0.033] |  |
| MEAN_FEATURE | 0.876 | 0.855 | -0.021 | [-0.056, +0.012] |  |


Findings: no feature differs for fusion (hf_sum equals canny; every CI includes 0; the 5-point drops on 7 features are the same 3 scenes, i.e. Qwen answer flips, not a trend). Depth s0.8 vs canny: only `suitcase` is significantly better (+0.067); driver_safety_belt (-0.150) and baby (-0.117) trend worse but CI includes 0; ALL_MATCH 0.15 vs 0.25. Five features (colabottle, colacan, passenger_codriver, rear seats, phone_codriver) are saturated at ~1.0 for all runs; only baby, baby_seat, driver_phone, driver_safety_belt (0.52-0.78) carry signal. Note: min over the 11 `feature_matches` gives 0.250 all-match vs 1-failure_rate = 0.217, so `failure` also includes non-listed fields (expected has 14 keys incl. suitcase_location/pose, baby_seat_orientation); the per-feature table omits those.

## Other
- make_sheet.py now pairs real refs by stem (`data/real/<stem>_real.jpg`); batch1.png regenerated (same scenes, seed 0). The shuffle order changed, so cand C/D swapped: new key is C=hf_sum_60, D=hf_region_60 (A, B unchanged); batch1.key.json overwritten to match the new image.
- Eval OOM: the generating process holds ~28 GB while its child evaluator runs, so seed runs' in-process eval crashed; evals re-run via tools/eval_queue.sh (lines appended to chain_a.status).
