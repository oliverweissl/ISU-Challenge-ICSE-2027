# Track A evaluator analysis (evaluator-analyst)

All numbers come from real runs of `track_a/evaluator_cli.py` (`--failure-model qwen --device cuda`, n=60 scenes with real refs),
reports in `track_a/track_a_evaluation_output/ref_*/report.json`. Submissions built by
`track_a/solution/analysis/build_ref_submissions.py` (12 evaluator runs, `ref_*`). Aggregation: `analysis/summarize_refs.py`.
Seed-noise rows additionally use the baseline-grid runs `base_canny_s1.0_label{,_seed1,_seed2}` (same config, 3 generator seeds) run by others.
Not edited: evaluator files, data/.

Directions: SAM score lower = structure preserved (1 - matched IoU of SAM partitions, sim vs generated); failure higher = better for the
challenge; FID/KID lower; SSIM/LPIPS-ish are vs the real reference.

## 0. Evaluator facts verified in code
- `track_a_metrics.py:349` hardcodes `"model": "vikhyatk/moondream2"` in the failure report. It is only a label. The `revision` field of the
  report says `main`, which is set only by `Qwen25VLAnalyzer` (Moondream would say 2025-06-21), and per-image Qwen answers are present. Qwen2.5-VL-3B really ran.
- Failure = generated and real ref give different Qwen answers for ANY of ~14 features (`public_features` after dropping *color/emotion/gender/env/light* etc.).
  It is a real-vs-generated agreement test, not a label test. Qwen is greedy, so identical pixels give identical answers (the evaluator also short-circuits identical pixels).
- SAM validity compares sim vs generated only (real unused). Sim partitions are cached by pixel hash, so they are the same across runs.
- FID/KID/IS use only the 60 refs vs the 60 generated images (FID with n=60 is very noisy in absolute terms but comparable across runs).
- Generated images are resized to sim size (960x540) by the user; refs are 1920x1080 and are resized to the generated size for paired metrics.

## 1. Ceilings and floors (n=60)

| submission | SAM sem. | failure | FID | KID | IS | SSIM v real | LPIPS-ish (VGG) | feat. hamming | CLIP div |
|---|---|---|---|---|---|---|---|---|---|
| identity sim (generated = sim) | 0.000 (trivial) | 0.800 | 133.3 | 0.138 | 1.26 | 0.456 | 0.791 | 0.278 | 0.085 |
| real ref resized to 960x540 | 0.702 | 0.667 | 0.42 | -0.005 | 1.37 | 0.996 | 0.005 | 0.288 | 0.114 |
| real ref JPEG q30 full-res | 0.713 | 0.583 | 9.9 | -0.000 | 1.39 | 0.935 | 0.041 | 0.274 | 0.124 |
| real of the NEXT scene (shuffled) | 0.718 | 0.933 | 0.42 | -0.005 | 1.39 | 0.740 | 0.449 | 0.274 | 0.115 |
| sim + gaussian noise sigma=3 (seed 0 / seed 1) | 0.199 / 0.193 | 0.733 / 0.783 | 129.2 / 129.3 | 0.132 | 1.29 | 0.410 | 0.775 | 0.268 / 0.284 | 0.088 |
| sim colour jitter | 0.166 | 0.833 | 131.9 | 0.133 | 1.29 | 0.434 | 0.808 | 0.288 | 0.107 |
| sim grayscale | 0.350 | 0.750 | 210.6 | 0.284 | 1.34 | 0.458 | 0.742 | 0.293 | 0.077 |
| sim gaussian blur r=6 | 0.452 | 0.967 | 328.1 | 0.492 | 1.43 | 0.547 | 0.519 | 0.272 | 0.058 |
| sim noise sigma=30 | 0.449 | 0.917 | 208.0 | 0.271 | 1.47 | 0.064 | 0.603 | 0.281 | 0.102 |
| 50% sim + 50% real blend | 0.576 | 0.700 | 98.2 | 0.085 | 1.31 | 0.774 | 0.478 | 0.275 | 0.104 |
| black image | 0.621 | 1.000 | 738.4 | 1.238 | 1.00 | 0.004 | 0.485 | 0.281 | 0.000 |
| baseline canny s0.8 | 0.759 | 0.867 | 266.7 | 0.370 | 1.48 | 0.316 | 0.955 | 0.266 | 0.142 |
| baseline canny s1.0 (3 seeds, mean) | 0.761 | 0.833 | 258.1 | 0.345 | 1.41 | 0.352 | 0.922 | 0.273 | 0.151 |
| img2img strength 0.5 (others' run `is_s0.5_60`) | 0.382 | 0.783 | 116.0 | 0.106 | 1.35 | 0.463 | 0.700 | 0.262 | 0.105 |

What good looks like and which axes conflict:
- The SAM score is nearly saturated: any image that is not pixel-aligned with the sim sits at 0.62-0.76 (black 0.62, real unrelated photo 0.70-0.72, canny baseline 0.76).
  The baseline is no better than an unrelated real photograph on "validity". The usable range is about 0.19 (pixel-perfect-ish) to 0.76.
- SAM is discontinuous at zero: invisible noise (sigma=3) already costs +0.19; a global colour jitter costs 0.17. So the practical SAM floor for
  any generated image with different pixels is about 0.17-0.20, and SAM penalises texture/speckle independently of structure.
- Failure is the cheapest axis. Doing nothing (identity sim) gives 0.80, which equals the baseline grid (0.78-0.88). The floor for real-vs-real is 0.58-0.67 (see section 2),
  the practical ceiling is about 0.97.
- The baseline is WORSE than the unmodified simulator on realism: FID 258-267 vs 133 for identity sim, KID 0.345-0.37 vs 0.138, SSIM/VGG similar or worse. The only metrics where
  the baseline beats identity sim are CLIP diversity (0.14-0.15 vs 0.085) and failure (not by a significant margin). The bar to beat is "identity sim" on FID/KID, and
  img2img from the sim (strength 0.5: FID 116, KID 0.106, failure 0.78, SAM 0.38) already beats it on FID/KID while keeping failure at baseline level.
- Axes in conflict: structure fidelity (SAM) vs realism/FID (more freedom lowers FID in the strength sweep only until the generator drifts; pure text-to-image from canny is at 255+);
  CLIP diversity vs SAM (more change = more diversity); failure vs realism for degradations (blur/noise/black raise failure and destroy FID).

## 2. Failure definition
Qwen disagrees with itself: same real image resized to 960x540 gives mismatch in **40/60 scenes (0.667)**; JPEG q30 re-encode gives **35/60 (0.583)**.
Mean mismatching features per scene: real resized 1.60, real JPEG 1.03, baseline canny 2.33. So the "failure rate" floor of an essentially perfect copy of the real image is
0.58-0.67 and the baseline sits at 0.78-0.88. The real-signal headroom is about 0.15-0.30 failure points, not 0-1.

Per-feature mismatch counts out of 60 (generated vs real answer):

| feature | real resized | real JPEG | identity sim | sim sigma=3 | canny s0.8 | canny s1.0 | blur | black | shuffled real |
|---|---|---|---|---|---|---|---|---|---|
| baby (answer often a list string like "['UNKNOWN','YES']") | 21 | 18 | 30 | 29/30 | 31 | 29 | 42 | 55 | 45 |
| baby_seat | 13 | 9 | 16 | 17/18 | 20 | 17 | 21 | 21 | 31 |
| baby_seat_orientation | 18 | 15 | 21 | 25/22 | 23 | 20 | 16 | 16 | 23 |
| driver_phone | 8 | 4 | 14 | 12 | 20 | 16 | 25 | 15 | 26 |
| driver_safety_belt | 9 | 9 | 12 | 9/11 | 22 | 13 | 36 | 37 | 23 |
| suitcase | 4 | 2 | 7 | 7 | 4 | 6 | 2 | 2 | 4 |
| suitcase_location | 7 | 2 | 7 | 7 | 10 | 12 | 6 | 6 | 11 |
| suitcase_pose | 10 | 3 | 7 | 8/7 | 10 | 12 | 6 | 6 | 12 |
| passenger_codriver, rear seats L/R, phone_codriver_seat, cola bottle/can | 1 each | 0 | 3 each | 3 each | 0 | 0-1 | 0 | 0 | 1 |

- Failures are dominated by `baby`, `baby_seat`, `baby_seat_orientation` (about 50-60% of all mismatches), then driver_phone, driver_safety_belt, suitcase_pose/location.
  These are tiny/hard objects Qwen-3B barely sees; Qwen itself is only 36/60 correct for baby_seat vs the GT label on real images, 36/60 for driver_phone, 10/60 for `baby` (the field returns lists).
  The `baby` feature is effectively a coin flip on any image pair, so it contributes failures that no method can control.
- Seven passenger/object features (passenger_codriver, rear seats, phone/cola on codriver seat) are almost never mismatched (0-1 for baseline/real); they only flip when the generator
  hallucinates a passenger/object (the identity sim and 50% blend flip them in 3/60 scenes; the hint-fusion runs 3/60 too). They are cheap signal, not noise.
- driver_safety_belt: baseline says "YES" where real says "NO" 20/60 times at s0.8 (13 at s1.0): the generator tends to draw a belt (prompt-driven), a controllable systematic bias
  and the largest baseline-specific excess over the real-vs-real floor (22 vs 9). driver_phone (20 vs 8) is the second.
- A high failure rate is reachable at good validity: the identity sim has 0.80 at SAM 0, sigma=3 noise has 0.73-0.78 at SAM 0.19, colour jitter 0.83 at SAM 0.17.
  Image properties that raise it: global sharpness/pixel-statistics shifts (blur 0.97, noise30 0.92, darkening/black 1.0), any change that makes Qwen's low-confidence features flip.
  Properties that lower it: realistic photographic content close to the real ref (real-vs-real 0.58-0.67, blend 0.70).
  Much of failure is therefore "domain gap in Qwen's eyes", not semantic change.

## 3. Gaming risks (cheap tricks vs. what catches them)
| trick | failure | SAM | FID/KID | caught by |
|---|---|---|---|---|
| black image | 1.00 | 0.62 | 738 / 1.24 | FID/KID, SSIM, IS=1.0, CLIP diversity=0. SAM does NOT catch it (0.62 beats the baseline's 0.76) |
| blur r=6 | 0.97 | 0.45 | 328 / 0.49 | FID/KID, CLIP div collapses (0.058) |
| noise sigma=30 | 0.92 | 0.45 | 208 / 0.27 | SSIM 0.06, LBP 0.50, GLCM 160, FID |
| grayscale | 0.75 | 0.35 | 211 / 0.28 | FID, histogram metrics; failure not raised |
| use the real refs (shuffled real) | 0.93 | 0.72 | 0.42 / -0.005 | NOTHING in the evaluator: FID/KID are computed on the same 60 images, so using or training on `data/real/` makes realism perfect and SAM better than the baseline. Must never be used; do not condition on or fine-tune with the real refs, and keep LoRA/data free of them |
| 50% blend with real | 0.70 | 0.58 | 98 / 0.085 | not caught (lower FID than any honest method), same warning |
- SAM validity can be gamed by "doing less": identity/colour jitter gives 0.0-0.17, and conversely it is blind to the black image. Always report SAM together with FID/KID and CLIP diversity.
- Failure rate rewards degradation: never accept a failure gain without FID/KID not rising. Failure increments of <0.14 are noise (section 4).
- The feature-hamming diversity is nearly constant (0.246-0.293 across everything incl. identity sim, black, real) and carries no information.
- CLIP visual diversity is computed only over *failing* scenes and is sensitive to image variety (blur, black, gray collapse it; baseline highest 0.14-0.17).

## 4. Seed noise and minimum meaningful deltas
Two noise sources: (a) the metrics themselves on a deterministic perturbation (sigma=3 noise seed 0 vs 1, same sim): FID 129.2/129.3, KID 0.1319/0.1320, SSIM identical
(1e-5), SAM 0.1987/0.1932, failure 0.733/0.783 (3 scenes flip), feature hamming 0.268/0.284. So all pixel-based metrics are deterministic;
SAM moves 0.006 and Qwen failure 0.05 from invisible pixel noise. (b) generator seed noise: `base_canny_s1.0` seeds 0/1/2 (n=60 each):

| metric | mean over 3 seeds | sd | max-min | recommended min meaningful delta (about 2.8 sd, two runs) |
|---|---|---|---|---|
| SAM semantic | 0.7614 | 0.0041 | 0.008 | 0.012 |
| failure rate | 0.833 | 0.050 | 0.100 | 0.14 (about 8 of 60 scenes); use mean-mismatch-count or >=3 seeds for smaller effects |
| FID | 258.1 | 5.4 | 10 | 15 |
| KID | 0.345 | 0.011 | 0.022 | 0.03 |
| IS | 1.41 | 0.057 | 0.11 | 0.16 (do not use IS, near 1.3-1.5 for everything) |
| SSIM v real | 0.352 | 0.004 | 0.008 | 0.012 |
| VGG perceptual dist. | 0.922 | 0.015 | 0.027 | 0.04 |
| GLCM texture dist. | 9.3 | 2.2 | 4.0 | 6 (very noisy) |
| LBP similarity | 0.877 | 0.0014 | 0.003 | 0.004 |
| feature hamming | 0.273 | 0.012 | 0.024 | 0.033 |
| CLIP diversity | 0.151 | 0.0017 | 0.003 | 0.005 |
A binomial check agrees for failure: sd at p=0.8, n=60 is 0.052. With n=3 seeds the standard error of the mean failure is about 0.03, so deltas below 0.06 never matter.
Only 3 seeds exist, so these are rough; the FID/KID values are n=60 FID (very small sample), whose scale also shifts between scales/prompts, not only seeds.
Note that SAM also saturates (section 1), so a SAM delta of 0.012 matters only below about 0.7.

## 5. Fast proxy
`track_a/solution/analysis/fast_proxy.py <submission_dir> [--out f.json]` computes in about 35-65 s per 60-scene submission on one GPU (first run builds the sim cache, +about 60 s):
- `sam_proxy`: 1 - matched IoU of SAM partitions (coarser grid `points_per_crop=12`, 480x270 inputs, sim partitions cached in `analysis/cache/`).
- `kid_r50` (unbiased polynomial-kernel MMD on ResNet50 features of generated vs real refs), `fd_diag` (diagonal Frechet), `col_emd` (RGB histogram Wasserstein),
  `hf_ratio` (gradient energy ratio vs real), `fail_proxy` (1 - cosine of ResNet50 embeddings of generated vs its real ref).
Rank correlation (Spearman) with the full evaluator on 25 evaluated runs (my 12 plus others' runs; `analysis/proxy_corr.py`):

| proxy | full metric | all runs (n=25) | generator runs only (n=13) |
|---|---|---|---|
| sam_proxy | SAM semantic | 0.98 | 0.88 |
| kid_r50 | KID | 0.73 | 0.85 |
| kid_r50 | FID | 0.65 | 0.63 |
| fd_diag | FID | 0.79 | 0.72 |
| col_emd | wasserstein | 0.98 | 0.99 |
| fail_proxy | failure rate | 0.55 | 0.03 (useless) |
Use SAM proxy for validity, `fd_diag`/`kid_r50` as a coarse realism screen (do not rank within +-15%); there is no VLM-free proxy for the failure rate, which requires Qwen (about 5-9 s per image pair of Qwen, 5-8 min per 60 scenes when the GPU is free, 15-20 min when shared).
Helper scripts: `build_ref_submissions.py`, `run_refs.sh`, `summarize_refs.py`, `run_proxy_all.sh`, `proxy_corr.py` (all in `track_a/solution/analysis/`).
Note: `per_feature.py` in that folder was not written by me (another agent).

## 6. The 10 most decision-relevant findings
1. The unmodified simulator beats the Z-Image canny baseline on realism: FID 133 vs 258-267, KID 0.138 vs 0.345-0.37, and failure is equal (0.80 vs 0.78-0.88). The target is identity-sim FID/KID.
2. Image-to-image from the sim (strength 0.5) gives FID 116, KID 0.106 and failure 0.78 at SAM 0.38: a better trade than the text-to-image baseline. Starting from the sim image is the high-value direction.
3. Failure rate has little dynamic range: real-vs-real is 0.58-0.67, identity sim 0.80, baseline 0.78-0.88, degradations up to 0.97. Differences below 0.14 are noise.
4. Qwen disagrees with itself on resize/JPEG in 58-67% of scenes; baby/baby_seat/baby_seat_orientation cause most mismatches and are uncontrollable noise.
5. Controllable failure drivers: driver_safety_belt (generator adds belts) and driver_phone (baseline excess 22 vs 9, 20 vs 8 over the real floor), plus hallucinated passengers/objects (the 7 near-zero features).
6. SAM validity saturates at 0.70-0.76 for any unaligned content (even a real photo or black image); baseline is as "invalid" as an unrelated real image. Useful range is only below about 0.7.
7. SAM jumps to 0.17-0.20 for invisible noise or colour jitter: a method that adds texture/noise cannot score below about 0.2.
8. Gaming: the real refs make FID about 0 and beat the baseline on all axes; never use `data/real/`. Black image gets failure 1.0 and SAM 0.62 (better than baseline), caught only by FID/KID/diversity.
9. Min meaningful deltas (3-seed spread): SAM 0.012, failure 0.14, FID 15, KID 0.03, SSIM 0.012, VGG 0.04, CLIP div 0.005. IS, GLCM and feature hamming are uninformative or noisy.
10. Fast proxy (`fast_proxy.py`): SAM proxy rho 0.98 (0.88 on generator runs), realism proxy rho 0.65-0.85; no VLM-free failure proxy (rho 0.03 on generator runs).
