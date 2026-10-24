# Verdicts (critic)

## 2026-10-17 - batch 1 (baseline grid, 5 hint_fusion runs, ref_* references)

**Decision: PIVOT away from hint-space fusion (F1-F4); ITERATE on structure preservation (I1) and measurement. Not ready to ship.**

Provisional: research/evaluator_analysis.md did not exist at review time; evaluator claims below were checked directly against
track_a/track_a_metrics.py and track_a/image_metrics_utils.py.

### Evidence rows (runs.csv, n=60, one seed each; failure/SAM/FID)
| run | fail | SAM (LOWER better) | FID | s/scene |
|---|---|---|---|---|
| base_canny_s0.8 | 0.867 | 0.759 | 267 | 2.5 |
| base_canny_s1.0 | 0.783 | 0.761 | 254 | 2.5 |
| base_depth_s0.8 | 0.850 | 0.592 | 447 | 2.5 |
| hf_sum_60 | 0.817 | 0.757 | 255 | 5.9 |
| hf_region_60 | 0.783 | 0.755 | 256 | 9.1 |
| hf_conflict_60 | 0.817 | 0.756 | 256 | 7.5 |
| hf_phase_canny_full_60 | 0.783 | 0.755 | 258 | 6.3 |
| hf_sum_w83_60 | 0.833 | 0.737 | 276 | 6.8 |
| ref_real_rs (real photo resized) | 0.667 | 0.702 | 0.4 | 0 |
| ref_real_jpeg | 0.583 | 0.713 | 9.9 | 0 |
| ref_sim_noise3_s1 (sim + tiny noise) | 0.783 | 0.193 | 129 | 0 |
| ref_sim_jitter | 0.833 | 0.166 | 132 | 0 |

### Contact sheet (track_a/solution/out/sheets/batch1.png, key in batch1.key.json; A=canny1.0, B=depth0.8, C=hf_region, D=hf_sum)
- C and D are visually near-identical to each other and close in layout to A: fusion = canny baseline with a different colour cast. No visible depth contribution, no visible region effect.
- B (depth) is not a photo: pastel, washed-out depth-like render with black windows. Its good SAM (0.59) is structure copied from the sim, not realism (FID 447). Metric-gaming-style low validity score, do not rank it as better.
- A/C/D: plausible but clean "stock-photo" cabins, saturated colour, different framing and different people/hair/ethnicity than the paired real photo; colour/lighting shifted vs the dim IR-like real photos.

### Challenges to the main thread's reading
1. **SAM direction**: semantic_segmentation_score = 1 - matched_IoU, lower is better (image_metrics_utils.py:617, 729). Depth s0.8 has the BEST validity (0.59), the w(0.8,0.3) fusion is better than sum (0.737 vs 0.757). Any text that treats a higher SAM as better, or calls depth "worse", must be corrected. Depth loses on realism, not validity.
2. **"Modes indistinguishable" is right for failure, but the conclusion is stronger: the fusion variants are not just within noise, they are a no-op.** FID/SAM/VGG match canny s1.0 to 3 digits; sum at scale 0.8 with two hints is behaving like canny at higher strength. They cost 2.4-3.6x time and F1-F4 break the <=5 s/scene DoD item (region 9.1 s, conflict 7.5 s, sum 5.9 s). Negative result: do not spend more on F2-F4 variants with this backbone. Also F4 conflict-aware is the only candidate for novelty and shows zero effect, so no empirical support to claim it.
3. **The failure-rate floor is much worse than "SE 0.05"**. Failure = generated vs REAL photo of the same stem, ANY of the public features mismatching under Moondream (track_a_metrics.py:292-348). A real photo only resized scores 0.667 and real jpeg 0.583 (VLM instability across many features). So failure is a saturated, binary any-mismatch metric; our 0.78-0.87 is only 0.1-0.2 above the floor of identical content. Seed noise (SE ~0.05 binomial, plus Moondream non-invariance) swamps 0.78 vs 0.87. Use per-feature agreement from report.json feature_matches and paired per-scene tests instead.
4. **Real refs ARE paired to scenes** (data/real/sample_000NN_real.jpg shows the same layout, person and objects as sample_000NN sim; checked 00002 and 00024). The failure metric and SSIM/PSNR/VGG are paired. tools/make_sheet.py comments say "no 1:1 pairing" and picks reals[i % len] by row index, not by stem, so the "real ref" column in batch1.png is the WRONG real image for every row. Fix needed. Consequence: the metrics reward reproducing the real photo's person/objects/colours, which the label prompt cannot fully specify (gender/colour/light are ignored features but appearance still drives VLM answers).
5. **The main thread's I1 reading is half right, and the danger is bigger than stated.** Trivial references ref_sim_noise3_s1 / ref_sim_jitter beat every generated run on SAM (0.19 vs 0.76), FID (129 vs 255), VGG (0.78 vs 0.93), SSIM, efficiency (0 s) and tie on failure (0.78). They lose only on clip_div (0.09 vs 0.15). So the current Definition of done ("Pareto-dominate baseline on >=3 of 5 objectives") is satisfiable by a near-copy of the sim, and I1 at low strength will approach it. That is a metric exploit, not sim-to-real realism (sim renders look CGI). I1 is worth testing but only reported alongside the human realism rating, and the best I1 strength is the one that stays photoreal in the sheet, not the lowest FID. Expect I1 strength to trade FID/SAM against realism monotonically; report the whole curve.
6. Generated FID (255) is farther from real than raw sim (~130) mainly because colour statistics (bright, saturated, stock-photo look) differ from dim real photos, not because geometry is wrong. Cheaper than I1 and orthogonal: prompt/colour style matching to the real set (P1 extended: "dim infrared-like in-cabin camera, fisheye, low saturation"). Not yet tested.
7. Missing: seeds (all runs single seed, no repeat), no baseline_depth s1.0 row in runs.csv though the submission exists, no ref_id_sim/ref_black/ref_blend/ref_sim_gray/ref_shuf_real rows, ref rows lack config. "sim_blur FID 328" is not a baseline for us.

### Risks
- Overfitting to FID/VGG with a near-copy of sim: low human realism, possibly disqualifying in spirit.
- n=60 single seed: all differences < ~0.08 in failure are noise; SAM/FID/VGG differences (depth vs canny) are real.
- Novelty: F1 is done before (Cross-ControlNet, MaxFusion), F2 close to AtteConDA; both are now also empirically null, so the paper cannot claim fusion. F4 untested at equation level in related_work.
- Time budget DoD (5 s/scene) already broken by all fusion variants.
- Human realism rating never collected (0 so far); >=3 iterations threshold effectively reached with visually distinct candidates.

### Readiness vs Definition of done
- Novelty verdict: not stated; scout has not full-text-verified Cross-ControlNet/MaxFusion/AtteConDA. MISSING.
- Pareto-dominance of best run over baseline on >=3/5 without worse validity: NOT met (hf_sum ~ canny s1.0; depth better only on SAM). MISSING.
- Human realism >= baseline: no data. MISSING.
- <= 5 s/scene: baseline 2.5 OK; all fusion variants fail except none. 
- Reproduce README, frozen code, full submission: not started.

### Next 3 tasks
1. **method-engineer then experiment-runner**: implement I1 init-from-sim (single canny hint, scale 1.0, no fusion, 8-step Turbo; strengths 0.5/0.7/0.9) and run n=60 each (seed 0); finish: three runs.csv rows named i1_canny_str{50,70,90}_60 with s/scene, plus a blind sheet (sim | paired real | base_canny_s1.0 | i1 strengths | ref_sim_noise3_s1) and a note on where photorealism visibly breaks.
2. **experiment-runner**: noise and measurement hygiene; finish: (a) 3 seeds of base_canny_s1.0 and 1 repeated seed of hf_sum_60 giving a seed-SD table for failure/SAM/FID; (b) per-feature agreement table from report.json for all existing runs with paired per-scene bootstrap CIs vs base_canny_s1.0; (c) runs.csv rows for base_depth_s1.0, ref_id_sim, ref_black, ref_blend, ref_sim_gray, ref_shuf_real, and sim_vs_real realism row; (d) make_sheet.py fixed to use the stem-paired real image (data/real/<stem>_real.jpg) and the stale wrong sheet regenerated.
3. **literature-scout**: finish: updated related_work.md with full-text checks of Cross-ControlNet, MaxFusion, AtteConDA (does any use a negative-cosine projection) and a new search for SDEdit/img2img-from-simulation with ControlNet (sim-to-real in-cabin / driver monitoring, e.g. Z-Image or FLUX img2img+control), ending in a novelty verdict for F4 and for I1 with the closest 3 papers named. Optionally in parallel, once task 1 produces candidates, request the human realism rating (artifact page, shuffled: base_canny_s1.0, base_depth_s0.8, hf_sum_60, best I1).
