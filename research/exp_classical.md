# Family C (classical / style) - agent `classical`

Code: `track_a/solution/classical/` (c1.py, c2.py, c3.py, c5.py, gen_style.py, make_hold.py, chain_*.sh). Python: zimage env. GPU: shared L40S via `tools/gpu_run.sh`
(GPUs were blocked by an external ollama process holding 39-44 GB for the first ~hour, then by flock contention with other agents; wait times, not compute, dominated).
Data facts used (checked by looking at 6 sim/real pairs): real photos are RGB (not IR), warm/olive cast, mean chroma 25 vs sim 9.6, brighter windows, stronger contrast, slight grain/JPEG.
Sim is geometrically near-aligned with the real photo (same camera), so every method below is a near-copy of the sim by construction. All SAM values are therefore "copy-ish" (0.14-0.48).
Leakage rule: `_R` = uses real photos of scenes 0-39 only (stats/style target); `*_hold20` = same generated images, scenes 40-59 only, evaluated against the full real reference set
(n=20, FID/KID NOT comparable to n=60 values; compare only with other hold20 rows). No real photo ever enters `generated/`.

## Results (n=60, numbers from report.json via research/runs.csv)
| run | R? | failure | SAM | FID | KID | VGG | SSIM | clip_div | s/scene |
|---|---|---|---|---|---|---|---|---|---|
| ref_id_sim (identity sim, reference) | - | 0.800 | 0.000 | 133.3 | 0.137 | 0.791 | 0.456 | 0.085 | 0 |
| is_s0.5_60 (init-from-sim, other agent) | - | 0.783 | 0.382 | 116.0 | 0.105 | 0.700 | 0.463 | 0.105 | 1.46 |
| c1_hand_60 | no | 0.750 | 0.179 | 137.6 | 0.142 | 0.838 | 0.351 | 0.101 | 0.02 |
| c1_hand_fx_60 | no | 0.733 | 0.372 | 139.0 | 0.142 | 0.689 | 0.357 | 0.109 | 0.10 |
| c1_reinhard_60 | R | 0.750 | 0.137 | 140.7 | 0.142 | 0.900 | 0.457 | 0.095 | 0.10 |
| c1_hist_60 | R | 0.733 | 0.230 | 126.1 | 0.122 | 0.812 | 0.455 | 0.087 | 0.09 |
| c1_hist_fx_60 | R | 0.733 | 0.324 | 122.6 | 0.112 | 0.661 | 0.458 | 0.103 | 0.18 |
| c2_hand_s0.35_60 | no | 0.750 | 0.288 | 143.1 | 0.150 | 0.839 | 0.365 | 0.101 | 1.11 |
| c2_hand_s0.5_60 | no | 0.767 | 0.390 | 127.7 | 0.120 | 0.770 | 0.362 | 0.106 | 1.50 |
| c2_reinhard_s0.5_60 | R | 0.633 | 0.392 | 121.5 | 0.102 | 0.782 | 0.466 | 0.102 | 1.57 |
C3_RESULTS_PLACEHOLDER

Holdout scenes 40-59 (n=20; only compare with each other; baseline s1.0 hold20 FID 272, is_s0.7 hold20 145):
| run | failure | SAM | FID | KID | VGG | SSIM | clip_div |
|---|---|---|---|---|---|---|---|
| c1_hist_fx_60_hold20 | 0.70 | 0.310 | 134.1 | 0.120 | 0.691 | 0.448 | 0.094 |
| c1_reinhard_60_hold20 | 0.80 | 0.134 | 159.3 | 0.173 | 0.993 | 0.445 | 0.077 |
| c2_reinhard_s0.5_60_hold20 | 0.60 | 0.389 | 139.8 | 0.123 | 0.854 | 0.454 | 0.086 |
Held-out numbers are in line with in-sample ones (same ordering, FID inflated by n=20), so the `_R` gains are not just memorisation of scenes 0-39 (the fitted quantities are colour-curve/stat vectors or a global style).

## C1 - pure image processing (classical/c1.py, no network)
Hypothesis: a trivial colour/noise pipeline closes part of the FID gap. DECISIONS: parameters of `hand` set by eye from the 6 pairs I inspected (warm tint (1.06,1,0.86), saturation x1.9, gamma 0.9, contrast 1.25) - this is NOT strictly "no real information" (I looked at photos, and computed channel means), so treat `hand*` as weakly informed; `reinhard`/`hist` fit pooled statistics (one target for all scenes, never the paired photo; paired matching would leak) of real 0-39 at 480x270 (Lab mean/std; 257 per-channel quantiles of RGB). fx = Gaussian blur 0.7 px, vignette 0.25*r^2, luminance-weighted gaussian grain (sigma 3.5) + chroma noise, JPEG q65, seed = scene seed.
Setup: 960x540, CPU, 0.02-0.18 s/scene.
Findings: no variant beats the plain sim by a meaningful margin on realism except `hist_fx` (FID 122.6 vs 133.3, KID 0.112 vs 0.137, VGG 0.661 vs 0.791, i.e. 10 FID points - about the seed SD x2 - and this is the only one using real stats). `hand*` and `reinhard` are at or above identity-sim FID (137-141). Reinhard in Lab gives the worst VGG (0.90): over-saturated skin tones and cyan windows (images). `hist` quantile mapping is the better statistics match but washes colours. Grain/blur/JPEG alone are what lowers VGG (0.84 -> 0.69 for hand -> hand_fx) while SAM rises 0.18 -> 0.37 (SAM penalises texture, evaluator_analysis section 3). FID cost of SAM 0.2-0.4 is nothing here: these are near-copies; failure 0.73-0.75 is in the noise (0.14 min delta).
Visual: all remain obviously CGI (plastic faces/hair, clean geometry); only global colour/noise changed. Verdict: control confirms a trivial pipeline moves FID by <=10 points; NOT a realism win. Near-copy flag: yes for all (SAM <= 0.37).

## C2 - classical pre-process + Z-Image polish (classical/c2.py)
Hypothesis: colour-correct the sim first, then low-strength ZImageInitSim (reused read-only, clean sigma schedule like n004's `is_s0.5`, canny+depth sum fusion, scale 0.8, label prompt, 8-step Turbo, seed = scene seed; same config as is_s0.5_60 except the init image).
Setup: s in {0.5, 0.35}; pre = hand (no-real, weakly informed) or reinhard (_R). 1.1-1.6 s/scene.
Findings: is_s0.5 on raw sim remains the best FID/KID of the whole family (116.0/0.105): colour pre-processing does not help the polish. hand+s0.5: FID 127.7 (worse than raw), the saturated palette is partly kept by Z-Image; hand+s0.35: FID 143 (too close to the copy, SAM 0.29). Reinhard+s0.5 (_R): FID 121.5, KID 0.102, SSIM 0.466, failure 0.633 (lowest failure of the family; real-vs-real floor is 0.58-0.67, so this is 0.15 below identity sim, 1.1 SD seed noise ~0.05 -> about 3 SD, but only one seed; hold20 failure 0.60 agrees). Visual: images look like a softer, more photographic render with muted palette; faces/bodies still CGI; an orange tint on red child seats. Verdict: promising only on the failure metric and only with real colour statistics; not distinguishable from is_s0.5 on FID. Needs seeds before any claim. Near-copy flag: SAM 0.39 (same as is_s0.5), structure-preserving img2img not a copy at pixel level.

## C3 - neural photo-style transfer (classical/c3.py, gen_style.py)
DECISION: no AdaIN/WCT2 decoders available offline, training one is out of scope; implemented optimisation-based statistics transfer on VGG16 (torchvision weights in torch hub cache; relu1_2..relu4_3 channel mean/std = AdaIN statistics, pooled over the style set; content loss relu3_3; luminance layout anchor at 1/16 res; TV; Adam lr 0.02, 60 steps, 960x544, bf16 autocast, w_c 1, w_s 30, w_p 50). Style sets: `real` = 40 real photos 0-39 (_R); `zgen` = 12 Z-Image-Turbo text-to-image renders with generic in-cabin camera prompts (canny of scenes 100-111, two prompts, gen_style.py; no real photos).
Setup: 0.75 s/scene on the L40S.
C3_TEXT_PLACEHOLDER

## C4
C4 GAN removed by user decision, not pursued.

## Cross-family take-aways
1. Everything in C is a re-colouring/re-texturing of the sim; FID cannot go below ~115-120 this way, the same level as init-from-sim s0.5 (116.0). The metric floor for near-copies is ~120 vs 258 for text-to-image; the open problem is content realism (CGI faces/materials), which classical tools cannot touch.
2. Real colour statistics (R) are worth ~10 FID points and clearly lower VGG; the no-real variants do not beat identity sim.
3. Failure is noise-level except c2_reinhard (0.633 / hold20 0.60); verify with 2-3 seeds.
