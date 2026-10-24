# Negative / dropped results (2026-10-18)

All numbers are rows of `research/runs.csv` (n=60 unless stated). Seed noise of the baseline (3 seeds, `measurement.md`):
failure rate SD 0.050, SAM SD 0.004, FID SD 5.4. Images, SAM visualisations and run folders of these methods were deleted to save
disk; their `report.json` files and `runs.csv` rows stay. Code of H1/H2 is archived in `research/archive/hynea_code.tar.gz`.
`track_a/solution/hint_fusion/` is kept because `init_sim` subclasses it (the fusion modes themselves are dropped).

| Method | Evidence | Why it did not work |
|---|---|---|
| **Training-free hint fusion F1-F4** (`hf_sum/region/conflict/sum_w83/phase_canny_full`) | Failure 0.78-0.83, SAM 0.737-0.757, FID 255-276, 5.9-9.1 s/scene, vs baseline canny s1.0: 0.783 / 0.761 / 254 / 2.5 s | Indistinguishable from plain canny (FID/SAM within seed SD) but 2.4-3.6x slower (two control passes). Union-ControlNet hints of canny and depth are already aligned (mean cos ~0.8) and only ~0.7% of tokens conflict, so conflict projection (F4) has nothing to fix. F1 was done before (MaxFusion, Cross-ControlNet). Depth weight > 0.3 washes colours out. |
| **Phase-gated fusion, default schedule** | Visual: ghosting / double exposure (smoke test) | Weak canny early lets depth dominate layout; schedule never fixed beyond the `canny_full` variant, which equals baseline. |
| **Depth-only control** (`base_depth_s0.8/s1.0`, fixed 16-bit depth loader) | Failure 0.85, SAM 0.592 / 0.580 (best structure), FID 447 / 451, KID 0.71 | Pastel, washed-out depth-like render with black windows, not a photo. Good SAM only reflects copied structure. Only `suitcase` feature agreement is better than canny. |
| **Init-from-sim at strength 0.85** (`is_s0.85_60`) | Failure 0.85, SAM 0.733, FID 233 | Starting at sigma 0.882 with big Turbo steps gives ghosting, double people and distorted faces. Sweet spot is 0.5-0.7 (kept). |
| **HyNeA-style per-instance optimisation H1** (`h1_k2_10`, n=10) | Failure 10/10, 38 mismatches of 130, SAM 0.69, FID 238, 59 s/scene | Steering-off control (`h1ctl_k2_10`) has failure 10/10, 30 mismatches, better SAM 0.63 and FID 198: steering adds only ~8 mismatches, the rest comes from the 256x144 low-resolution regeneration. Output is blurry. Failure metric is saturated (resized real photo already 0.667), so little room to gain. 59 s/scene vs 5 s budget. |
| **Hybrid H2** (T1 checkpoint + 40 per-instance steps, n=10) | Failure 10/10, 32 mismatches, SAM 0.647, FID 300, 41 s/scene | Same limits as H1; 7 of 10 scenes reached the target already at init (behaviour of the T1 trunk, not of per-instance optimisation). |
| **T1 smoke1** (`t1s_k*`, knob embedder frozen) | Knob 1.0 and 0.5 rows bit-identical | Bug: `requires_grad_(False)` on the base also froze the copied knob embedder (last layer stays zero). Invalid run; fixed in smoke2, smoke1 checkpoint deleted. |
| **Reference submissions** (`ref_*`, sim copies, blur, noise, black, shuffled/real photos) | `evaluator_analysis.md` | Not methods. Evaluator probes. Regenerable with `track_a/solution/analysis/build_ref_submissions.py`; reports kept. |

## Lessons carried forward
- Failure rate is saturated and noisy; judge on per-feature agreement, FID/KID/VGG, SAM together.
- Fusion of two near-identical control hints is not where the leverage is; the control signal and realism mapping are.
- Per-instance optimisation needs full-resolution backprop through the sampler (memory) to be useful; not affordable.

## GANs removed (2026-10-18, user decision)
Small unpaired residual GAN (C4, `c4_c4a_60`, `c4_c4b_60`, trained on sims vs real photos 0-39; SAM 0.39 / 0.48, FID 139 / 129, failure 0.75) was deleted with its code, checkpoints, submissions, reports and `runs.csv` rows. Reason: GAN-based sim-to-real is outdated; diffusion-based methods only. Never run, train or re-add GAN / CycleGAN / CUT / pix2pix-style methods. (InstructPix2Pix `m3_ip2p` is a diffusion editing model, not a GAN, and stays.)
