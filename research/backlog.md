# Backlog

| id | status | idea | expected effect | cost |
|---|---|---|---|---|
| B0 | done | Baseline: ControlNet-Union 2.1, single control, label prompts (grid: canny/depth x scale 0.8/1.0) | reference | done |
| F1 | dropped (see negative_results.md) | Training-free hint-space fusion: run Union net per modality (canny, depth), merge per-layer hints with gates: scalar sum / mean | validity, realism | low |
| F2 | dropped (see negative_results.md) | F1 + region gates from instance_seg (canny on belt/phone/object pixels, depth on seats/cabin, ...) | validity on thin objects | low |
| F3 | dropped | F1 + timestep gates (structure early, release late) | realism | low |
| F4 | dropped (see negative_results.md) | F1 + conflict-aware merge: where hint vectors disagree (cos<0) project out the conflicting component (PCGrad-style) | fewer artifacts | low-med |
| T1 | idea (blocked) | Trained shared control trunk: modality tokens + fidelity-knob embedding, modality dropout, frozen backbone, zero-init, control-off distillation | the real contribution | high |
| D1 | idea (blocked) | Self-generated training pairs: Z-Image-Turbo renders photoreal in-car scenes from label prompts; extract canny/depth/seg -> (control, image) pairs; plus 60 real photos | data for T1 | med |
| P1 | active | Label-aware prompting + per-feature diversity sampling (clothing, light, skin) that keeps labels valid | failure rate, diversity | low |
| I1 | active | Init-from-sim (SDEdit-style): start denoising from the noised VAE latent of the sim image (strength s in 0.5-0.9) plus ControlNet hints. Keeps colours/layout the metrics reward; Turbo supplies photo realism | FID/VGG/validity up, human realism unknown | low |

## Critic ranking (2026-10-17, see verdicts.md)
1. I1 init-from-sim (active; report full strength curve, must pass human realism, beware near-copy exploit)
2. P1 extended: style/colour prompt matching to real set (dim IR-like fisheye, low saturation) and per-feature agreement analysis (active, cheap, targets FID/failure)
3. M1 measurement: seeds, per-feature agreement, paired CIs, fix make_sheet pairing (active, prerequisite for any claim)
4. T1/D1: blocked until I1/P1 show where the metric headroom is; training cost high, novelty only in combination
Notes: F1-F4 hint fusion = no-op vs canny s1.0 at 2.4-3.6x time (F1 sum 255 FID, F2 256, F4 256); F3 dropped (no signal in fusion). Depth control alone is structurally best (SAM 0.59, lower better) but unrealistic (FID 447); possible role only as weak low-frequency prior inside I1. Scale 1.0 beats 0.8 for canny (failure 0.78 vs 0.87, FID 254 vs 267, within noise on failure).
| B1 | idea | Baseline with the clean Turbo sigma schedule: the 2-step warm-up in run_experiment.py leaves scheduler.config.shift_terminal=2/3 set, so all baseline and hint_fusion 8-step runs used sigmas 1,.98,.96,.93,.90,.85,.78,.667,0 instead of 1,.947,.882,.8,.692,.545,.333,0. Rerun canny s1.0 with a clean schedule (skip or reset after warm-up) | possibly better realism for free; invalidates the comparability of earlier rows | low |
| T1s | active | T1 smoke subset on GPU 1: train trunk (K=4, 512x288) on scene pairs (sim canny/depth/seg -> paired REAL photo) for scenes 0-39, hold out 40-59 for eval vs baseline on the same 20 | first trained evidence, overfit expected | med |
| R1 | queued (after I1/T1s/measurement) | Two-stage realism mapping: stage 1 = structure-faithful but less realistic image (sim itself, or I1 s0.5, or hint-guided low-realism render preserving canny/seg); stage 2 = realism mapping on top: low-strength Z-Image img2img/refinement, photo-style transfer (e.g. PhotoWCT/WCT2, CUT/CycleGAN-style or 'Enhancing Photorealism' trained on the 60 real photos), colour/tone matching to real references, seg-guided region-aware blend (keep belts/phones/objects, restyle seats/windows/skin). Must keep SAM validity and label features | decouples fidelity from realism; may beat I1's curve | med |
| H1 | dropped (see negative_results.md) | HyNeA-faithful per-instance optimisation: per scene, train a zero-init copy of the first K Z-Image blocks (hypernet) with AdamW/OneCycle for 100-500 steps through the full 8-step denoising + VAE decode; loss = fidelity to an anchor (I1 s0.7 output and/or sim structure) + steering (differentiable Qwen2.5-VL yes/no feature logits pushed away from the real-ref reading = evaluator failure objective) | targets failure rate directly; slow (minutes/scene) | high |
| H2 | dropped (see negative_results.md) | Hybrid: amortised T1 trunk (T1s checkpoint) + few-step per-instance HyNeA-style adaptation with the H1 loss | most of T1's speed + direct failure targeting | high |
