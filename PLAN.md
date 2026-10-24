# PLAN: region-wise repainting of the simulation (next executions, NOT yet run)

Written 2026-10-18. Status: plan only. Nothing here has been started.

## Why this direction
Human review (the user, 2026-10-18) of the sprint outputs: FLUX / Kontext renders (m2, m3) look very good; Realistic-Vision img2img (m4) makes great people but garbles
everything else. FID/SAM could not separate "looks real" from "copies the sim", so **human ranking is the primary criterion; metrics are sanity checks** (see
`research/exp_models.md`, `research/negative_results.md`, `research/deleted_experiments.md`).
Task goal (user): **feature-level perturbation**: change attributes (colour, texture, structure, lighting, clothing, style) of items, but **never the presence or the quantity**
of people/objects (baby seat, phone, suitcase, cola bottle/can, passengers, belts). So: the sim segmentation defines *what exists and where*; models only repaint *inside* regions.

## Principle: one model per job, regions come from `data/instance_seg` (class map: `data/instance_seg/instance_seg_class_map.json`)
| Stage | Region (mask from sim seg) | Model candidates | Job |
|---|---|---|---|
| S1 Surroundings | cabin: seats, windows/exterior, dashboard, doors, headliner (everything not a person/object) | FLUX.1-schnell img2img (`other_models/flux_gen.py`, 0.6 s/scene), FLUX.1-Kontext-dev (`kontext_gen.py`, 10.5 s: too slow full-frame, test on region/fewer steps) | photoreal materials, lighting; geometry kept by low strength + canny |
| S2 People / faces | person + belt + driver-phone masks (class ids 1-4, belts 5-8, 15; see `hybrid_gen.py`) | Realistic Vision V5.1 + canny ControlNet on a crop (`sd15_gen.py`), alt: FLUX/Kontext on the crop | realistic skin, faces, hair, clothing |
| S3 Objects | baby seat, phone, suitcase, cola can/bottle masks | FLUX img2img on the crop, low strength, colour taken from the label | keep each object, repaint texture/colour |
| S4 Normaliser | whole image after S1-S3 | (a) no-real: Z-Image-Turbo img2img polish s<=0.3 + prompt-driven grade ("dim IR-like fisheye in-cabin camera"); (b) `_R`: colour/tone transfer fitted on real photos 0-39 (Reinhard/histogram/3D-LUT, `classical/c1.py`), held-out 40-59 | make the whole frame share the real dataset's look (dim, warm-green cast, blown windows, fisheye softness) without touching content |
Compositing: feathered mask (as in `hybrid_gen.py`: MaxFilter 5 + Gaussian 4); every stage reads the previous stage's output and only writes inside its mask.

## Executions (all via `track_a/solution/tools/gpu_run.sh`, n=60, seed 0, harness `exp_common.run_submission`; smoke n=3 first)
| id | what | needs | success check |
|---|---|---|---|
| R0 | Re-render references: S1 only (FLUX s0.5/0.6, no people repaint) = existing m2 rows; Kontext-on-regions smoke | existing code | baseline for the human sheet |
| R1 | S1 + S2 hybrid = existing `m6_hybrid_flux_rv` at n=60 (presence by construction) | `hybrid_gen.py` | human rank vs m2, m3; presence-violation rate = 0 |
| R2 | R1 + S3 object repaint (FLUX crops, label colours) | new `objects_gen.py` | objects keep identity/count (presence check) |
| R3 | S1 via Kontext restricted to cabin mask / crop tiles, steps and resolution tuned to <= 5 s total | `kontext_gen.py` | human rank, time |
| R4 | S2 alternatives: RV vs FLUX-crop vs Kontext-crop for faces | crop pipeline | human face-quality rank (blind) |
| R5 | R2/R3 + S4a no-real normaliser (Z-Image s<=0.3 / prompt grade) | `init_sim` | look matches real photos in blind sheet |
| R6 | R2/R3 + S4b `_R` colour/tone fit (0-39), eval on 40-59 only | `classical/c1.py` | held-out realism; flag leakage; also keep the no-real best |
| R7 | Feature-perturbation suite on the best pipeline: for each of {clothing colour, seat colour, baby-seat colour, suitcase colour, lighting dim/bright, head angle/emotion if label-driven} produce a perturbed variant, others fixed | label edits + masks (`classical/c5.py` is the cheap recolour control) | target feature changes (VLM/colour stats on the mask), everything else unchanged, presence/quantity identical |
| R8 | Ablation: each stage off in turn (S1-only, +S2, +S3, +S4) on the same 20 scenes | R1-R6 outputs | shows per-stage contribution for the paper |
| R9 | Seeds: best pipeline x 3 seeds; paired per-feature bootstrap (`analysis/per_feature.py`) | R5/R6 best | reports effect sizes beyond noise (failure SD 0.05) |
| R10 | Freeze: pick winner by human rank, then generate all 1060 scenes via `tools/make_submission.py` (`TestTeam`), validate, README | winner | <= 5 s/scene on L40S; no real photos in `generated/` |

## Checks every run must report
1. **Presence/quantity**: per scene, every labelled person/object present exactly once (`other_models/presence_check.py`, `feature_check.py`); violation rate must be 0.
2. **Human blind sheet** (`tools/make_sheet.py`, <= 6 scenes, shuffled, sim | paired real | candidates): the user ranks; this decides.
3. Evaluator numbers as sanity only: SAM validity (<0.7), KID/FID, failure, clip diversity, **s/scene (<= 5 s target; report honestly)**.
4. Near-copy flag: if a run is visually just the sim with a grade, it is not a result.

## Rules carried over (research/PROTOCOL.md, disk_policy.md)
- Every GPU process through `gpu_run.sh` (one job/GPU, `--min-free` memory). Two L40S (46 GB) shared; FLUX + RV together must fit <= 38 GB (use CPU offload; model swapping within the generator).
- Real photos: never in `generated/`; `_R` variants fit on scenes 0-39, judged on 40-59, plus a no-real variant; organisers' permission for training on the 60 real photos still unconfirmed.
- **No GANs, ever** (user decision). Diffusion/flow models and classical colour ops only.
- Disk: delete each run's `sam/` folder right after recording; delete weights of dropped models; keep `/data` >= 80 GB free.
- Document every decision/method/setup in `research/exp_regional.md` (one section per R-id); `methodology.md` / `setup.md` are updated by the notekeeper hook.

## Order and cost
R0/R1 (existing code, ~1 h GPU) -> human check -> R3/R4 in parallel -> R2 -> R5/R6 -> R7 -> R8/R9 -> R10. Stop after the first human check if R1 is not ranked above m2/m3 alone.
Agents: `method-engineer` for new generators (objects_gen, mask-restricted Kontext), `experiment-runner` for runs + sheets, `critic` once after R6; keep at most 2 agents active (2 GPUs).
