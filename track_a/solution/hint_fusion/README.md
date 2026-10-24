# hint_fusion (backlog F1-F4): training-free hint-space fusion

Z-Image-Turbo + ControlNet-Union 2.1. The control branch runs once per modality (canny, depth); the 17 per-layer
hints (2 noise-refiner + 15 main-layer) are merged before being added to the base transformer. No training.
Cost: control branch runs 2x (~4.1-4.7 s/img vs ~4.1 s single on a shared GPU, 26.9 GiB peak, full load).
`FusionConfig` defaults + `canny_only` weights (1,0) reproduce the baseline bit-exactly (checked, max diff 0).

## Modes (`FusionConfig.mode`)
- `sum`: sum_m weights[m] * hint_m (raw weights).
- `mean`: weights normalised to sum 1.
- `region_gated`: per-token weights from `data/instance_seg/<stem>_instance_seg.png` through `gate_table.json`
  (class name -> raw canny/depth multipliers; average over each 16x16 token; caption tokens use `_default`).
- `phase_gated`: weights go linearly (or `phase_shape=step` at `phase_switch`) from `phase_start` (t=0, noise) to
  `phase_end` (t=1, clean); t is the pipeline-normalised timestep.
- `conflict_aware`: per token, cos(h_canny, h_depth) < `conflict_thresh` -> project each hint off the other
  (PCGrad), then combine with `conflict_base` (sum|mean). Raw hints have mean cos ~0.8 and never conflict, so cos and
  projection use the token-mean-centered residual (`conflict_center=True`); ~0.7% of tokens conflict on average
  (up to ~8% in some layers).

Other fields: `weights`, `gate_table`, `gate_normalize`, `seg_class_map`, `phase_start/end/shape/switch/normalize`,
`conflict_thresh/strength/center/base`, `modalities`.

## Data findings (important)
- `data/depth/png/*.png` is 16-bit; `Image.convert("RGB")` saturates it to white. Also bright = far and the exterior
  (windows) is noise. `load_control` converts depth to 8-bit, near = bright, exterior (from instance_seg) = far,
  1-99 percentile stretch. The baseline depth grid in run_experiment.py (unchanged) uses the broken white image.
- Depth hints are strong: at effective depth weight > ~0.3 next to canny 1.0, output collapses to a pastel depth
  render. Depth-only works at scale ~0.5. Defaults therefore use depth weight 0.2.

## Run
    python run_experiment.py --name hf_sum_60 --method hint_fusion --fusion-mode sum --n 60 --eval
    python run_experiment.py --name hf_region_60 --method hint_fusion --fusion-mode region_gated --n 60 --eval
    python run_experiment.py ... --fusion-config '{"weights":{"canny":0.8,"depth":0.3}}'   # JSON/file overrides
    python hint_fusion/smoke.py sample_00000,sample_00003      # side-by-side to ../out/hint_fusion_<stem>.png
`--w-canny/--w-depth/--gate-table/--phase-shape/--conflict-base/--memory-mode/--device` are also available.
