# Bookkeeping tools (python: `~/miniconda3/envs/zimage/bin/python`)

- `python tools/record_run.py <run>... [--note TEXT] [--all]` : append finished run(s) to `research/runs.csv`
  (skips runs without `efficiency` in report.json and names already present; `n/a` for missing values).
- `python tools/make_sheet.py <run>... [--n 6] [--seed 0] [--name SHEET]` : blind-shuffled sheet
  (sim | real ref | candidates) to `out/sheets/<name>.png` + `<name>.key.json`.
- `python tools/env_snapshot.py [--python PATH]` : writes `research/env.md` (key packages, GPUs, HF snapshots, commits).
- `CUDA_VISIBLE_DEVICES=<gpu> python tools/make_submission.py --name NAME [--n 1060] [--out DIR] -- <run_experiment.py args>`
  : generate full submission (default 1060 scenes) and validate; `--validate-only --out DIR [--n N]` validates an existing folder.
  Example args: `-- --control canny --scale 0.8` or `-- --method hint_fusion --fusion-mode conflict_aware`.
- `python paper/scripts/make_pareto.py` : Pareto figure from runs.csv (60-scene rows).

Note: run `make_submission.py` from any cwd; run_experiment.py copies all 60 real refs into `reference/`.
- `tools/gpu_run.sh [--min-free MiB] [--log F] -- <cmd>` : MANDATORY wrapper for every GPU job (generation, evaluator, training, smoke tests).
  One job per GPU at a time (flock in /tmp/isu_gpu_locks), waits while another user leaves < min-free MiB. Never launch a GPU process without it.
