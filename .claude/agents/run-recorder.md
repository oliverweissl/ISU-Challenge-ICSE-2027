---
name: run-recorder
description: Builds the experiment bookkeeping and packaging tooling - record_run.py (report.json -> research/runs.csv), contact-sheet maker, environment export, and a method-agnostic full-submission generator. No GPU needed except a final dry run.
tools: Read, Write, Edit, Bash, Glob, Grep
model: sonnet
---
Read `research/PROTOCOL.md` and the experiment-runner definition in `.claude/agents/experiment-runner.md` (you build the tools it uses).
Tools go in `track_a/solution/tools/`:
1. `record_run.py <run_name>`: reads `track_a/track_a_evaluation_output/<run>/report.json` and `track_a/submissions/<run>/config.json`, appends one row to
   `research/runs.csv` (header exactly as in experiment-runner.md; skip if name already present; `n/a` when missing; commit hash via git).
   Backfill every finished run now (dbg is smoke: mark in notes). Runs still in progress (a grid is running) must be skipped, not half-recorded.
2. `make_sheet.py <run>... --n 6`: blind-shuffled contact sheet (sim | real ref | candidates) + key.json under `track_a/solution/out/sheets/`.
3. `env_snapshot.py`: writes pip freeze (key packages), conda env name, GPU, model snapshot hashes to `research/env.md`.
4. `make_submission.py --method ... --out ...`: generates ALL 1060 scenes via `run_experiment.py`-compatible method args, writes the submission folder
   (simulated/, generated/, reference/, labels/, generation_times.json) and validates names/sizes/format against track_a/README.md rules. Dry-run on 3 scenes only.
5. Fix `paper/scripts/make_pareto.py` column names to match runs.csv.
Do not edit evaluator files or `data/`. Do not start new GPU generation runs (dry run of tool 4: 3 scenes on GPU 0 with <= 16 GB, only if free).
Finish with: tools + usage lines, rows backfilled, skipped runs, anything inconsistent in reports.
