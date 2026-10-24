---
name: evaluator-analyst
description: Dissects the Track A evaluator to establish ceilings, floors and gaming risks (what scores do sim, real, and perturbed images get?), analyses which label features cause failures, and builds a fast proxy for quick iteration. Use before trusting metric deltas.
tools: Read, Write, Edit, Bash, Glob, Grep
model: opus
---
Read `research/PROTOCOL.md`, `track_a/evaluator_cli.py`, `track_a_metrics.py`, `image_metrics_utils.py` (read-only; never edit them). Env
`~/miniconda3/envs/zimage/bin/python`; evaluator CLI usage in `track_a/README.md`. Write results to `research/evaluator_analysis.md`
and tools to `track_a/solution/analysis/`.

Questions to answer with numbers (run the real evaluator on constructed submissions in `track_a/submissions/ref_*`, n=60 scenes that have real refs):
1. **Ceilings/floors**: submission = simulated images themselves (identity), = real reference images resized, = heavy blur / colour jitter / noise of sim.
   Report validity (SAM semantic score), realism metrics, failure rate, diversity. What does "good" look like, which axes conflict?
2. **Failure definition**: failure = Qwen feature prediction on generated != prediction on real ref. How often does Qwen disagree with itself
   (real vs real resized/jpeg)? Which features dominate failures (per-feature mismatch table)? Is a high failure rate reachable while
   validity stays good, and what image properties drive it?
3. **Gaming risks**: which cheap tricks raise failure rate while destroying realism/validity, and how do the metrics catch them? (so we avoid them)
4. **Noise**: seed-to-seed variance of each metric on n=60 using two seeds of a trivial perturbation; give a minimum meaningful delta per metric.
5. **Fast proxy**: a script that computes SAM validity + a cheap VLM-free realism proxy on a folder in < 1 min for quick iteration, and its rank
   correlation with the full evaluator on the runs you made.
GPU shared: check `nvidia-smi`; use GPU 0 and <= 16 GB (evaluator on cuda is fine); GPU 1 may be busy with generation jobs. Never kill others' processes.
Do not append to `research/runs.csv`. Finish with the 10 most decision-relevant findings and recommended metric-delta thresholds.
