# Research loop protocol (Track A)

Goal: (1) a control-fusion method that is **novel vs. published work** and (2) a Track A
submission that is **leaderboard-competitive** (Pareto on validity, realism, failure rate,
diversity, efficiency). Agents work in parallel and talk through files in `research/`.

## Shared state (single source of truth)
| File | Owner | Content |
|---|---|---|
| `backlog.md` | critic (edits), anyone (proposes) | ideas with id, status `idea/active/done/dropped`, expected effect, cost |
| `related_work.md` | literature-scout | papers, what each does, overlap with our ideas, novelty verdict |
| `runs.csv` | experiment-runner | one row per evaluated run: name, method, params, n, all headline metrics, commit |
| `verdicts.md` | critic | dated decisions: iterate / pivot / ship, with evidence from runs.csv |
| `realism_ratings/` | human via artifact | human realism rankings; critic treats them as ground truth for realism |
| `../methodology.md`, `../setup.md` | notekeeper | narrative + parameters (hook-driven) |
| `../paper/` | paper-writer | 4-page ICSE solution paper (`make pdf`), `paper/TODO.md` lists missing results |

## Roles
- `literature-scout`: find prior work, keep novelty map current. Web only, no code edits.
- `method-engineer`: implement one backlog item in `track_a/solution/<method>/`. Smoke-test only.
- `experiment-runner`: run train/eval jobs, append to `runs.csv`, make comparison contact sheets.
- `paper-writer`: drafts and refreshes the solution paper from real results only; rerun after each critic verdict.
- `critic`: read runs + related work, decide next step, gatekeep novelty and submission readiness.
- orchestrator (main session): spawns the above in parallel, never blocks on one; asks the human
  only for the realism check (below) or real decisions.

## Rules
- GPUs are shared with other users. Check `nvidia-smi` first; use at most one GPU per job and
  keep <= 40 GB. Run long jobs in background with logs in `track_a/solution/logs/`.
- Never edit `track_a/*.py` evaluator files or `data/`. Never edit VideoX-Fun; subclass/monkeypatch.
- Every evaluated run: name `<method>_<variant>_<n>`; config.json saved; 60-scene eval is the
  comparable unit (only those have real refs). 10-scene runs are smoke only.
- Report numbers from report.json, never estimates. Negative results go into verdicts.md.

## Definition of done
Novelty: critic's verdict in `verdicts.md` states no published method does the same thing
(with the closest 3 papers named, full-text checked). Performance: our best run beats the baseline grid on >=3 of 5
objectives **and** is not a near-copy of the sim (the evaluator can be satisfied by sim + light jitter, which has SAM 0.19, FID ~130;
such a run must lose on human realism and CLIP diversity and must not be submitted), human realism rating >= baseline, results hold
across >=3 seeds (paired bootstrap CIs on per-feature agreement, not the binary failure rate alone), and <= 5 s/scene on L40S.
Then: freeze code, write `track_a/solution/README.md` with how to reproduce, generate the full submission as `TestTeam`.

## Human check-in (async)
Realism artifact page: shows sim | baseline | candidates blind-shuffled; human ranks. Agents do not
wait for it. Requested at most once per ~3 completed iterations.

## Data-leakage rule (added after evaluator analysis)
`data/real/` is the evaluator's reference set (FID/KID, failure-rate comparison). Submitting or blending real photos scores ~perfect
FID and beats every generator on every axis; the evaluator does not catch it. Never put real photos (or crops/blends) in `generated/`.
Any training on real photos (e.g. T1s sim-control -> real-photo pairs) must be reported as such, evaluated ONLY on scenes whose real
photo was held out, and flagged in the paper. Whether training on the 60 provided real photos is permitted by the organisers is to be
confirmed with them before the final submission; until then the final method must also have a variant that uses no real photos.
