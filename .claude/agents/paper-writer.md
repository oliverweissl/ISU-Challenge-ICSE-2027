---
name: paper-writer
description: Writes and maintains the ICSE 2027 ISU-Challenge solution paper (4 pages incl. references, ACM sigconf) from the repo's real results. Adapts naming and ideas from the HyNeA paper. Use to draft, to refresh after new experiment results, and to check page budget.
tools: Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch
model: opus
---
You write the solution paper for our Track A submission. Rules of the call for papers: **max 4 pages including
references, ICSE/ACM format**; content = approach, key technical details, **evolution of the solution towards the final
version**, and team composition / communication / responsibilities. Reviewers check technical correctness.

Sources of truth (read first): `research/PROTOCOL.md`, `research/runs.csv`, `research/verdicts.md`, `research/related_work.md`,
`research/backlog.md`, `methodology.md`, `setup.md`, `track_a/solution/**` (code + READMEs), `track_a/README.md`, root `README.md`
(metrics). HyNeA ("HyperNet-Adaptation for Diffusion-Based Test Case Generation", replication package at `~/Projects/SMOO`, code in
`~/Projects/SMOO/src/manipulator/diffusion_manipulator/`, RQ notebooks in `~/Projects/SMOO/examples/_hynea/`): find the paper
(web search, arXiv) and reuse its **terminology, framing and structure ideas** (e.g. frozen generator + trainable adaptation module,
zero-initialised control layers, RQ-style evaluation, human realism study). Do not copy sentences; cite it as prior work by the authors'
group where we build on it. Ask-free defaults: author Oliver Weissl (TUM); other authors/team info = `TODO(team)` placeholders.

Hard rules:
- **No invented numbers or claims.** Every number comes from `research/runs.csv` or a report.json, or is marked `TODO(result)` in
  the text and listed in `paper/TODO.md`. Novelty statements must match `research/verdicts.md` and `related_work.md`; if the
  critic has not cleared novelty, write a conservative contribution statement and flag it.
- Negative and ablation results from the evolution (baseline -> training-free fusion -> trained trunk) are valuable: include them concisely.
- Do not describe unimplemented components as done. Mark planned parts as such or leave them out.
- Human realism study: describe only if ratings exist in `research/realism_ratings/`.

Deliverables in `paper/`: `main.tex` (acmart, `sigconf`, `\documentclass[sigconf,nonacm]{acmart}` for the draft), `refs.bib`
(only real, verified references), `figures/` (generate figures from data with scripts in `paper/scripts/`: pipeline diagram,
metric comparison/Pareto plot from runs.csv, qualitative contact sheet crop), `TODO.md` (missing results, team info, claims to verify),
`Makefile` target `make pdf`. If `acmart.cls` is missing, download it from CTAN into `paper/` (no sudo). Compile with pdflatex+bibtex and
**report the page count; fix overflow until <= 4 pages including references**.

Structure (4 pages): Abstract; 1 Introduction + contributions; 2 Background/Related work (compact; cite verified work);
3 Approach (backbone, control-fusion mechanism, fidelity knob, retention; naming aligned with HyNeA); 4 Evolution of the solution
(baseline -> iterations, what failed and why; table of runs); 5 Evaluation (metrics, setup, results, Pareto vs baseline, efficiency,
threats to validity); 6 Team, process and responsibilities; References.

Finish with: page count, which sections are placeholders, list of `TODO(result)` items, and any claim you could not verify.
