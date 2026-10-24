# Disk policy (written 2026-10-18)

`/data` hit 97% (56 GB free) during the exploration sprint. What eats space and what to do about it.

## Deleted (2026-10-18, one path at a time)
| Path | Size | Why unusable |
|---|---|---|
| `/data/weissl/isu_ckpt/t3a_R/` | 15 GB | Trunk (T3) continued on real photos. No better than the LoRA, slower; trunk route dropped (`exp_trained.md`). Numbers stay in `runs.csv`. |
| `/data/weissl/isu_ckpt/t3b_noreal/` | 15 GB | Fresh trunk, clearly worse (FID 200). Dropped. |
| `/data/weissl/isu_ckpt/smoke_t2/` | 0.9 GB | Smoke-test checkpoint. |
| `track_a/solution/checkpoints/t1_smoke2/` | 3 GB | Smoke trunk, only needed to continue T3a. Retrain from `trunk/train.py` if ever needed. |

## Kept on purpose
`isu_ckpt/t1p_a` (best no-real LoRA), `isu_ckpt/t2_a` (real-photo LoRA, held-out only), `teacher_s0.6`, `pairs`, small caches; `submissions/*` (about 85 MB each, needed for contact sheets); every `report.json`.

## Pending: SAM visualisation dirs (about 34 GB, deletion not yet done)
Every finished evaluation leaves `track_a/track_a_evaluation_output/<run>/<run>/sam/` (about 540 MB each, 69 runs). It is a by-product of the evaluator (per-image SAM overlays). All numbers are in `<run>/report.json`; the evaluator regenerates the folder when re-run. Safe to delete for any run whose `report.json` contains `"efficiency"` (finished). Never delete it for a run that is still evaluating.

## Rules so we do not do this again
1. A run's `sam/` folder is deleted right after its `report.json` is recorded with `tools/record_run.py` (do it per run, explicit path, no globs/loops).
2. Smoke-test checkpoints and outputs (`smoke_*`, `*_smoke*`, `dbg`) are deleted as soon as the real run exists.
3. Checkpoint of a dropped method is deleted when the verdict is written to `negative_results.md`; keep only the report row.
4. Training jobs write to `/data/weissl/isu_ckpt/<name>` with a keep-last-1 policy (no per-N-step archives); trunk/LoRA full-state checkpoints are 2-15 GB.
5. Before downloading a model > 10 GB: check `df -h /data` (keep >= 80 GB free) and delete the weights after the experiment is verdicted (record the HF repo id in `exp_*.md` so it can be re-fetched).
6. `/data/weissl/.cache/huggingface` also holds large models from other projects (Kimi-VL 31 GB, Nemotron 30B 33 GB, InternVL3.5-8B 16 GB, deepseek-vl2-tiny 6 GB, gemma-3-4b 8 GB, wav2vec2/speechbrain/utmos): not used by this repo's evaluator (Qwen). Owner decides; not touched by agents.
