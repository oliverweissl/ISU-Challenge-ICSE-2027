# Track A submission (team TestTeam): FLUX.1-schnell img2img translator

Official format (see `track_a/README.md#official-submission`): `translator.py` defines `create_translator(device)`,
`Dockerfile` starts with `FROM isu-challenge-base`.

- Method: VAE-encode the sim image (960x544), noise to strength 0.6, 4-step FLUX.1-schnell denoise (guidance 0), prompt
  from the scene label (`prompts.py`); output resized back to the input size. Scene layout comes from the sim latent, so
  people/objects keep presence and position; appearance (materials, light, faces) is repainted.
- Weights: `build_weights.py` runs during `docker build` (internet available), downloads `black-forest-labs/FLUX.1-schnell`
  (Apache-2.0), stores the transformer as fp8 (15 GB; layer-wise upcast to bf16 at run time), keeps T5/CLIP/VAE bf16 (9 GB).
  Image = base + ~24 GB (limit 40 GB). No real photos and no network access at run time.
- Resources: ~1.2 s/scene on L40S, 24.5 GB GPU resident (on cards < 30 GB, T5 is parked in RAM between scenes),
  host RAM ~1.3 GB anonymous (limit 14 GB), deterministic seed = crc32(stem).

## Use
```bash
mkdir -p track_a/submissions/TestTeam && cp track_a/submission/* track_a/submissions/TestTeam/   # from repo root
docker build -t isu-challenge-base docker/ && docker build -t isu-track-a track_a/submissions/TestTeam
# then the "Test your submission locally" commands of track_a/README.md, and tools/encrypt_submission.py
```
Local runs without Docker: `ISU_FLUX_WEIGHTS=<dir made by build_weights.py> python track_a/translate_cli.py --translator-dir track_a/submission --data-dir data --output <out>`.
`build_weights.py OUT --local <FLUX.1-schnell snapshot dir>` converts an existing local snapshot instead of downloading.
