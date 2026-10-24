#!/usr/bin/env bash
# Baseline grid: single ControlNet-Union control x scale, label prompts, 60 scenes (those with real refs).
PY=~/miniconda3/envs/zimage/bin/python
for c in canny depth; do for s in 0.8 1.0; do
  CUDA_VISIBLE_DEVICES=${GPU:-1} $PY -W ignore run_experiment.py --name base_${c}_s${s}_label --control $c --scale $s --prompt label --n 60 --eval
done; done
