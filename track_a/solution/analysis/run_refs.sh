#!/bin/bash
# sequential evaluator runs on ref_* submissions (GPU 0)
cd /home/weissl/PycharmProjects/ISU-Challenge-ICSE-2027/track_a
for n in "$@"; do
  grep -qs '"efficiency"' track_a_evaluation_output/ref_$n/report.json && continue
  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/zimage/bin/python evaluator_cli.py evaluate --submission-path submissions/ref_$n \
    --output track_a_evaluation_output/ref_$n/report.json --failure-model qwen --device cuda > solution/logs/ref_$n.log 2>&1
done
