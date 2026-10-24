#!/usr/bin/env bash
# Re-run evaluation for finished generations whose evaluator crashed (GPU OOM on the shared machine).
# usage: tools/eval_queue.sh run1 run2 ...   (run from track_a/solution)
PY=~/miniconda3/envs/zimage/bin/python
ROOT=$(cd ../.. && pwd)
for run in "$@"; do
  rep=$ROOT/track_a/track_a_evaluation_output/$run/report.json
  # wait for the generating job to finish (chain writes "<run> exit N")
  until grep -q "^$run exit" logs/chain_a.status 2>/dev/null; do sleep 20; done
  for try in 1 2 3 4 5 6; do
    grep -q '"efficiency"' "$rep" 2>/dev/null && break
    gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
    free=${gpu#*, }; idx=${gpu%%,*}
    if [ "$free" -lt 14000 ]; then sleep 120; continue; fi
    (cd ../ && CUDA_VISIBLE_DEVICES=$idx $PY -W ignore evaluator_cli.py evaluate --submission-path submissions/$run \
        --output track_a_evaluation_output/$run/report.json --failure-model qwen --device cuda > solution/logs/eval_$run.log 2>&1)
  done
  grep -q '"efficiency"' "$rep" 2>/dev/null && echo "$run eval ok" || echo "$run eval FAILED"
done
