#!/bin/bash
cd /home/weissl/PycharmProjects/ISU-Challenge-ICSE-2027/track_a
for f in track_a_evaluation_output/*/report.json; do
  n=$(basename $(dirname $f)); [ $n = dbg ] && continue
  grep -q '"efficiency"' $f || continue
  [ -d submissions/$n/generated ] || continue
  [ -f solution/analysis/proxy_out/$n.json ] && continue
  CUDA_VISIBLE_DEVICES=0 ~/miniconda3/envs/zimage/bin/python solution/analysis/fast_proxy.py submissions/$n --out solution/analysis/proxy_out/$n.json > /dev/null 2>&1
done
