cd /home/weissl/PycharmProjects/ISU-Challenge-ICSE-2027/track_a
export CUDA_VISIBLE_DEVICES=1
for n in t1s_k1.0_20 t1s_k0.5_20 base_canny_s1.0_label_hold20 is_s0.7_60_hold20; do
  ~/miniconda3/envs/zimage/bin/python evaluator_cli.py evaluate --submission-path submissions/$n --output track_a_evaluation_output/$n/report.json --failure-model qwen --device cuda > solution/logs/eval_$n.log 2>&1
  echo "$n exit $?" >> solution/logs/t1s_eval.status
done
