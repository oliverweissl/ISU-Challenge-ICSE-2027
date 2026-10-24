cd /home/weissl/PycharmProjects/ISU-Challenge-ICSE-2027/track_a/solution
PY=~/miniconda3/envs/zimage/bin/python
export CUDA_VISIBLE_DEVICES=0
$PY -W ignore -m hynea.optimize --name h1ctl_k2_10 --K 2 --res 256 --steps 30 --check-every 10 --extra 99 --strength 0.5 --vlm-scale 0.5 --lr 5e-5 --lam-fid 3 --lam-st 0 --device cuda:0 > logs/h1ctl_k2_10.log 2>&1
echo "h1ctl_k2_10 gen exit $?" >> logs/chain_h1ctl.status
cd .. && $PY -W ignore evaluator_cli.py evaluate --submission-path submissions/h1ctl_k2_10 --output track_a_evaluation_output/h1ctl_k2_10/report.json --failure-model qwen --device cuda > solution/logs/eval_h1ctl_k2_10.log 2>&1
echo "h1ctl_k2_10 eval exit $?" >> solution/logs/chain_h1ctl.status
cd solution && $PY tools/record_run.py h1ctl_k2_10 --note "steering-off control for h1_k2_10 (n=10 holdout)" >> logs/chain_h1ctl.status 2>&1
