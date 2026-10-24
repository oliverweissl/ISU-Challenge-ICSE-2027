cd /home/weissl/PycharmProjects/ISU-Challenge-ICSE-2027/track_a/solution
for s in 0.7 0.85 0.5; do
 CUDA_VISIBLE_DEVICES=1 ~/miniconda3/envs/zimage/bin/python -W ignore run_experiment.py --name is_s${s}_60 --method init_sim --init-strength $s --fusion-mode sum --prompt label --n 60 --eval > logs/is_s${s}_60.log 2>&1
 echo "is_s${s}_60 exit $?" >> logs/chain_is.status
done
