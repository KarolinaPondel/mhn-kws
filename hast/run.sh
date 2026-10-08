#!/usr/bin/env bash
# Train + noise-evaluate one (model, paradigm) cell over 5 seeds.
#   ./run.sh <hopfield|transformer> <clean|augment|supcon|nt_xent>      [GPU=0] [SEEDS="0 1 2 3 4"]
# Results: runs/<paradigm>/<model>/seed_<k>/{config.json,history.json,eval_result.json,noise_snr_metrics.csv}
set -euo pipefail
MODEL=${1:?hopfield|transformer}
PARADIGM=${2:?clean|augment|supcon|nt_xent}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
cd "$(dirname "$0")"

for seed in $SEEDS; do
  dir=runs/$PARADIGM/$MODEL/seed_$seed
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$GPU python -W ignore src/train.py --model_type "$MODEL" --paradigm "$PARADIGM" \
    --seed "$seed" --exp_dir "$dir" 2>&1 | tee "$dir/train.log"
  CUDA_VISIBLE_DEVICES=$GPU python -W ignore src/eval_noise.py --exp_dir "$dir" 2>&1 | tee "$dir/eval_noise.log"
done
