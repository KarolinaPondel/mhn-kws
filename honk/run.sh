#!/usr/bin/env bash
# Train + noise-evaluate one (aggregation, paradigm) cell over 5 seeds.
#   ./run.sh <baseline|att_one_step|att_pure|mhn|mhnpl> <clean|augment|nt_xent|supcon>   [GPU=0] [SEEDS="0 1 2 3 4"] [STEPS=1]
# Results: runs/<paradigm>/<aggregation>/seed_<k>/{config.json,result.json,model.pt,noise_snr_accuracy.csv}
set -euo pipefail
AGG=${1:?baseline|att_one_step|att_pure|mhn|mhnpl}
PARADIGM=${2:?clean|augment|nt_xent|supcon}
GPU=${GPU:-0}
SEEDS=${SEEDS:-"0 1 2 3 4"}
STEPS=${STEPS:-1}
cd "$(dirname "$0")"

for seed in $SEEDS; do
  dir=runs/$PARADIGM/$AGG/seed_$seed
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$GPU python -W ignore src/train.py --aggregation "$AGG" --paradigm "$PARADIGM" \
    --steps "$STEPS" --seed "$seed" --exp_dir "$dir" 2>&1 | tee "$dir/train.log"
  CUDA_VISIBLE_DEVICES=$GPU python -W ignore src/eval_noise.py --exp_dir "$dir" 2>&1 | tee "$dir/eval_noise.log"
done
