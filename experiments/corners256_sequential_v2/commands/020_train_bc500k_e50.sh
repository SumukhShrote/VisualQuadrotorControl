#!/usr/bin/env bash

cd /home/sumukh/AerialCatching || return

EXP="/home/sumukh/AerialCatching/experiments/corners256_sequential_v2"

DATA_DIR="/home/sumukh/AerialCatching/dagger_data/corners256_sequential_v2"

MANIFEST="$EXP/manifests/round_00_manifest.csv"

OUTPUT="/home/sumukh/AerialCatching/checkpoints/corners256_sequential_v2/stage2/bc500k_e50_best.pt"

METRICS="$EXP/logs/020_bc500k_e50_metrics.csv"

if [[ ! -s "$MANIFEST" ]]; then
  echo "[ERROR] Missing validated manifest:"
  echo "        $MANIFEST"
else
  PYTHONUNBUFFERED=1 python -m rl.train_dagger \
    --data_dir "$DATA_DIR" \
    --output_checkpoint "$OUTPUT" \
    --metrics_csv "$METRICS" \
    --manifest "$MANIFEST" \
    --epochs 50 \
    --batch_size 512 \
    --validation_batch_size 2048 \
    --validation_fraction 0.10 \
    --seed 2200 \
    --device cuda:0 \
    --learning_rate 1.0e-3 \
    --min_learning_rate 1.0e-5 \
    --gradient_clip_norm 1.0
fi

echo
echo "[DONE] BC500K E50 training command finished."
