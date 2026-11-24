#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ ! -d "$REPO" ]]; then
  echo "[ERROR] Repository not found: $REPO"
else
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"
  SELECTION="$EXP/reports/030_bc500k_promoted_checkpoint.txt"
  MANIFEST="$EXP/manifests/aggregate_through_round_01.csv"

  OUTPUT="$REPO/checkpoints/corners256_sequential_v2/stage2/round01_1m_e10_best.pt"
  METRICS="$EXP/logs/050_round01_1m_e10_metrics.csv"

  problems=0
  PARENT=""

  if [[ -s "$SELECTION" ]]; then
    parent_relative=$(tr -d '\r\n' < "$SELECTION")
    PARENT="$REPO/$parent_relative"
  else
    echo "[ERROR] Missing promoted checkpoint selection:"
    echo "        $SELECTION"
    problems=$((problems + 1))
  fi

  if [[ -z "$PARENT" || ! -s "$PARENT" ]]; then
    echo "[ERROR] Parent checkpoint not found:"
    echo "        $PARENT"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$MANIFEST" ]]; then
    echo "[ERROR] Aggregate manifest not found:"
    echo "        $MANIFEST"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    echo "=============================================================="
    echo "CORNERS256 SEQUENTIAL V2 — ROUND 01 TRAINING"
    echo "parent=$PARENT"
    echo "manifest=$MANIFEST"
    echo "epochs=10"
    echo "resume_optimizer=true"
    echo "learning_rate=3e-4 -> 1e-5"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python -m rl.train_dagger \
      --data_dir \
        "$REPO/dagger_data/corners256_sequential_v2" \
      --output_checkpoint "$OUTPUT" \
      --metrics_csv "$METRICS" \
      --manifest "$MANIFEST" \
      --init_checkpoint "$PARENT" \
      --resume_optimizer \
      --epochs 10 \
      --batch_size 512 \
      --validation_batch_size 2048 \
      --validation_fraction 0.10 \
      --seed 2400 \
      --device cuda:0 \
      --learning_rate 3.0e-4 \
      --min_learning_rate 1.0e-5 \
      --gradient_clip_norm 1.0
  else
    echo "[ERROR] Training was not launched because validation found"
    echo "        $problems problem(s)."
  fi
fi

echo
echo "[DONE] Round 01 1.0M E10 training command finished."
