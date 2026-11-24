#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ ! -d "$REPO" ]]; then
  echo "[ERROR] Repository not found: $REPO"
else
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"

  SELECTION="$EXP/reports/090_round02_promoted_checkpoint.txt"
  MANIFEST="$EXP/manifests/aggregate_through_round_03.csv"

  OUTPUT="$REPO/checkpoints/corners256_sequential_v2/stage2/round03_2m_e5_best.pt"
  METRICS="$EXP/logs/110_round03_2m_e5_metrics.csv"

  PARENT=""
  problems=0

  if [[ -s "$SELECTION" ]]; then
    parent_reference=$(tr -d '\r\n' < "$SELECTION")

    if [[ "$parent_reference" = /* ]]; then
      PARENT="$parent_reference"
    else
      PARENT="$REPO/$parent_reference"
    fi
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
    echo "CORNERS256 SEQUENTIAL V2 — ROUND 03 TRAINING"
    echo "parent=$PARENT"
    echo "manifest=$MANIFEST"
    echo "epochs=5"
    echo "resume_optimizer=true"
    echo "learning_rate=1e-4 -> 1e-5"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python -m rl.train_dagger \
      --data_dir \
        "$REPO/dagger_data/corners256_sequential_v2" \
      --output_checkpoint "$OUTPUT" \
      --metrics_csv "$METRICS" \
      --manifest "$MANIFEST" \
      --init_checkpoint "$PARENT" \
      --resume_optimizer \
      --epochs 5 \
      --batch_size 512 \
      --validation_batch_size 2048 \
      --validation_fraction 0.10 \
      --seed 2800 \
      --device cuda:0 \
      --learning_rate 1.0e-4 \
      --min_learning_rate 1.0e-5 \
      --gradient_clip_norm 1.0

    training_status=$?

    if [[ "$training_status" -eq 0 ]]; then
      echo "[PASS] Round 03 training process completed"
    else
      echo "[ERROR] Training returned status $training_status"
    fi
  else
    echo "[ERROR] Training was not launched; found $problems problem(s)"
  fi
fi

echo "[DONE] Round 03 2.0M E5 training command finished."
