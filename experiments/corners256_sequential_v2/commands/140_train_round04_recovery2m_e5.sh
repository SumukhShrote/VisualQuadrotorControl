#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"
  SELECTION="$EXP/reports/090_round02_promoted_checkpoint.txt"
  MANIFEST="$EXP/manifests/aggregate_mainline_through_round_04.csv"

  OUTPUT="$REPO/checkpoints/corners256_sequential_v2/stage2/round04_recovery2m_e5_best.pt"
  METRICS="$EXP/logs/140_round04_recovery2m_e5_metrics.csv"

  problems=0
  PARENT=""

  if [[ -s "$SELECTION" ]]; then
    parent_reference=$(tr -d '\r\n' < "$SELECTION")

    if [[ "$parent_reference" = /* ]]; then
      PARENT="$parent_reference"
    else
      PARENT="$REPO/$parent_reference"
    fi
  else
    echo "[ERROR] Missing parent-selection file:"
    echo "        $SELECTION"
    problems=$((problems + 1))
  fi

  if [[ -z "$PARENT" || ! -s "$PARENT" ]]; then
    echo "[ERROR] Missing parent checkpoint:"
    echo "        $PARENT"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$MANIFEST" ]]; then
    echo "[ERROR] Missing aggregate manifest:"
    echo "        $MANIFEST"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    echo "=============================================================="
    echo "ROUND 04 CONSERVATIVE REPLACEMENT 2.0M TRAINING"
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
      --seed 3000 \
      --device cuda:0 \
      --learning_rate 1.0e-4 \
      --min_learning_rate 1.0e-5 \
      --gradient_clip_norm 1.0

    status=$?

    if [[ "$status" -eq 0 ]]; then
      echo "[PASS] Conservative 2.0M training completed"
    else
      echo "[ERROR] Training returned status $status"
    fi
  else
    echo "[ERROR] Training was not launched; $problems problem(s) found"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
