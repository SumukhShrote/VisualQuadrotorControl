#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"
  DATA="$REPO/dagger_data/corners256_sequential_v2"

  SELECTION="$EXP/reports/150_round04_promoted_checkpoint.txt"
  MANIFEST="$EXP/manifests/aggregate_mainline_through_round_05.csv"

  OUTPUT="$REPO/checkpoints/corners256_sequential_v2/stage2/round05_bridge2p5m_e3_best.pt"
  METRICS="$EXP/logs/170_round05_bridge2p5m_e3_metrics.csv"

  PARENT=""
  problems=0

  active_count=$(
    find "$DATA" \
      -type f \
      -name 'round_*_shard_*.pt' \
      | wc -l
  )

  rejected_count=$(
    find "$DATA" \
      -type f \
      -name 'round_03_shard_*.pt' \
      | wc -l
  )

  if [[ "$active_count" -ne 25 ]]; then
    echo "[ERROR] Active shard count=$active_count, expected=25"
    problems=$((problems + 1))
  fi

  if [[ "$rejected_count" -ne 0 ]]; then
    echo "[ERROR] Rejected Round 03 shards remain active"
    problems=$((problems + 1))
  fi

  if [[ -s "$SELECTION" ]]; then
    reference=$(tr -d '\r\n' < "$SELECTION")

    if [[ "$reference" = /* ]]; then
      PARENT="$reference"
    else
      PARENT="$REPO/$reference"
    fi
  else
    echo "[ERROR] Missing parent selection:"
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
    echo "ROUND 05 BRIDGE 2.5M TRAINING"
    echo "parent=$PARENT"
    echo "manifest=$MANIFEST"
    echo "epochs=3"
    echo "resume_optimizer=true"
    echo "learning_rate=7.5e-5 -> 1e-5"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python -m rl.train_dagger \
      --data_dir "$DATA" \
      --output_checkpoint "$OUTPUT" \
      --metrics_csv "$METRICS" \
      --manifest "$MANIFEST" \
      --init_checkpoint "$PARENT" \
      --resume_optimizer \
      --epochs 3 \
      --batch_size 512 \
      --validation_batch_size 2048 \
      --validation_fraction 0.10 \
      --seed 3200 \
      --device cuda:0 \
      --learning_rate 7.5e-5 \
      --min_learning_rate 1.0e-5 \
      --gradient_clip_norm 1.0

    status=$?

    if [[ "$status" -eq 0 ]]; then
      echo "[PASS] Round 05 bridge training completed"
    else
      echo "[ERROR] Training returned status $status"
    fi
  else
    echo "[ERROR] Training was not launched; found $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
