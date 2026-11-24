#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"
  DATA="$REPO/dagger_data/corners256_sequential_v2"

  SELECTION="$EXP/reports/150_round04_promoted_checkpoint.txt"
  MANIFEST="$EXP/manifests/aggregate_mainline_through_round_06.csv"

  OUTPUT="$REPO/checkpoints/corners256_sequential_v2/stage2/round06_beta50a_2p25m_e2_best.pt"
  METRICS="$EXP/logs/210_round06_beta50a_2p25m_e2_metrics.csv"

  PARENT=""
  problems=0

  all_pt_count=$(
    find "$DATA" \
      -type f \
      -name '*.pt' \
      | wc -l
  )

  shard_count=$(
    find "$DATA" \
      -type f \
      -name 'round_*_shard_*.pt' \
      | wc -l
  )

  active_rounds=$(
    find "$DATA" \
      -type f \
      -name 'round_*_shard_*.pt' \
      -printf '%f\n' \
      | sed -n 's/^round_\([0-9][0-9]\)_shard_.*/\1/p' \
      | sort -u \
      | tr '\n' ' '
  )

  if [[ "$all_pt_count" -ne 25 ]]; then
    echo "[ERROR] All .pt count=$all_pt_count, expected=25"
    problems=$((problems + 1))
  fi

  if [[ "$shard_count" -ne 25 ]]; then
    echo "[ERROR] Shard count=$shard_count, expected=25"
    problems=$((problems + 1))
  fi

  if [[ "$active_rounds" != "00 01 02 04 06 " ]]; then
    echo "[ERROR] Active rounds='$active_rounds'"
    echo "        Expected='00 01 02 04 06 '"
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
    echo "ROUND 06 — BETA-50 A, 2.25M, TWO EPOCHS"
    echo "parent=$PARENT"
    echo "manifest=$MANIFEST"
    echo "epochs=2"
    echo "resume_optimizer=true"
    echo "learning_rate=5e-5 -> 1e-5"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python -m rl.train_dagger \
      --data_dir "$DATA" \
      --output_checkpoint "$OUTPUT" \
      --metrics_csv "$METRICS" \
      --manifest "$MANIFEST" \
      --init_checkpoint "$PARENT" \
      --resume_optimizer \
      --epochs 2 \
      --batch_size 512 \
      --validation_batch_size 2048 \
      --validation_fraction 0.10 \
      --seed 3400 \
      --device cuda:0 \
      --learning_rate 5.0e-5 \
      --min_learning_rate 1.0e-5 \
      --gradient_clip_norm 1.0

    status=$?

    if [[ "$status" -eq 0 ]]; then
      echo "[PASS] Round 06 two-epoch training completed"
    else
      echo "[ERROR] Training returned status $status"
    fi
  else
    echo "[ERROR] Training was not launched; found $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
