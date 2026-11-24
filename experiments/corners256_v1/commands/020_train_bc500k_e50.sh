#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  DATA="$REPO/dagger_data/corners256_v1"
  OUTPUT="$REPO/checkpoints/corners256_v1/stage0_bc500k_e50.pt"
  LOG="$REPO/experiments/corners256_v1/logs/020_train_bc500k_e50.log"

  problems=0

  shard_count=$(
    find "$DATA" \
      -maxdepth 1 \
      -type f \
      -name 'round_0[0-4]_shard_0000.pt' \
      | wc -l
  )

  total_pt_count=$(
    find "$DATA" \
      -maxdepth 1 \
      -type f \
      -name '*.pt' \
      | wc -l
  )

  if [[ "$shard_count" -ne 5 ]]; then
    echo "[ERROR] Expected 5 Stage 0 shards, found $shard_count"
    problems=$((problems + 1))
  fi

  if [[ "$total_pt_count" -ne 5 ]]; then
    echo "[ERROR] Data directory contains $total_pt_count .pt files"
    echo "        Expected exactly 5"
    problems=$((problems + 1))
  fi

  if [[ -e "$OUTPUT" ]]; then
    echo "[ERROR] Output already exists: $OUTPUT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

    echo "=============================================================="
    echo "CORNERS256 V1 — STAGE 0 BC500K"
    echo "initialization=random"
    echo "rows=500000"
    echo "epochs=50"
    echo "batch_size=512"
    echo "complete_batches_per_shard=195"
    echo "updates_per_epoch=975"
    echo "expected_optimizer_steps=48750"
    echo "training_seed=4200"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python \
      -m rl.train_dagger_corners256_v1 \
      --data_dir "$DATA" \
      --output_checkpoint "$OUTPUT" \
      --epochs 50 \
      --batch_size 512 \
      --device cuda:0 \
      --seed 4200 \
      2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 && -s "$OUTPUT" ]]; then
      echo "[PASS] Stage 0 BC500K training completed"
    else
      echo "[ERROR] Stage 0 training status=$status"
    fi
  else
    echo "[ERROR] Stage 0 training was not launched"
    echo "        problems=$problems"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
