#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  DATA="$REPO/dagger_data/corners256_v1"
  PARENT="$REPO/checkpoints/corners256_v1/stage0_bc500k_e50.pt"
  OUTPUT="$REPO/checkpoints/corners256_v1/stage1_dagger1m_e10.pt"
  LOG="$REPO/experiments/corners256_v1/logs/050_train_stage1_dagger1m_e10.log"

  problems=0

  shard_count=$(
    find "$DATA" \
      -maxdepth 1 \
      -type f \
      -name 'round_*_shard_*.pt' \
      | wc -l
  )

  if [[ "$shard_count" -ne 10 ]]; then
    echo "[ERROR] Expected 10 cumulative shards, found $shard_count"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$PARENT" ]]; then
    echo "[ERROR] Missing parent checkpoint: $PARENT"
    problems=$((problems + 1))
  fi

  if [[ -e "$OUTPUT" ]]; then
    echo "[ERROR] Output already exists: $OUTPUT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

    echo "=============================================================="
    echo "CORNERS256 V1 — STAGE 1 DAGGER 1M"
    echo "parent=$PARENT"
    echo "cumulative_rows=1000000"
    echo "cumulative_shards=10"
    echo "epochs=10"
    echo "global_epochs=51-60"
    echo "batch_size=256"
    echo "learning_rate=1e-5"
    echo "new_optimizer_steps=39000"
    echo "expected_total_optimizer_steps=87750"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python \
      -m rl.train_dagger_corners256_v1 \
      --data_dir "$DATA" \
      --output_checkpoint "$OUTPUT" \
      --resume_checkpoint "$PARENT" \
      --epochs 10 \
      --batch_size 256 \
      --device cuda:0 \
      --seed 4500 \
      2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 && -s "$OUTPUT" ]]; then
      echo "[PASS] Stage 1 training completed"
    else
      echo "[ERROR] Stage 1 training status=$status"
    fi
  else
    echo "[ERROR] Stage 1 training was not launched"
    echo "        problems=$problems"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
