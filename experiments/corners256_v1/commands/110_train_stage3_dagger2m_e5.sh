#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  DATA="$REPO/dagger_data/corners256_v1"
  PARENT="$REPO/checkpoints/corners256_v1/stage2_dagger1p5m_e5.pt"
  OUTPUT="$REPO/checkpoints/corners256_v1/stage3_dagger2m_e5.pt"
  LOG="$REPO/experiments/corners256_v1/logs/110_train_stage3_dagger2m_e5.log"

  problems=0

  shard_count=$(
    find "$DATA" -maxdepth 1 -type f \
      -name 'round_*_shard_*.pt' | wc -l
  )

  if [[ "$shard_count" -ne 20 ]]; then
    echo "[ERROR] Expected 20 shards, found $shard_count"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$PARENT" ]]; then
    echo "[ERROR] Missing parent: $PARENT"
    problems=$((problems + 1))
  fi

  if [[ -e "$OUTPUT" ]]; then
    echo "[ERROR] Output already exists: $OUTPUT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    echo "=============================================================="
    echo "CORNERS256 V1 — STAGE 3 DAGGER 2M"
    echo "cumulative_shards=20"
    echo "cumulative_rows=2000000"
    echo "global_epochs=66-70"
    echo "batch_size=256"
    echo "learning_rate=1e-5"
    echo "expected_total_optimizer_steps=156000"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python \
      -m rl.train_dagger_corners256_v1 \
      --data_dir "$DATA" \
      --output_checkpoint "$OUTPUT" \
      --resume_checkpoint "$PARENT" \
      --epochs 5 \
      --batch_size 256 \
      --device cuda:0 \
      --seed 4900 \
      2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 && -s "$OUTPUT" ]]; then
      echo "[PASS] Stage 3 training completed"
    else
      echo "[ERROR] Stage 3 training status=$status"
    fi
  else
    echo "[ERROR] Stage 3 training was not launched"
    echo "        problems=$problems"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
