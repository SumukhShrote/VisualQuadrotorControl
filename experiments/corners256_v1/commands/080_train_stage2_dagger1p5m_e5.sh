#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  DATA="$REPO/dagger_data/corners256_v1"
  PARENT="$REPO/checkpoints/corners256_v1/stage1_dagger1m_e10.pt"
  OUTPUT="$REPO/checkpoints/corners256_v1/stage2_dagger1p5m_e5.pt"
  LOG="$REPO/experiments/corners256_v1/logs/080_train_stage2_dagger1p5m_e5.log"

  problems=0

  shard_count=$(
    find "$DATA" \
      -maxdepth 1 \
      -type f \
      -name 'round_*_shard_*.pt' \
      | wc -l
  )

  if [[ "$shard_count" -ne 15 ]]; then
    echo "[ERROR] Expected 15 cumulative shards, found $shard_count"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$PARENT" ]]; then
    echo "[ERROR] Missing parent checkpoint:"
    echo "        $PARENT"
    problems=$((problems + 1))
  fi

  if [[ -e "$OUTPUT" ]]; then
    echo "[ERROR] Output checkpoint already exists:"
    echo "        $OUTPUT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    mkdir -p \
      "$(dirname "$OUTPUT")" \
      "$(dirname "$LOG")"

    echo "=============================================================="
    echo "CORNERS256 V1 — STAGE 2 DAGGER 1.5M"
    echo "parent=$PARENT"
    echo "cumulative_rows=1500000"
    echo "cumulative_shards=15"
    echo "epochs=5"
    echo "global_epochs=61-65"
    echo "batch_size=256"
    echo "learning_rate=1e-5"
    echo "new_optimizer_steps=29250"
    echo "expected_total_optimizer_steps=117000"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python \
      -m rl.train_dagger_corners256_v1 \
      --data_dir "$DATA" \
      --output_checkpoint "$OUTPUT" \
      --resume_checkpoint "$PARENT" \
      --epochs 5 \
      --batch_size 256 \
      --device cuda:0 \
      --seed 4700 \
      2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 && -s "$OUTPUT" ]]; then
      echo "[PASS] Stage 2 training completed"
    else
      echo "[ERROR] Stage 2 training failed"
      echo "        status=$status"
    fi
  else
    echo "[ERROR] Stage 2 training was not launched"
    echo "        problems=$problems"
  fi
else
  echo "[ERROR] Repository not found:"
  echo "        $REPO"
fi
