#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  REPLAY="$REPO/dagger_data/corners256_v1_beta50a_replay"

  PARENT="$REPO/checkpoints/corners256_v1/stage3_dagger2m_e5.pt"

  OUTPUT="$REPO/checkpoints/corners256_v1/stage4a_beta50_e2.pt"

  LOG="$REPO/experiments/corners256_v1/logs/140_train_stage4a_beta50_e2.log"

  problems=0

  shard_count=$(
    find -L "$REPLAY" \
      -maxdepth 1 \
      -type f \
      -name 'round_*_shard_*.pt' \
      | wc -l
  )

  if [[ "$shard_count" -ne 20 ]]; then
    echo "[ERROR] Expected 20 replay shards, found $shard_count"
    problems=$((problems + 1))
  fi

  for round in 15 16 17 18 19; do
    round_text=$(printf '%02d' "$round")

    if [[ -e "$REPLAY/round_${round_text}_shard_0000.pt" ]]; then
      echo "[ERROR] Excluded round is present: $round_text"
      problems=$((problems + 1))
    fi
  done

  if [[ ! -s "$PARENT" ]]; then
    echo "[ERROR] Missing parent checkpoint: $PARENT"
    problems=$((problems + 1))
  fi

  if [[ -e "$OUTPUT" ]]; then
    echo "[ERROR] Output already exists: $OUTPUT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    echo "=============================================================="
    echo "CORNERS256 V1 — STAGE 4A BETA-50"
    echo "parent=$PARENT"
    echo "replay_rounds=00-14,20-24"
    echo "excluded_rounds=15-19"
    echo "replay_rows=1750000"
    echo "replay_shards=20"
    echo "epochs=2"
    echo "global_epochs=71-72"
    echo "batch_size=512"
    echo "learning_rate=1e-5"
    echo "updates_per_epoch=3410"
    echo "new_optimizer_steps=6820"
    echo "expected_total_optimizer_steps=162820"
    echo "=============================================================="

    PYTHONUNBUFFERED=1 python \
      -m rl.train_dagger_corners256_v1 \
      --data_dir "$REPLAY" \
      --output_checkpoint "$OUTPUT" \
      --resume_checkpoint "$PARENT" \
      --epochs 2 \
      --batch_size 512 \
      --device cuda:0 \
      --seed 5100 \
      2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 && -s "$OUTPUT" ]]; then
      echo "[PASS] Stage 4A training completed"
    else
      echo "[ERROR] Stage 4A training status=$status"
    fi
  else
    echo "[ERROR] Stage 4A training was not launched"
    echo "        problems=$problems"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
