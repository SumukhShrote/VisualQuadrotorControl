#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  REPLAY="$REPO/dagger_data/corners256_v1_stage0_only"

  OUTPUT_ROOT="$REPO/checkpoints/corners256_v1/diagnostics/batch256_seed_sweep"

  LOG_ROOT="$REPO/experiments/corners256_v1/logs/batch256_seed_sweep"

  seeds=(4000 4001 4002 4003 4004 4005 4006 4007)

  mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

  shard_count=$(
    find -L "$REPLAY" \
      -maxdepth 1 \
      -type f \
      -name 'round_0[0-4]_shard_0000.pt' \
      | wc -l
  )

  problems=0

  if [[ "$shard_count" -ne 5 ]]; then
    echo "[ERROR] Expected five Stage 0 shards, found $shard_count"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for seed in "${seeds[@]}"; do
      output="$OUTPUT_ROOT/seed_${seed}_e3.pt"
      log="$LOG_ROOT/seed_${seed}_e3.log"

      if [[ -s "$output" && -s "$log" ]]; then
        echo "[SKIP] Existing completed-looking probe: seed=$seed"
        continue
      fi

      rm -f "$output" "$log"

      echo
      echo "=============================================================="
      echo "BATCH-256 STAGE 0 SEED PROBE"
      echo "seed=$seed"
      echo "epochs=3"
      echo "batch_size=256"
      echo "expected_optimizer_steps=5850"
      echo "=============================================================="

      PYTHONUNBUFFERED=1 python \
        -m rl.train_dagger_corners256_v1 \
        --data_dir "$REPLAY" \
        --output_checkpoint "$output" \
        --epochs 3 \
        --batch_size 256 \
        --device cuda:0 \
        --seed "$seed" \
        2>&1 | tee "$log"

      status=${PIPESTATUS[0]}

      if [[ "$status" -eq 0 && -s "$output" ]]; then
        echo "[PASS] Probe completed: seed=$seed"
      else
        echo "[ERROR] Probe failed: seed=$seed status=$status"
        problems=$((problems + 1))
      fi
    done
  fi

  echo
  echo "seed_sweep_problems=$problems"
else
  echo "[ERROR] Repository not found: $REPO"
fi
