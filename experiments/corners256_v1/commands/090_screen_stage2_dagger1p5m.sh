#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_v1"

  CHECKPOINT="$REPO/checkpoints/corners256_v1/stage2_dagger1p5m_e5.pt"

  GLOBAL_ROOT="$REPO/eval_logs/initial_episode_records"

  RECORD_ROOT="$REPO/eval_logs/corners256_v1/stage2_dagger1p5m_screen/initial_episode_records"

  LOG_ROOT="$EXP/logs/stage2_dagger1p5m_screen"

  mkdir -p \
    "$GLOBAL_ROOT" \
    "$RECORD_ROOT" \
    "$LOG_ROOT"

  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  names=("l000" "l025" "l050" "l075" "l100")

  seed="4300"
  problems=0

  if [[ ! -s "$CHECKPOINT" ]]; then
    echo "[ERROR] Missing checkpoint: $CHECKPOINT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!levels[@]}"; do
      level="${levels[$index]}"
      name="${names[$index]}"

      prefix="corners256_v1_stage2_${name}_s${seed}"

      destination="$RECORD_ROOT/${prefix}_initial_episodes.pt"
      log_file="$LOG_ROOT/${prefix}.log"

      if [[ -s "$destination" ]]; then
        echo "[SKIP] Existing evaluation: $destination"
        continue
      fi

      source_count=$(
        find "$GLOBAL_ROOT" \
          -maxdepth 1 \
          -type f \
          -name "${prefix}*_initial_episodes.pt" \
          | wc -l
      )

      if [[ "$source_count" -eq 1 ]]; then
        source=$(
          find "$GLOBAL_ROOT" \
            -maxdepth 1 \
            -type f \
            -name "${prefix}*_initial_episodes.pt" \
            -print \
            -quit
        )

        mv "$source" "$destination"
        echo "[RECOVERED] $destination"
        continue
      fi

      if [[ "$source_count" -gt 1 ]]; then
        echo "[ERROR] Multiple global records for $prefix"
        problems=$((problems + 1))
        continue
      fi

      echo
      echo "=============================================================="
      echo "CORNERS256 V1 — STAGE 2 SCREEN"
      echo "level=$level"
      echo "seed=$seed"
      echo "num_envs=100"
      echo "checkpoint=$CHECKPOINT"
      echo "=============================================================="

      PYTHONUNBUFFERED=1 python -m rl.eval_rslrl \
        --task Isaac-QuadrotorHoverAsymmetricPPO-v0 \
        --student_checkpoint "$CHECKPOINT" \
        --num_envs 100 \
        --seed "$seed" \
        --curriculum_level "$level" \
        --save_prefix "$prefix" \
        --device cuda:0 \
        2>&1 | tee "$log_file"

      status=${PIPESTATUS[0]}

      source_count=$(
        find "$GLOBAL_ROOT" \
          -maxdepth 1 \
          -type f \
          -name "${prefix}*_initial_episodes.pt" \
          | wc -l
      )

      if [[ "$status" -eq 0 && "$source_count" -eq 1 ]]; then
        source=$(
          find "$GLOBAL_ROOT" \
            -maxdepth 1 \
            -type f \
            -name "${prefix}*_initial_episodes.pt" \
            -print \
            -quit
        )

        mv "$source" "$destination"
        echo "[MOVED] $destination"
      else
        echo "[ERROR] Evaluation failed:"
        echo "        level=$level"
        echo "        status=$status"
        echo "        source_count=$source_count"
        problems=$((problems + 1))
      fi
    done
  fi

  count=$(
    find "$RECORD_ROOT" \
      -maxdepth 1 \
      -type f \
      -name '*.pt' \
      | wc -l
  )

  echo
  echo "record_count=$count"

  if [[ "$count" -eq 5 && "$problems" -eq 0 ]]; then
    echo "[PASS] Stage 2 screen completed"
  else
    echo "[ERROR] Stage 2 screen has $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
