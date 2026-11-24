#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"

  TEACHER="$REPO/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"
  SELECTION="$EXP/reports/150_round04_promoted_checkpoint.txt"

  DATA_ROOT="$REPO/dagger_data/corners256_sequential_v2/round_06_beta50a_250k"
  LOG_ROOT="$EXP/logs"

  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  names=("level_0_00" "level_0_25" "level_0_50" "level_0_75" "level_1_00")
  seeds=("3300" "3301" "3302" "3303" "3304")

  STUDENT=""
  problems=0

  if [[ -s "$SELECTION" ]]; then
    reference=$(tr -d '\r\n' < "$SELECTION")

    if [[ "$reference" = /* ]]; then
      STUDENT="$reference"
    else
      STUDENT="$REPO/$reference"
    fi
  else
    echo "[ERROR] Missing Round 04 selection: $SELECTION"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$TEACHER" ]]; then
    echo "[ERROR] Missing teacher checkpoint: $TEACHER"
    problems=$((problems + 1))
  fi

  if [[ -z "$STUDENT" || ! -s "$STUDENT" ]]; then
    echo "[ERROR] Missing promoted Round 04 checkpoint: $STUDENT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!levels[@]}"; do
      level="${levels[$index]}"
      name="${names[$index]}"
      seed="${seeds[$index]}"

      output_dir="$DATA_ROOT/$name"
      log_file="$LOG_ROOT/200_round06_beta50a_${name}.log"

      mkdir -p "$output_dir" "$LOG_ROOT"

      existing=$(
        find "$output_dir" \
          -maxdepth 1 \
          -type f \
          -name 'round_06_shard_*.pt' \
          | wc -l
      )

      if [[ "$existing" -eq 1 ]]; then
        echo "[SKIP] Existing shard: $output_dir"
        continue
      fi

      if [[ "$existing" -gt 1 ]]; then
        echo "[ERROR] Multiple shards found: $output_dir"
        problems=$((problems + 1))
        continue
      fi

      echo
      echo "=============================================================="
      echo "ROUND 06 — VERIFIED-SCALE BETA-50 REFINEMENT A"
      echo "level=$level"
      echo "seed=$seed"
      echo "samples=50000"
      echo "teacher_probability=0.5"
      echo "student_probability=0.5"
      echo "controller_hold_steps=25"
      echo "student=$STUDENT"
      echo "=============================================================="

      PYTHONUNBUFFERED=1 python -m rl.collect_dagger \
        --task Isaac-QuadrotorHoverStudentDistill-v0 \
        --num_envs 256 \
        --seed "$seed" \
        --device cuda:0 \
        --teacher_checkpoint "$TEACHER" \
        --student_checkpoint "$STUDENT" \
        --output_dir "$output_dir" \
        --round_index 6 \
        --curriculum_level "$level" \
        --teacher_probability 0.5 \
        --controller_hold_steps 25 \
        --num_samples 50000 \
        --shard_size 50000 \
        2>&1 | tee "$log_file"

      status=${PIPESTATUS[0]}

      produced=$(
        find "$output_dir" \
          -maxdepth 1 \
          -type f \
          -name 'round_06_shard_*.pt' \
          | wc -l
      )

      if [[ "$status" -eq 0 && "$produced" -eq 1 ]]; then
        echo "[PASS] Produced $name"
      else
        echo "[ERROR] $name status=$status shards=$produced"
        problems=$((problems + 1))
      fi
    done
  fi

  total=$(
    find "$DATA_ROOT" \
      -type f \
      -name 'round_06_shard_*.pt' \
      2>/dev/null \
      | wc -l
  )

  echo
  echo "round06_shards=$total"

  if [[ "$total" -eq 5 && "$problems" -eq 0 ]]; then
    echo "[PASS] Round 06 beta-50 250k collection completed"
  else
    echo "[ERROR] Collection completed with $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
