#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"

  TEACHER="$REPO/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"

  SELECTION="$EXP/reports/150_round04_promoted_checkpoint.txt"

  DATA_ROOT="$REPO/dagger_data/corners256_sequential_v2/round_05_bridge375"
  LOG_ROOT="$EXP/logs"

  levels=(
    "0.00"
    "0.25"
    "0.50"
    "0.75"
    "1.00"
  )

  level_names=(
    "level_0_00"
    "level_0_25"
    "level_0_50"
    "level_0_75"
    "level_1_00"
  )

  seeds=(
    "3100"
    "3101"
    "3102"
    "3103"
    "3104"
  )

  problems=0
  STUDENT=""

  if [[ -s "$SELECTION" ]]; then
    reference=$(tr -d '\r\n' < "$SELECTION")

    if [[ "$reference" = /* ]]; then
      STUDENT="$reference"
    else
      STUDENT="$REPO/$reference"
    fi
  else
    echo "[ERROR] Missing promoted-checkpoint selection:"
    echo "        $SELECTION"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$TEACHER" ]]; then
    echo "[ERROR] Missing teacher checkpoint:"
    echo "        $TEACHER"
    problems=$((problems + 1))
  fi

  if [[ -z "$STUDENT" || ! -s "$STUDENT" ]]; then
    echo "[ERROR] Missing promoted student checkpoint:"
    echo "        $STUDENT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    echo "teacher=$TEACHER"
    echo "student=$STUDENT"

    for index in "${!levels[@]}"; do
      level="${levels[$index]}"
      level_name="${level_names[$index]}"
      seed="${seeds[$index]}"

      output_dir="$DATA_ROOT/$level_name"
      log_file="$LOG_ROOT/160_round05_bridge375_${level_name}.log"

      mkdir -p "$output_dir" "$LOG_ROOT"

      existing_count=$(
        find "$output_dir" \
          -maxdepth 1 \
          -type f \
          -name 'round_05_shard_*.pt' \
          | wc -l
      )

      if [[ "$existing_count" -eq 1 ]]; then
        echo "[SKIP] Existing shard: $output_dir"
        continue
      fi

      if [[ "$existing_count" -gt 1 ]]; then
        echo "[ERROR] Multiple shards found: $output_dir"
        problems=$((problems + 1))
        continue
      fi

      echo
      echo "================================================================"
      echo "ROUND 05 — 37.5% STUDENT BRIDGE"
      echo "curriculum_level=$level"
      echo "seed=$seed"
      echo "teacher_probability=0.625"
      echo "student_probability=0.375"
      echo "controller_hold_steps=25"
      echo "student=$STUDENT"
      echo "================================================================"

      PYTHONUNBUFFERED=1 python -m rl.collect_dagger \
        --task Isaac-QuadrotorHoverStudentDistill-v0 \
        --num_envs 256 \
        --seed "$seed" \
        --device cuda:0 \
        --teacher_checkpoint "$TEACHER" \
        --student_checkpoint "$STUDENT" \
        --output_dir "$output_dir" \
        --round_index 5 \
        --curriculum_level "$level" \
        --teacher_probability 0.625 \
        --controller_hold_steps 25 \
        --num_samples 100000 \
        --shard_size 100000 \
        2>&1 | tee "$log_file"

      collector_status=${PIPESTATUS[0]}

      produced_count=$(
        find "$output_dir" \
          -maxdepth 1 \
          -type f \
          -name 'round_05_shard_*.pt' \
          | wc -l
      )

      if [[ "$collector_status" -eq 0 && "$produced_count" -eq 1 ]]; then
        echo "[PASS] Produced $level_name"
      else
        echo "[ERROR] Collection problem for $level_name"
        echo "        collector_status=$collector_status"
        echo "        shard_count=$produced_count"
        problems=$((problems + 1))
      fi
    done
  fi

  total_count=$(
    find "$DATA_ROOT" \
      -type f \
      -name 'round_05_shard_*.pt' \
      2>/dev/null \
      | wc -l
  )

  echo
  echo "round05_shard_count=$total_count"

  if [[ "$total_count" -eq 5 && "$problems" -eq 0 ]]; then
    echo "[PASS] Round 05 bridge collection completed"
  else
    echo "[ERROR] Round 05 completed with $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
