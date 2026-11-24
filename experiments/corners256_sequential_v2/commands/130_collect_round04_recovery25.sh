#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"

  TEACHER="$REPO/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"
  SELECTION="$EXP/reports/090_round02_promoted_checkpoint.txt"

  DATA_ROOT="$REPO/dagger_data/corners256_sequential_v2/round_04_recovery25"
  LOG_ROOT="$EXP/logs"

  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  names=("level_0_00" "level_0_25" "level_0_50" "level_0_75" "level_1_00")
  seeds=("2900" "2901" "2902" "2903" "2904")

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
    echo "[ERROR] Missing selection: $SELECTION"
    problems=$((problems + 1))
  fi

  [[ -s "$TEACHER" ]] || {
    echo "[ERROR] Missing teacher: $TEACHER"
    problems=$((problems + 1))
  }

  [[ -s "$STUDENT" ]] || {
    echo "[ERROR] Missing student: $STUDENT"
    problems=$((problems + 1))
  }

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!levels[@]}"; do
      level="${levels[$index]}"
      name="${names[$index]}"
      seed="${seeds[$index]}"

      output_dir="$DATA_ROOT/$name"
      log_file="$LOG_ROOT/130_round04_recovery25_${name}.log"

      mkdir -p "$output_dir" "$LOG_ROOT"

      count=$(
        find "$output_dir" \
          -maxdepth 1 \
          -type f \
          -name 'round_04_shard_*.pt' \
          | wc -l
      )

      if [[ "$count" -eq 1 ]]; then
        echo "[SKIP] Existing shard: $output_dir"
        continue
      fi

      if [[ "$count" -gt 1 ]]; then
        echo "[ERROR] Multiple shards: $output_dir"
        problems=$((problems + 1))
        continue
      fi

      echo
      echo "=============================================================="
      echo "ROUND 04 CONSERVATIVE RECOVERY"
      echo "level=$level"
      echo "seed=$seed"
      echo "teacher_probability=0.75"
      echo "student_probability=0.25"
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
        --round_index 4 \
        --curriculum_level "$level" \
        --teacher_probability 0.75 \
        --controller_hold_steps 25 \
        --num_samples 100000 \
        --shard_size 100000 \
        2>&1 | tee "$log_file"

      status=${PIPESTATUS[0]}

      produced=$(
        find "$output_dir" \
          -maxdepth 1 \
          -type f \
          -name 'round_04_shard_*.pt' \
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
      -name 'round_04_shard_*.pt' \
      2>/dev/null \
      | wc -l
  )

  echo
  echo "round04_shard_count=$total"

  if [[ "$total" -eq 5 && "$problems" -eq 0 ]]; then
    echo "[PASS] Round 04 recovery collection completed"
  else
    echo "[ERROR] Collection completed with $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
