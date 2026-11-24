#!/usr/bin/env bash

cd /home/sumukh/AerialCatching || return

TEACHER="/home/sumukh/AerialCatching/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"

DATA_ROOT="/home/sumukh/AerialCatching/dagger_data/corners256_sequential_v2/round_00_bc500k"

LOG_ROOT="/home/sumukh/AerialCatching/experiments/corners256_sequential_v2/logs"

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
  "2100"
  "2101"
  "2102"
  "2103"
  "2104"
)

if [[ ! -s "$TEACHER" ]]; then
  echo "[ERROR] Missing teacher checkpoint:"
  echo "        $TEACHER"
else
  for index in "${!levels[@]}"; do
    level="${levels[$index]}"
    level_name="${level_names[$index]}"
    seed="${seeds[$index]}"

    output_dir="$DATA_ROOT/$level_name"
    log_file="$LOG_ROOT/010_bc500k_${level_name}.log"

    mkdir -p "$output_dir" "$LOG_ROOT"

    existing_count=$(
      find "$output_dir" \
        -maxdepth 1 \
        -type f \
        -name 'round_00_shard_*.pt' \
        | wc -l
    )

    if [[ "$existing_count" -gt 0 ]]; then
      echo "[SKIP] Existing shard found: $output_dir"
      continue
    fi

    echo
    echo "================================================================"
    echo "CORNERS256 SEQUENTIAL V2 — BC500K"
    echo "level=$level"
    echo "seed=$seed"
    echo "teacher_probability=1.0"
    echo "controller_hold_steps=25"
    echo "================================================================"

    PYTHONUNBUFFERED=1 python -m rl.collect_dagger \
      --task Isaac-QuadrotorHoverStudentDistill-v0 \
      --num_envs 256 \
      --seed "$seed" \
      --device cuda:0 \
      --teacher_checkpoint "$TEACHER" \
      --output_dir "$output_dir" \
      --round_index 0 \
      --curriculum_level "$level" \
      --teacher_probability 1.0 \
      --controller_hold_steps 25 \
      --num_samples 100000 \
      --shard_size 100000 \
      2>&1 | tee "$log_file"
  done
fi

echo
echo "[DONE] BC500K collection command finished."
