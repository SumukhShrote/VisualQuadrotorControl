#!/usr/bin/env bash

cd /home/sumukh/AerialCatching

EXP="/home/sumukh/AerialCatching/experiments/corners256_sequential_v2"

GLOBAL_RECORD_ROOT="/home/sumukh/AerialCatching/eval_logs/initial_episode_records"

SCREEN_ROOT="/home/sumukh/AerialCatching/eval_logs/corners256_sequential_v2/bc500k_e50_screen"

LOG_ROOT="$EXP/logs/bc500k_e50_screen"

mkdir -p \
  "$GLOBAL_RECORD_ROOT" \
  "$SCREEN_ROOT" \
  "$LOG_ROOT"

candidate_names=(
  "best_e49"
  "last_e50"
)

candidate_checkpoints=(
  "/home/sumukh/AerialCatching/checkpoints/corners256_sequential_v2/stage2/bc500k_e50_best.pt"
  "/home/sumukh/AerialCatching/checkpoints/corners256_sequential_v2/stage2/bc500k_e50_best_last.pt"
)

levels=(
  "0.00"
  "0.25"
  "0.50"
  "0.75"
  "1.00"
)

level_names=(
  "l000"
  "l025"
  "l050"
  "l075"
  "l100"
)

seed="3000"
failures=0

for candidate_index in "${!candidate_names[@]}"; do
  candidate_name="${candidate_names[$candidate_index]}"
  checkpoint="${candidate_checkpoints[$candidate_index]}"

  record_root="$SCREEN_ROOT/$candidate_name/initial_episode_records"
  candidate_log_root="$LOG_ROOT/$candidate_name"

  mkdir -p \
    "$record_root" \
    "$candidate_log_root"

  if [[ ! -s "$checkpoint" ]]; then
    echo "[ERROR] Missing checkpoint: $checkpoint"
    failures=$((failures + 1))
    continue
  fi

  for level_index in "${!levels[@]}"; do
    level="${levels[$level_index]}"
    level_name="${level_names[$level_index]}"

    prefix="c256_seqv2_bc500k_${candidate_name}_${level_name}_s${seed}"

    destination="$record_root/${prefix}_initial_episodes.pt"
    log_file="$candidate_log_root/${prefix}.log"

    if [[ -s "$destination" ]]; then
      echo "[SKIP] Existing record: $destination"
      continue
    fi

    source_record=$(
      find "$GLOBAL_RECORD_ROOT" \
        -maxdepth 1 \
        -type f \
        -name "${prefix}*_initial_episodes.pt" \
        -print \
        -quit
    )

    if [[ -n "$source_record" ]]; then
      mv "$source_record" "$destination"
      echo "[RECOVERED] $destination"
      continue
    fi

    echo
    echo "========================================================================"
    echo "BC500K E50 CLOSED-LOOP SCREEN"
    echo "candidate=$candidate_name"
    echo "checkpoint=$checkpoint"
    echo "level=$level"
    echo "seed=$seed"
    echo "========================================================================"

    PYTHONUNBUFFERED=1 python -m rl.eval_rslrl \
      --task Isaac-QuadrotorHoverAsymmetricPPO-v0 \
      --student_checkpoint "$checkpoint" \
      --num_envs 100 \
      --seed "$seed" \
      --curriculum_level "$level" \
      --save_prefix "$prefix" \
      --device cuda:0 \
      2>&1 | tee "$log_file"

    source_record=$(
      find "$GLOBAL_RECORD_ROOT" \
        -maxdepth 1 \
        -type f \
        -name "${prefix}*_initial_episodes.pt" \
        -print \
        -quit
    )

    if [[ -n "$source_record" ]]; then
      mv "$source_record" "$destination"
      echo "[MOVED] $destination"
    else
      echo "[ERROR] Evaluator record not found for $prefix"
      failures=$((failures + 1))
    fi
  done

  count=$(
    find "$record_root" \
      -maxdepth 1 \
      -type f \
      -name '*.pt' \
      | wc -l
  )

  echo "$candidate_name record_count=$count"

  if [[ "$count" -ne 5 ]]; then
    echo "[ERROR] Expected five records for $candidate_name"
    failures=$((failures + 1))
  fi
done

echo

if [[ "$failures" -eq 0 ]]; then
  echo "[PASS] Both BC500K checkpoints completed the screening matrix."
else
  echo "[ERROR] Screening completed with $failures problem(s)."
fi
