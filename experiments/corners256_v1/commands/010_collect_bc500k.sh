#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  NAME="corners256_v1"
  EXP="$REPO/experiments/$NAME"
  DATA="$REPO/dagger_data/$NAME"

  TEACHER="$REPO/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"

  rounds=(0 1 2 3 4)
  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  seeds=(4200 4201 4202 4203 4204)

  problems=0

  mkdir -p "$DATA" "$EXP/logs"

  if [[ ! -s "$TEACHER" ]]; then
    echo "[ERROR] Missing teacher checkpoint: $TEACHER"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!rounds[@]}"; do
      round="${rounds[$index]}"
      level="${levels[$index]}"
      seed="${seeds[$index]}"

      round_text=$(printf '%02d' "$round")
      target="$DATA/round_${round_text}_shard_0000.pt"
      log="$EXP/logs/010_collect_round_${round_text}.log"

      if [[ -s "$target" ]]; then
        echo "[SKIP] Existing shard: $target"
        continue
      fi

      echo
      echo "=============================================================="
      echo "CORNERS256 V1 — BC500K"
      echo "round=$round_text"
      echo "level=$level"
      echo "seed=$seed"
      echo "teacher_probability=1.0"
      echo "samples=100000"
      echo "=============================================================="

      PYTHONUNBUFFERED=1 python -m rl.collect_dagger_corners256_v1 \
        --task Isaac-QuadrotorHoverStudentDistill-v0 \
        --num_envs 256 \
        --seed "$seed" \
        --device cuda:0 \
        --teacher_checkpoint "$TEACHER" \
        --output_dir "$DATA" \
        --round_index "$round" \
        --curriculum_level "$level" \
        --teacher_probability 1.0 \
        --controller_hold_steps 25 \
        --num_samples 100000 \
        --shard_size 100000 \
        2>&1 | tee "$log"

      status=${PIPESTATUS[0]}

      if [[ "$status" -eq 0 && -s "$target" ]]; then
        echo "[PASS] Created $target"
      else
        echo "[ERROR] round=$round_text status=$status"
        problems=$((problems + 1))
      fi
    done
  fi

  count=$(
    find "$DATA" \
      -maxdepth 1 \
      -type f \
      -name 'round_0[0-4]_shard_0000.pt' \
      | wc -l
  )

  echo
  echo "bc_shards=$count"

  if [[ "$count" -eq 5 && "$problems" -eq 0 ]]; then
    echo "[PASS] BC500K collection complete"
  else
    echo "[ERROR] BC500K collection has $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
