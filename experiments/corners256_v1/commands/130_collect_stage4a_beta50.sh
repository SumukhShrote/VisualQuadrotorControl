#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_v1"
  RAW="$REPO/dagger_data/corners256_v1_beta50a_raw"

  TEACHER="$REPO/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"
  STUDENT="$REPO/checkpoints/corners256_v1/stage3_dagger2m_e5.pt"

  rounds=(20 21 22 23 24)
  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  seeds=(5000 5001 5002 5003 5004)

  problems=0

  mkdir -p "$RAW" "$EXP/logs"

  if [[ ! -s "$TEACHER" ]]; then
    echo "[ERROR] Missing teacher: $TEACHER"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$STUDENT" ]]; then
    echo "[ERROR] Missing Stage 3 checkpoint: $STUDENT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!rounds[@]}"; do
      round="${rounds[$index]}"
      level="${levels[$index]}"
      seed="${seeds[$index]}"

      round_text=$(printf '%02d' "$round")
      target="$RAW/round_${round_text}_shard_0000.pt"
      log="$EXP/logs/130_collect_round_${round_text}.log"

      if [[ -s "$target" ]]; then
        echo "[SKIP] Existing shard: $target"
        continue
      fi

      echo
      echo "=============================================================="
      echo "CORNERS256 V1 — STAGE 4A BETA-50"
      echo "round=$round_text"
      echo "level=$level"
      echo "seed=$seed"
      echo "teacher_probability=0.50"
      echo "student_probability=0.50"
      echo "samples=50000"
      echo "parent=$STUDENT"
      echo "=============================================================="

      PYTHONUNBUFFERED=1 python \
        -m rl.collect_dagger_corners256_v1 \
        --task Isaac-QuadrotorHoverStudentDistill-v0 \
        --num_envs 256 \
        --seed "$seed" \
        --device cuda:0 \
        --teacher_checkpoint "$TEACHER" \
        --student_checkpoint "$STUDENT" \
        --output_dir "$RAW" \
        --round_index "$round" \
        --curriculum_level "$level" \
        --teacher_probability 0.50 \
        --controller_hold_steps 25 \
        --num_samples 50000 \
        --shard_size 50000 \
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
    find "$RAW" \
      -maxdepth 1 \
      -type f \
      -name 'round_2[0-4]_shard_0000.pt' \
      | wc -l
  )

  echo
  echo "stage4a_new_shards=$count"

  if [[ "$count" -eq 5 && "$problems" -eq 0 ]]; then
    echo "[PASS] Stage 4A collection complete"
  else
    echo "[ERROR] Stage 4A collection problems=$problems"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
