#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_v1"
  DATA="$REPO/dagger_data/corners256_v1"

  TEACHER="$REPO/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"
  STUDENT="$REPO/checkpoints/corners256_v1/stage0_bc500k_e50.pt"

  rounds=(5 6 7 8 9)
  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  seeds=(4400 4401 4402 4403 4404)

  problems=0

  mkdir -p "$DATA" "$EXP/logs"

  if [[ ! -s "$TEACHER" ]]; then
    echo "[ERROR] Missing teacher: $TEACHER"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$STUDENT" ]]; then
    echo "[ERROR] Missing Stage 0 student: $STUDENT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!rounds[@]}"; do
      round="${rounds[$index]}"
      level="${levels[$index]}"
      seed="${seeds[$index]}"

      round_text=$(printf '%02d' "$round")
      target="$DATA/round_${round_text}_shard_0000.pt"
      log="$EXP/logs/040_collect_round_${round_text}.log"

      if [[ -s "$target" ]]; then
        echo "[SKIP] Existing shard: $target"
        continue
      fi

      echo
      echo "=============================================================="
      echo "CORNERS256 V1 — STAGE 1 DAGGER"
      echo "round=$round_text"
      echo "level=$level"
      echo "seed=$seed"
      echo "teacher_probability=0.90"
      echo "student_probability=0.10"
      echo "samples=100000"
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
        --output_dir "$DATA" \
        --round_index "$round" \
        --curriculum_level "$level" \
        --teacher_probability 0.90 \
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

  stage1_count=$(
    find "$DATA" \
      -maxdepth 1 \
      -type f \
      -name 'round_0[5-9]_shard_0000.pt' \
      | wc -l
  )

  total_count=$(
    find "$DATA" \
      -maxdepth 1 \
      -type f \
      -name 'round_*_shard_*.pt' \
      | wc -l
  )

  echo
  echo "stage1_new_shards=$stage1_count"
  echo "cumulative_shards=$total_count"

  if [[ "$stage1_count" -eq 5 \
     && "$total_count" -eq 10 \
     && "$problems" -eq 0 ]]; then
    echo "[PASS] Stage 1 collection complete"
  else
    echo "[ERROR] Stage 1 collection has $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
