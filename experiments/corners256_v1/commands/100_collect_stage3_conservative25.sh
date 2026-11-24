#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_v1"
  DATA="$REPO/dagger_data/corners256_v1"

  TEACHER="$REPO/logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"
  STUDENT="$REPO/checkpoints/corners256_v1/stage2_dagger1p5m_e5.pt"

  rounds=(15 16 17 18 19)
  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  seeds=(4800 4801 4802 4803 4804)

  problems=0

  if [[ ! -s "$TEACHER" ]]; then
    echo "[ERROR] Missing teacher: $TEACHER"
    problems=$((problems + 1))
  fi

  if [[ ! -s "$STUDENT" ]]; then
    echo "[ERROR] Missing Stage 2 checkpoint: $STUDENT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!rounds[@]}"; do
      round="${rounds[$index]}"
      level="${levels[$index]}"
      seed="${seeds[$index]}"

      round_text=$(printf '%02d' "$round")
      target="$DATA/round_${round_text}_shard_0000.pt"
      log="$EXP/logs/100_collect_round_${round_text}.log"

      if [[ -s "$target" ]]; then
        echo "[SKIP] Existing shard: $target"
        continue
      fi

      echo
      echo "=============================================================="
      echo "CORNERS256 V1 — STAGE 3 CONSERVATIVE DAGGER"
      echo "round=$round_text"
      echo "level=$level"
      echo "seed=$seed"
      echo "teacher_probability=0.75"
      echo "student_probability=0.25"
      echo "samples=100000"
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
        --teacher_probability 0.75 \
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

  new_count=$(
    find "$DATA" -maxdepth 1 -type f \
      -name 'round_1[5-9]_shard_0000.pt' | wc -l
  )

  total_count=$(
    find "$DATA" -maxdepth 1 -type f \
      -name 'round_*_shard_*.pt' | wc -l
  )

  echo
  echo "stage3_new_shards=$new_count"
  echo "cumulative_shards=$total_count"

  if [[ "$new_count" -eq 5 \
     && "$total_count" -eq 20 \
     && "$problems" -eq 0 ]]; then
    echo "[PASS] Stage 3 collection complete"
  else
    echo "[ERROR] Stage 3 collection problems=$problems"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
