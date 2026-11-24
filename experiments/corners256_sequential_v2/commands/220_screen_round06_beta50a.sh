#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"

  CHECKPOINT="$REPO/checkpoints/corners256_sequential_v2/stage2/round06_beta50a_2p25m_e2_best.pt"

  GLOBAL_RECORD_ROOT="$REPO/eval_logs/initial_episode_records"

  RECORD_ROOT="$REPO/eval_logs/corners256_sequential_v2/round06_beta50a_screen/initial_episode_records"

  LOG_ROOT="$EXP/logs/round06_beta50a_screen"

  mkdir -p \
    "$GLOBAL_RECORD_ROOT" \
    "$RECORD_ROOT" \
    "$LOG_ROOT"

  levels=("0.00" "0.25" "0.50" "0.75" "1.00")
  names=("l000" "l025" "l050" "l075" "l100")

  seed="3000"
  problems=0

  if [[ ! -s "$CHECKPOINT" ]]; then
    echo "[ERROR] Missing checkpoint: $CHECKPOINT"
    problems=$((problems + 1))
  fi

  if [[ "$problems" -eq 0 ]]; then
    for index in "${!levels[@]}"; do
      level="${levels[$index]}"
      name="${names[$index]}"

      prefix="c256_seqv2_r06_beta50a_${name}_s${seed}"

      destination="$RECORD_ROOT/${prefix}_initial_episodes.pt"
      log_file="$LOG_ROOT/${prefix}.log"

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
      echo "=============================================================="
      echo "ROUND 06 BETA-50 A SCREEN"
      echo "level=$level"
      echo "seed=$seed"
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

      source_record=$(
        find "$GLOBAL_RECORD_ROOT" \
          -maxdepth 1 \
          -type f \
          -name "${prefix}*_initial_episodes.pt" \
          -print \
          -quit
      )

      if [[ "$status" -eq 0 && -n "$source_record" ]]; then
        mv "$source_record" "$destination"
        echo "[MOVED] $destination"
      else
        echo "[ERROR] level=$level status=$status record=$source_record"
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
    echo "[PASS] Round 06 closed-loop screen completed"
  else
    echo "[ERROR] Screen completed with $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
