#!/usr/bin/env bash

REPO="/home/sumukh/AerialCatching"

if [[ -d "$REPO" ]]; then
  cd "$REPO"

  EXP="$REPO/experiments/corners256_sequential_v2"

  CHECKPOINT="$REPO/checkpoints/corners256_sequential_v2/stage2/round04_recovery2m_e5_best.pt"

  GLOBAL_RECORD_ROOT="$REPO/eval_logs/initial_episode_records"

  RECORD_ROOT="$REPO/eval_logs/corners256_sequential_v2/round04_recovery2m_screen/initial_episode_records"

  LOG_ROOT="$EXP/logs/round04_recovery2m_screen"

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

      prefix="c256_seqv2_r04_recovery2m_${name}_s${seed}"

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
      echo "================================================================"
      echo "ROUND 04 CONSERVATIVE 2.0M CLOSED-LOOP SCREEN"
      echo "checkpoint=$CHECKPOINT"
      echo "curriculum_level=$level"
      echo "seed=$seed"
      echo "================================================================"

      PYTHONUNBUFFERED=1 python -m rl.eval_rslrl \
        --task Isaac-QuadrotorHoverAsymmetricPPO-v0 \
        --student_checkpoint "$CHECKPOINT" \
        --num_envs 100 \
        --seed "$seed" \
        --curriculum_level "$level" \
        --save_prefix "$prefix" \
        --device cuda:0 \
        2>&1 | tee "$log_file"

      evaluation_status=${PIPESTATUS[0]}

      source_record=$(
        find "$GLOBAL_RECORD_ROOT" \
          -maxdepth 1 \
          -type f \
          -name "${prefix}*_initial_episodes.pt" \
          -print \
          -quit
      )

      if [[ "$evaluation_status" -eq 0 && -n "$source_record" ]]; then
        mv "$source_record" "$destination"
        echo "[MOVED] $destination"
      else
        echo "[ERROR] Evaluation problem for $prefix"
        echo "        status=$evaluation_status"
        echo "        record=$source_record"
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
    echo "[PASS] Round 04 conservative screen completed"
  else
    echo "[ERROR] Screen completed with $problems problem(s)"
  fi
else
  echo "[ERROR] Repository not found: $REPO"
fi
