#!/usr/bin/env bash
set -uo pipefail

GPU="$1"
shift

ROOT="/home/sumukh/AerialCatching"
PYTHON_BIN="/home/sumukh/env_isaaclab/bin/python"

STUDENT="checkpoints/corners_tcn_student_v1/stage3_dagger2m/stage3_dagger2m_e70.pt"

LOG_DIR="experiments/state_bottleneck/eval_logs"

cd "$ROOT" || return 1 2>/dev/null || false

for SWAP in "$@"; do
    LOG="$LOG_DIR/swap_${SWAP}.log"

    echo "[START] GPU=$GPU swap=$SWAP"

    if env \
        PYTHONPATH="$ROOT:$ROOT/rl${PYTHONPATH:+:$PYTHONPATH}" \
        CUDA_VISIBLE_DEVICES="$GPU" \
        "$PYTHON_BIN" -u \
        experiments/state_bottleneck/eval_closed_loop.py \
        --headless \
        --task Isaac-QuadrotorHoverStudentDistill-v0 \
        --num_envs 64 \
        --seed 4300 \
        --curriculum_level 0.0 \
        --student_checkpoint "$STUDENT" \
        --state_bottleneck_mode visual \
        --state_bottleneck_swap "$SWAP" \
        --save_prefix "state_bottleneck_swap_${SWAP}_seed4300_" \
        > "$LOG" 2>&1
    then
        echo "[COMPLETE] swap=$SWAP"

        grep -E \
          '^\[EVAL\] Initial episodes:|^Mean completed return:|^Mean completed length:' \
          "$LOG" | tail -n 4

        if ! grep -q '^\[EVAL\] Initial episodes:' "$LOG"; then
            echo "[ERROR] Evaluation exited without episode summary."
            break
        fi
    else
        echo "[FAILED] swap=$SWAP"
        tail -n 25 "$LOG"
        break
    fi
done

echo "[FINISHED] GPU=$GPU"
