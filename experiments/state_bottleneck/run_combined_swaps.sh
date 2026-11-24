#!/usr/bin/env bash
set -uo pipefail

run_experiments() {
    local GPU="$1"
    shift

    local ROOT="/home/sumukh/AerialCatching"
    local PYTHON_BIN="/home/sumukh/env_isaaclab/bin/python"
    local STUDENT="checkpoints/corners_tcn_student_v1/stage3_dagger2m/stage3_dagger2m_e70.pt"
    local LOG_DIR="experiments/state_bottleneck/eval_logs"

    cd "$ROOT" || return 1

    for SWAP in "$@"; do
        local LOG="$LOG_DIR/combo_${SWAP}.log"

        if [ -e "$LOG" ]; then
            echo "[STOP] Existing log preserved: $LOG"
            return 1
        fi

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
            --save_prefix "state_bottleneck_combo_${SWAP}_seed4300_" \
            > "$LOG" 2>&1
        then
            if ! grep -q '^\[EVAL\] Initial episodes:' "$LOG"; then
                echo "[FAILED] No episode summary: $SWAP"
                tail -n 30 "$LOG"
                return 1
            fi

            echo "[COMPLETE] $SWAP"

            grep -E \
              '^\[EVAL\] Initial episodes:|^Mean completed return:|^Mean completed length:' \
              "$LOG" | tail -n 3
        else
            echo "[FAILED] $SWAP"
            tail -n 30 "$LOG"
            return 1
        fi
    done
}

run_experiments "$@"
