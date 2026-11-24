#!/usr/bin/env bash
set -uo pipefail

run_group() {
    local GPU="$1"
    shift

    local ROOT="/home/sumukh/AerialCatching"
    local PYTHON_BIN="/home/sumukh/env_isaaclab/bin/python"
    local STUDENT="checkpoints/corners_tcn_student_v1/stage3_dagger2m/stage3_dagger2m_e70.pt"
    local LOG_DIR="experiments/state_bottleneck/eval_logs"

    cd "$ROOT" || return 1

    for MODE in "$@"; do

        local CONTROL_MODE="visual"
        local SPECIALIST_MODE="$MODE"

        if [ "$MODE" = "oracle" ]; then
            CONTROL_MODE="oracle"
            SPECIALIST_MODE="none"
        fi

        if [ "$MODE" = "aux13" ]; then
            SPECIALIST_MODE="none"
        fi

        local LOG="$LOG_DIR/fresh9300_${MODE}.log"

        if [ -e "$LOG" ]; then
            echo "[STOP] Existing evaluation log preserved: $LOG"
            return 1
        fi

        echo "[START] GPU=$GPU mode=$MODE"

        if env \
            PYTHONPATH="$ROOT:$ROOT/rl${PYTHONPATH:+:$PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES="$GPU" \
            "$PYTHON_BIN" -u \
            experiments/state_bottleneck/eval_closed_loop.py \
            --headless \
            --task Isaac-QuadrotorHoverStudentDistill-v0 \
            --num_envs 64 \
            --seed 9300 \
            --curriculum_level 0.0 \
            --student_checkpoint "$STUDENT" \
            --state_bottleneck_mode "$CONTROL_MODE" \
            --critical6_mode "$SPECIALIST_MODE" \
            --save_prefix "critical6_${MODE}_seed9300_" \
            > "$LOG" 2>&1
        then
            if ! grep -q '^\[EVAL\] Initial episodes:' "$LOG"; then
                echo "[FAILED] Missing evaluation summary: $MODE"
                tail -n 30 "$LOG"
                return 1
            fi

            echo "[COMPLETE] $MODE"

            grep -E \
              '^\[EVAL\] Initial episodes:|^Mean completed return:|^Mean completed length:' \
              "$LOG" | tail -n 3
        else
            echo "[FAILED] $MODE"
            tail -n 30 "$LOG"
            return 1
        fi

    done
}

run_group "$@"
