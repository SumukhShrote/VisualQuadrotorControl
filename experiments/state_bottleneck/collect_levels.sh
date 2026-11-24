#!/usr/bin/env bash
set -uo pipefail

collect_levels() {
    local GPU="$1"
    shift

    local ROOT="/home/sumukh/AerialCatching"
    local PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

    local TEACHER="logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt"
    local STUDENT="checkpoints/corners_tcn_student_v1/stage3_dagger2m/stage3_dagger2m_e70.pt"

    local OUT="experiments/state_bottleneck/data"
    local LOG_DIR="experiments/state_bottleneck/logs"

    cd "$ROOT" || return 1

    mkdir -p "$OUT" "$LOG_DIR"

    for SPEC in "$@"; do
        local ROUND LEVEL LOG STATUS
        IFS=: read -r ROUND LEVEL <<< "$SPEC"

        LOG="$LOG_DIR/collect_round_${ROUND}.log"

        echo "[START] GPU=$GPU round=$ROUND level=$LEVEL"

        if compgen -G "$OUT/round_$(printf '%02d' "$ROUND")_shard_*.pt" > /dev/null; then
            echo "[ERROR] Existing shard found for round $ROUND."
            echo "        Preserving existing data."
            return 1
        fi

        if env \
            PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            CUDA_VISIBLE_DEVICES="$GPU" \
            "$PYTHON_BIN" -u \
            rl/collect_tcn_probe_data.py \
            --headless \
            --task Isaac-QuadrotorHoverStudentDistill-v0 \
            --num_envs 64 \
            --seed "$((4300 + ROUND))" \
            --teacher_checkpoint "$TEACHER" \
            --student_checkpoint "$STUDENT" \
            --output_dir "$OUT" \
            --round_index "$ROUND" \
            --curriculum_level "$LEVEL" \
            --teacher_probability 0.0 \
            --mixing_mode episode \
            --num_samples 20000 \
            --shard_size 20000 \
            > "$LOG" 2>&1
        then
            echo "[COMPLETE] round=$ROUND level=$LEVEL"
        else
            STATUS=$?
            echo "[FAILED] round=$ROUND status=$STATUS"
            tail -n 35 "$LOG"
            return "$STATUS"
        fi
    done

    echo "[FINISHED] GPU=$GPU all assigned levels complete"
}

collect_levels "$@"
