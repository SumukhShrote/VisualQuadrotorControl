#!/usr/bin/env bash

cd /home/sumukh/AerialCatching

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
export PYTHONFAULTHANDLER=1

RUN="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

TRAINER="experiments/state_bottleneck/train_k8_rsl_rl_kl_guard_fixed.py"

echo "=============================================="
echo "K8 VISUAL HOVER — KL-CONSTRAINED PPO"
echo "=============================================="
echo "Started: $(date)"
echo

for SPEC in "0 1" "1 5"; do

    read GPU EPOCHS <<< "$SPEC"

    LABEL="gpu${GPU}_epochs${EPOCHS}"
    OUT="$RUN/$LABEL"
    LOG="$RUN/$LABEL.log"

    echo
    echo "=============================================="
    echo "Training GPU: $GPU"
    echo "PPO epochs: $EPOCHS"
    echo "=============================================="

    /home/sumukh/env_isaaclab/bin/python -u \
        "$TRAINER" \
        --device "cuda:$GPU" \
        --num_envs 4096 \
        --rollout_steps 32 \
        --iterations 5 \
        --epochs "$EPOCHS" \
        --minibatches 4 \
        --level 1.0 \
        --seed 7400 \
        --learning_rate 3e-5 \
        --init_std 0.03 \
        --desired_kl 0.01 \
        --anchor_coef 0 \
        --out_dir "$OUT" \
        > "$LOG" 2>&1

    if grep -Fq \
        "[PASS] RSL-RL PPO integration completed" \
        "$LOG"; then

        echo "[PASS] $LABEL completed"

    elif grep -Fq "[EARLY STOP]" "$LOG"; then

        echo "[GATED] $LABEL stopped for precision regression"

    else

        echo "[ERROR] $LABEL failed unexpectedly"
        tail -70 "$LOG"
        break

    fi

    echo
    echo "========== TRUST-REGION HISTORY =========="

    if [[ -f "$OUT/trust_guard.csv" ]]; then
        cat "$OUT/trust_guard.csv"
    fi

    echo
    echo "========== HOVERING PERFORMANCE =========="

    if [[ -f "$OUT/benchmark_history.csv" ]]; then
        cat "$OUT/benchmark_history.csv"
    fi

    echo
    echo "========== BEST CONTROLLER =========="

    if [[ -f "$OUT/best_metrics.json" ]]; then
        cat "$OUT/best_metrics.json"
    fi

done

echo
echo "[FINISHED] $(date)"
