#!/usr/bin/env bash

cd /home/sumukh/AerialCatching

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES

RUN="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

TRAINER="experiments/state_bottleneck/train_k8_rsl_rl_small_steps.py"

echo "=========================================="
echo "SMALL-STEP PPO — VISUAL QUADROTOR HOVER"
echo "=========================================="

echo "[1] Integration smoke test"

/home/sumukh/env_isaaclab/bin/python -u \
    "$TRAINER" \
    --device cuda:0 \
    --num_envs 64 \
    --rollout_steps 32 \
    --iterations 1 \
    --epochs 1 \
    --minibatches 4 \
    --level 1.0 \
    --seed 7400 \
    --learning_rate 3e-5 \
    --init_std 0.03 \
    --desired_kl 0.01 \
    --anchor_coef 0 \
    --out_dir "$RUN/smoke" \
    > "$RUN/smoke.log" 2>&1

if grep -Fq "[PASS] Actor optimizer steps reduced 50x" "$RUN/smoke.log" \
    && grep -Fq "[TRUST GUARD]" "$RUN/smoke.log" \
    && grep -Fq "[PASS] RSL-RL PPO integration completed" "$RUN/smoke.log"; then

    echo "[PASS] Smoke test"

    echo "[2] Training 10 PPO iterations"

    /home/sumukh/env_isaaclab/bin/python -u \
        "$TRAINER" \
        --device cuda:0 \
        --num_envs 4096 \
        --rollout_steps 32 \
        --iterations 10 \
        --epochs 1 \
        --minibatches 4 \
        --level 1.0 \
        --seed 7400 \
        --learning_rate 3e-5 \
        --init_std 0.03 \
        --desired_kl 0.01 \
        --anchor_coef 0 \
        --out_dir "$RUN/model" \
        > "$RUN/train.log" 2>&1

    if grep -Fq \
        "[PASS] RSL-RL PPO integration completed" \
        "$RUN/train.log"; then

        echo "[PASS] Training completed"

    elif grep -Fq "[EARLY STOP]" "$RUN/train.log"; then

        echo "[STOP] Precision regression detected"

    else

        echo "[ERROR] Unexpected training failure"
        tail -60 "$RUN/train.log"

    fi

    echo
    echo "========== RESULTS =========="

    if [[ -f "$RUN/model/benchmark_history.csv" ]]; then
        cat "$RUN/model/benchmark_history.csv"
    fi

    echo
    echo "========== BEST POLICY =========="

    if [[ -f "$RUN/model/best_metrics.json" ]]; then
        cat "$RUN/model/best_metrics.json"
    fi

else
    echo "[ERROR] Smoke test failed"
    tail -70 "$RUN/smoke.log"
fi
