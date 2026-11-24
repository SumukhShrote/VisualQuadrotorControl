#!/usr/bin/env bash

cd /home/sumukh/AerialCatching

RUN="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
export PYTHONFAULTHANDLER=1

set -o pipefail

echo "=================================================="
echo "K8 TRANSFORMER — ANCHORED PRECISION PPO"
echo "=================================================="
echo "Start: $(date)"
echo "Directory: $RUN"
echo

echo "[STAGE 1] Native PPO integration smoke test"

/home/sumukh/env_isaaclab/bin/python -u \
    experiments/state_bottleneck/train_k8_rsl_rl_ppo.py \
    --device cuda:0 \
    --num_envs 32 \
    --rollout_steps 32 \
    --iterations 2 \
    --epochs 3 \
    --minibatches 4 \
    --level 1.0 \
    --seed 7400 \
    --learning_rate 3e-5 \
    --init_std 0.03 \
    --desired_kl 0.01 \
    --anchor_coef 20.0 \
    --out_dir "$RUN/smoke" \
    > "$RUN/smoke.log" 2>&1

if grep -Fq \
    "[PASS] RSL-RL PPO integration completed" \
    "$RUN/smoke.log" \
    && grep -Fq "[ANCHOR] loss=" "$RUN/smoke.log"; then

    echo "[PASS] Native PPO + reference regularization"
    echo
    echo "[STAGE 2] Precision PPO pilot"
    echo "Maximum transitions: 3,276,800"
    echo "Evaluation iterations: 1, 3, 5, 10, 25"
    echo "Automatic regression gate: enabled"
    echo

    /home/sumukh/env_isaaclab/bin/python -u \
        experiments/state_bottleneck/train_k8_rsl_rl_ppo.py \
        --device cuda:0 \
        --num_envs 4096 \
        --rollout_steps 32 \
        --iterations 25 \
        --epochs 5 \
        --minibatches 4 \
        --level 1.0 \
        --seed 7400 \
        --learning_rate 3e-5 \
        --init_std 0.03 \
        --desired_kl 0.01 \
        --anchor_coef 20.0 \
        --out_dir "$RUN/pilot" \
        > "$RUN/pilot.log" 2>&1

    if grep -Fq \
        "[PASS] RSL-RL PPO integration completed" \
        "$RUN/pilot.log"; then

        echo "[PASS] Precision PPO pilot completed"
        touch "$RUN/PILOT_COMPLETE"

    elif grep -Fq \
        "[EARLY STOP]" \
        "$RUN/pilot.log"; then

        echo "[STOPPED] Precision safety gate triggered"
        touch "$RUN/STOPPED_BY_GATE"

    else
        echo "[ERROR] Pilot ended unexpectedly"
        touch "$RUN/PILOT_ERROR"
    fi

    echo
    echo "========== EVALUATION HISTORY =========="

    if [[ -f "$RUN/pilot/benchmark_history.csv" ]]; then
        cat "$RUN/pilot/benchmark_history.csv"
    fi

    echo
    echo "========== BEST POLICY =========="

    if [[ -f "$RUN/pilot/best_metrics.json" ]]; then
        cat "$RUN/pilot/best_metrics.json"
    fi

else
    echo "[STOPPED] Smoke test did not pass"
    echo "Inspect: $RUN/smoke.log"
    touch "$RUN/SMOKE_FAILED"
fi

echo
echo "Finished: $(date)"
