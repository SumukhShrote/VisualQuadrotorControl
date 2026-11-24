#!/usr/bin/env bash

cd /home/sumukh/AerialCatching

LOGROOT="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES

set -o pipefail

echo "=============================================="
echo "K8 TRANSFORMER — NATIVE RSL-RL PPO"
echo "=============================================="
echo "Started: $(date)"
echo "Logs: $LOGROOT"
echo

# First verify full-model gradients and optimizer
# through two actual PPO iterations.

echo "[STAGE] Full-model smoke test"

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
    --out_dir "$LOGROOT/smoke" \
    2>&1 | tee "$LOGROOT/smoke.log"

if grep -Fq \
    "[PASS] RSL-RL PPO integration completed" \
    "$LOGROOT/smoke.log"; then

    echo "[PASS] Full-model smoke test"

    echo
    echo "[STAGE] Overnight training"
    echo "Maximum: 750 iterations / 98.3M transitions"
    echo "Checkpoint interval: 25 iterations"
    echo "Benchmark: L0 + L1 on GPU 1"
    echo

    /home/sumukh/env_isaaclab/bin/python -u \
        experiments/state_bottleneck/train_k8_rsl_rl_ppo.py \
        --device cuda:0 \
        --num_envs 4096 \
        --rollout_steps 32 \
        --iterations 750 \
        --epochs 5 \
        --minibatches 4 \
        --level 1.0 \
        --seed 7400 \
        --learning_rate 3e-5 \
        --init_std 0.03 \
        --desired_kl 0.01 \
        --out_dir "$LOGROOT/model" \
        2>&1 | tee "$LOGROOT/train.log"

    if grep -Fq \
        "[PASS] RSL-RL PPO integration completed" \
        "$LOGROOT/train.log"; then
        echo "[SUCCESS] Overnight PPO finished"
        echo "Results: $LOGROOT/model"
        touch "$LOGROOT/TRAINING_COMPLETE"
    else
        echo "[STOP] Training did not complete normally."
        echo "Inspect: $LOGROOT/train.log"
        touch "$LOGROOT/TRAINING_STOPPED"
    fi

else
    echo "[STOP] Full-model smoke test failed."
    echo "Inspect: $LOGROOT/smoke.log"
    touch "$LOGROOT/SMOKE_FAILED"
fi

echo
echo "Finished: $(date)"
