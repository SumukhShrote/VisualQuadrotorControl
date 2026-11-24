#!/usr/bin/env bash

cd /home/sumukh/AerialCatching

RUN="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES
export PYTHONFAULTHANDLER=1

TRAINER="experiments/state_bottleneck/train_k8_rsl_rl_anchor_ablation.py"

echo "======================================================"
echo "K8 TRANSFORMER — PPO ANCHOR STRENGTH ABLATION"
echo "======================================================"
echo "Started: $(date)"
echo "Output: $RUN"

echo
echo "[STAGE 1] Gradient diagnostic smoke test"

/home/sumukh/env_isaaclab/bin/python -u \
    "$TRAINER" \
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
    --anchor_coef 5 \
    --out_dir "$RUN/smoke" \
    > "$RUN/smoke.log" 2>&1

if grep -Fq \
    "[PASS] RSL-RL PPO integration completed" \
    "$RUN/smoke.log" \
    && grep -Fq \
    "[GRAD DIAG]" \
    "$RUN/smoke.log"; then

    echo "[PASS] Gradient diagnostics smoke test"

    echo
    echo "[STAGE 2] Controlled anchor ablation"

    for COEF in 5 1; do

        echo
        echo "=============================================="
        echo "ANCHOR COEFFICIENT: $COEF"
        echo "=============================================="

        OUT="$RUN/lambda_$COEF"
        LOG="$RUN/lambda_$COEF.log"

        /home/sumukh/env_isaaclab/bin/python -u \
            "$TRAINER" \
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
            --anchor_coef "$COEF" \
            --out_dir "$OUT" \
            > "$LOG" 2>&1

        if grep -Fq \
            "[PASS] RSL-RL PPO integration completed" \
            "$LOG"; then

            echo "[PASS] lambda=$COEF completed"
            touch "$RUN/LAMBDA_${COEF}_COMPLETE"

        elif grep -Fq \
            "[EARLY STOP]" \
            "$LOG"; then

            echo "[GATED] lambda=$COEF stopped for regression"
            touch "$RUN/LAMBDA_${COEF}_GATED"

        else

            echo "[ERROR] Unexpected failure for lambda=$COEF"
            touch "$RUN/LAMBDA_${COEF}_ERROR"

            echo "Last 50 log lines:"
            tail -50 "$LOG"

            echo "Stopping ablation after unexpected failure."
            break
        fi

        echo
        echo "========== EVALUATION: lambda=$COEF =========="

        if [[ -f "$OUT/benchmark_history.csv" ]]; then
            cat "$OUT/benchmark_history.csv"
        fi

        echo
        echo "========== GRADIENT DIAGNOSTICS =========="

        if [[ -f "$OUT/gradient_diagnostics.csv" ]]; then
            cat "$OUT/gradient_diagnostics.csv"
        fi

        echo
        echo "========== BEST CHECKPOINT =========="

        if [[ -f "$OUT/best_metrics.json" ]]; then
            cat "$OUT/best_metrics.json"
        fi

    done

else

    echo "[ERROR] Gradient diagnostic smoke test failed."
    echo "Inspect $RUN/smoke.log"

    touch "$RUN/SMOKE_FAILED"

fi

echo
echo "[FINISHED] $(date)"
