# Training reproduction

This procedure trains the visual student from the retained privileged PPO teacher and supervised dataset. Obtain the external inputs specified in training_inputs.json before training.

## Setup

    export PYTHONPATH="$PWD"
    export PYTHONDONTWRITEBYTECODE=1
    DATA="$PWD/datasets/state_estimation/pooled_student_epoch70_all_levels.pt"
    RUN="$PWD/results/runs/fresh_visual_student_$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$RUN"

## 1. Temporal representation pretraining

    python -m rl.pretrain_state_estimator --dataset "$DATA" --out "$RUN/01_temporal" --histories 32 --epochs 30 --batch 1024 --device cuda:0 --seed 20261119 --loss huber --lr 0.001

## 2. Behavioral cloning

    python -m experiments.imitation_learning.train_visual_imitation --variant pretrained --pretrained "$RUN/01_temporal/multiscale_transformer_H32.pt" --protocol "$RUN/01_temporal/protocol.json" --out "$RUN/02_visual_imitation" --epochs 30 --batch 1024

## 3. Action-history learning

    python -m experiments.imitation_learning.train_action_history train --k 8 --gpu 0 --initial "$RUN/02_visual_imitation/policy.pt" --protocol "$RUN/01_temporal/protocol.json" --out "$RUN/03_action_history" --epochs 25 --batch 1024

## 4. On-policy DAgger

The validated collection protocol uses two CUDA GPUs.

    python -m rl.train_action_history_dagger --round 1 --base-checkpoint "$RUN/03_action_history/policy.pt" --base-policy "$RUN/03_action_history/policy.ts" --protocol "$RUN/01_temporal/protocol.json" --out "$RUN/05_dagger_round1" --epochs 12 --batch 512

    python -m rl.train_action_history_dagger --round 2 --base-checkpoint "$RUN/05_dagger_round1/finetuned_k8/policy.pt" --base-policy "$RUN/05_dagger_round1/finetuned_k8/policy.ts" --prior-run "$RUN/05_dagger_round1" --protocol "$RUN/01_temporal/protocol.json" --out "$RUN/05_dagger_round2" --epochs 12 --batch 512

## 5. Frozen representation DAgger

    BASE="$RUN/05_dagger_round2/finetuned_k8"
    FINAL="$RUN/06_pose_attention_dagger"

    python -m rl.train_pose_attention_dagger --base-checkpoint "$BASE/policy.pt" --base-policy "$BASE/policy.ts" --out "$FINAL" --device cuda:0 --num-envs 512 --pose-updates 12000 --rounds 24 --rollout-steps 192 --updates-per-round 180 --batch 2048 --head-lr 0.0001 --seed 14001

## 6. End-to-end representation DAgger

    python -m rl.train_pose_attention_dagger --base-checkpoint "$BASE/policy.pt" --base-policy "$BASE/policy.ts" --pose-checkpoint "$FINAL/pose_best.pt" --out "$RUN/08_end_to_end_dagger" --device cuda:0 --num-envs 512 --rounds 24 --rollout-steps 192 --updates-per-round 180 --batch 2048 --head-lr 0.0001 --representation-lr 0.00001 --train-representation --seed 14001

The joint experiment starts from the same pretrained representations as the frozen experiment, not its trained controller head.

The original selection criterion retained round 18. Round 24 was selected using validation position accuracy as a tie-break and independently evaluated.

## 7. Independent evaluation

    for POLICY in frozen end_to_end; do
        python -m evaluation.evaluate_controller --out "$RUN/eval_${POLICY}_10s" --device cuda:0 --num-envs 256 --duration 10 --level 1.0 --seeds 38001 38002 --domains original expanded --policies candidate --candidate-checkpoint "checkpoints/hover/$POLICY/checkpoint.pt" --candidate-base checkpoints/reference/visual_baseline/dagger_round2.pt --candidate-policy "checkpoints/hover/$POLICY/policy.ts"

        python -m evaluation.evaluate_controller --out "$RUN/eval_${POLICY}_60s" --device cuda:0 --num-envs 128 --duration 60 --level 1.0 --seeds 39001 --domains original expanded --policies candidate --candidate-checkpoint "checkpoints/hover/$POLICY/checkpoint.pt" --candidate-base checkpoints/reference/visual_baseline/dagger_round2.pt --candidate-policy "checkpoints/hover/$POLICY/policy.ts"
    done

The evaluation commands above target the distributed controllers. Use the corresponding newly trained checkpoints and exports when evaluating a reproduction.

Exact reproducibility across hardware and software versions is not guaranteed.
