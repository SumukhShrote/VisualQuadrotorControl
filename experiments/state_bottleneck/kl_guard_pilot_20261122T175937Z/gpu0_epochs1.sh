#!/usr/bin/env bash

cd /home/sumukh/AerialCatching

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset CUDA_VISIBLE_DEVICES

/home/sumukh/env_isaaclab/bin/python -u     "/home/sumukh/AerialCatching/experiments/state_bottleneck/train_k8_rsl_rl_kl_guard.py"     --device cuda:0     --num_envs 4096     --rollout_steps 32     --iterations 5     --epochs 1     --minibatches 4     --level 1.0     --seed 7400     --learning_rate 3e-5     --init_std 0.03     --desired_kl 0.01     --anchor_coef 0     --out_dir "/home/sumukh/AerialCatching/experiments/state_bottleneck/kl_guard_pilot_20261122T175937Z/gpu0_epochs1"
