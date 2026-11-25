# Retained Research Methods

The working tree retains selected reusable research methods.
Superseded experimental implementations were removed.

Historical source Git commit: `2ad06bfbe9c0cb521dff746882e1d6c2e3e6c006`.

The historical source can be inspected using:

    git show <commit>:<original-path>

This does not guarantee complete historical retraining, because
unnecessary intermediate checkpoints have been pruned.

## Retained representation and policy methods

- `experiments/state_bottleneck/analyze_hover_trajectories.py`
- `experiments/state_bottleneck/collect_transformer_teacher_probe.py`
- `experiments/state_bottleneck/eval_closed_loop.py`
- `experiments/state_bottleneck/eval_three_visual_actors_timerfixed.py`
- `experiments/state_bottleneck/geometric_state_probe.py`
- `experiments/state_bottleneck/launch_action_history_study.py`
- `experiments/state_bottleneck/pnp_pose_geometry_corrected.py`
- `experiments/state_bottleneck/pnp_velocity_comparison.py`
- `experiments/state_bottleneck/pool_shards.py`
- `experiments/state_bottleneck/run_k8_dagger_round.py`
- `experiments/state_bottleneck/run_k8_dagger_round2.py`
- `experiments/state_bottleneck/supervised_multiscale_masked_huber.py`
- `experiments/state_bottleneck/supervised_multiscale_masked_mse.py`
- `experiments/state_bottleneck/supervised_multiscale_plain_huber.py`
- `experiments/state_bottleneck/supervised_multiscale_transformer.py`
- `experiments/state_bottleneck/supervised_observability_probe.py`
- `experiments/state_bottleneck/train_critical6.py`
- `experiments/state_bottleneck/train_k8_rsl_rl_direction_probe_20261123T174604Z.py`
- `experiments/state_bottleneck/train_k8_rsl_rl_long_horizon_gae.py`
- `experiments/state_bottleneck/train_k8_rsl_rl_ppo.py`
- `experiments/state_bottleneck/train_two_gpu_visual_policies.py`

## Research evidence

- Historical results: `results/historical_experiments/`.
- Path relocation mapping: `docs/reproducibility/EXPERIMENT_PATH_MAPPING.json`.
- Consolidated terminal evidence: `results/historical_experiments/terminal_experiment_evidence.jsonl`.
- Canonical checkpoints: `checkpoints/reference/`.
- Canonical controller evaluator: `evaluation/evaluate_controller.py`.

Historical methods may require updated artifact paths.
The canonical controller does not depend on those historical
experiment output locations at inference time.

