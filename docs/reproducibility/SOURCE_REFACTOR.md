# Source consolidation

This pass reorganizes the remaining methods rather than deleting numerical
research evidence. All canonical checkpoints and existing results are protected.

- Four supervised trainers -> one trainer, with MSE/Huber and optional corner
  validity features. Default plain Huber reproduces the selected architecture.
- Two K8 DAgger orchestrators -> one, with explicit round-specific aggregation.
- Two byte-identical generated K8 collectors -> one maintained source.
- Two visual evaluators -> one with explicit policy input mode.
- Three historical PPO variants are retired from the working tree because their
  required `rl.corners_asymmetric_ppo.config` was removed in the Dreamer cleanup.
  Their exact implementations are preserved in ordinary Git history.
- The old 1/8/32 CLI default is corrected to 32: the estimator rejects shorter
  histories. The masked experiments append visibility features, not loss masks.
- Old absolute roots and relocated data paths are corrected in curated methods.
- The asset conversion YAML is not modified: its original asset points to a
  different repository, and the converter is not part of the provided source.

The final reference controller, simulation dynamics, rewards, checkpoint
contents and canonical evaluator are not changed by this pass. The temporal
model gains an optional feature flag while retaining its default architecture
and checkpoint parameter names. Historical methods require their original
artifacts and still need runtime validation.

## Source relocations

| Original | Maintained path |
|---|---|
| `experiments/corners_tcn_student_v1/train_aux13_student.py` | `experiments/imitation_learning/train_aux13_student.py` |
| `experiments/state_bottleneck/action_history_study_20261120T200813Z/evaluate_action_history.py` | `evaluation/evaluate_visual_policy.py` |
| `experiments/state_bottleneck/analyze_hover_trajectories.py` | `evaluation/analyze_hover_trajectories.py` |
| `experiments/state_bottleneck/collect_transformer_teacher_probe.py` | `rl/collect_representation_data.py` |
| `experiments/state_bottleneck/eval_closed_loop.py` | `evaluation/evaluate_state_bottleneck.py` |
| `experiments/state_bottleneck/eval_three_visual_actors_timerfixed.py` | `evaluation/evaluate_visual_policy.py` |
| `experiments/state_bottleneck/geometric_state_probe.py` | `experiments/geometric_benchmarks/geometric_state_probe.py` |
| `experiments/state_bottleneck/k8_onpolicy_dagger_round1_20261120T214738Z/collect_k8_dagger.py` | `rl/collect_action_history.py` |
| `experiments/state_bottleneck/k8_onpolicy_dagger_round2_20261120T225553Z/collect_k8_dagger.py` | `rl/collect_action_history.py` |
| `experiments/state_bottleneck/launch_action_history_study.py` | `experiments/imitation_learning/train_action_history.py` |
| `experiments/state_bottleneck/pnp_pose_geometry_corrected.py` | `experiments/geometric_benchmarks/pnp_pose_geometry_corrected.py` |
| `experiments/state_bottleneck/pnp_velocity_comparison.py` | `experiments/geometric_benchmarks/pnp_velocity_comparison.py` |
| `experiments/state_bottleneck/pool_shards.py` | `experiments/representation_learning/pool_shards.py` |
| `experiments/state_bottleneck/run_k8_dagger_round.py` | `rl/train_action_history_dagger.py` |
| `experiments/state_bottleneck/run_k8_dagger_round2.py` | `rl/train_action_history_dagger.py` |
| `experiments/state_bottleneck/supervised_multiscale_masked_huber.py` | `rl/pretrain_state_estimator.py` |
| `experiments/state_bottleneck/supervised_multiscale_masked_mse.py` | `rl/pretrain_state_estimator.py` |
| `experiments/state_bottleneck/supervised_multiscale_plain_huber.py` | `rl/pretrain_state_estimator.py` |
| `experiments/state_bottleneck/supervised_multiscale_transformer.py` | `rl/pretrain_state_estimator.py` |
| `experiments/state_bottleneck/supervised_observability_probe.py` | `experiments/representation_learning/observability_probe.py` |
| `experiments/state_bottleneck/train_critical6.py` | `experiments/imitation_learning/train_critical6.py` |
| `experiments/state_bottleneck/train_two_gpu_visual_policies.py` | `experiments/imitation_learning/train_visual_imitation.py` |

## Retired PPO implementations

- `experiments/state_bottleneck/train_k8_rsl_rl_direction_probe_20261123T174604Z.py`
- `experiments/state_bottleneck/train_k8_rsl_rl_long_horizon_gae.py`
- `experiments/state_bottleneck/train_k8_rsl_rl_ppo.py`

## Artifact relocations

- `experiments/state_bottleneck/data` -> `datasets/state_estimation`
- `experiments/state_bottleneck/action_history_study_20261120T200813Z` -> `results/historical_experiments/state_bottleneck/action_history_study_20261120T200813Z`
- `experiments/state_bottleneck/k8_onpolicy_dagger_round1_20261120T214738Z` -> `results/historical_experiments/state_bottleneck/k8_onpolicy_dagger_round1_20261120T214738Z`
- `experiments/state_bottleneck/k8_onpolicy_dagger_round2_20261120T225553Z` -> `results/historical_experiments/state_bottleneck/k8_onpolicy_dagger_round2_20261120T225553Z`
- `experiments/corners_tcn_student_v1` -> `results/historical_experiments/corners_tcn_student_v1`

## Verification performed on the uploaded source snapshot

The four historical Transformer constructors were compared against the unified
model on seeded CPU inputs, including a missing corner. All state-dictionary
keys and seeded parameter values matched exactly; maximum output error was 0
for all four variants. Each loss/visibility combination also completed one
training epoch with a 240-row synthetic episode-disjoint dataset and wrote
metrics and checkpoints. This is a software test, not a research result.

The uploaded source snapshot excludes environment implementations and binary
checkpoints/datasets. Full historical retraining, actual artifact availability,
Isaac Lab integration, and physical results must be verified on the workstation.
The existing validated reference-controller checkpoint hashes are checked by
the installer before and after changes.

The K=0, 1 and 8 pretrained action-history initializers were also compared
against the original implementation with a synthetic pretrained checkpoint.
All parameters and forward outputs matched exactly (maximum error 0).

Pre-consolidation source Git commit: `39ee23f5fe9625e8bc51dc8ca14a96500f69bcc9`.
