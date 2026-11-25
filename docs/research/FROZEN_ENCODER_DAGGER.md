# Frozen-Encoder DAgger with Explicit Spatial Pose Supervision

## Date

November 24, 2026.

## Research question

Can a pretrained visual representation that explicitly preserves
marker-relative position and heading support precise closed-loop control
when only the downstream controller is optimized?

## Initialization

The original three temporal attention branches were recovered from
the established K=8 PPO controller.

The independently supervised spatial pose branch was added to this
controller. The new input connections in the expanded controller
head were zero-initialized to preserve the existing policy behavior.

All original model weights and the spatial pose branch were frozen.

## Representation

Each observation is converted to a 293-dimensional feature vector:

- Geometry attention features: 64.
- Linear velocity attention features: 64.
- Angular velocity attention features: 64.
- Previous executed control actions: 32.
- Spatial pose attention features: 64.
- Normalized explicit pose predictions: 5.

The policy head maps these features to four normalized CTBR commands.

Architecture: 293 -> 256 -> 128 -> 4.

## Data collection

512 environments are simulated in parallel.

Each DAgger round collects 192 transitions per environment,
producing 98,304 transitions per round.

The replay buffer stores frozen feature vectors, privileged teacher
actions and sample weights.

Replay capacity: 180,000 transitions.

The teacher provides action labels for every collected state,
regardless of whether the teacher or student controls the drone.

The controller used for execution is sampled per episode.

Teacher execution probability:

- Rounds 1 and 2: 1.0.
- Subsequent rounds:
  max(0.05, 0.4 * 0.8 ** (round - 3)).

Action history always contains the actual executed commands.

Action history is cleared when an environment terminates or resets.

## Reset curriculum

The expanded reset domain contains independent position, orientation,
linear velocity and angular velocity variations, including targeted
lateral-yaw compensation configurations.

Every fourth round collects experience under the original curriculum
at maximum difficulty.

Other rounds use the expanded curriculum, progressively increasing
its severity.

The privileged teacher must pass a recoverability gate before
student training begins.

The accepted expanded reset scale was 1.0.

## Supervision objective

Weighted teacher-action mean squared error.

Samples with position error below 0.25 m receive weight 3.
Other samples receive weight 1.

This emphasizes corrective control near the hover target while
retaining disturbance-recovery experience.

## Optimization

- AdamW.
- Learning rate: 0.0001.
- Weight decay: 0.000001.
- Batch size: 2048.
- Updates per round: 180.
- Gradient norm clipping: 1.0.
- Trainable parameters: 108,676.
- Frozen parameters: 447,289.

Only the expanded policy head is optimized.

## Checkpoint selection

Validation is performed every third DAgger round and at the
configured final round.

The selection score is the sum of precision-hold rates across
the original and expanded reset domains.

A checkpoint must also satisfy survival safeguards and must not
substantially degrade original-domain precision performance.

The selected reference checkpoint is round 15.

Transitions collected through round 15: 1,474,560.

The process subsequently reached round 18. The selected reference
checkpoint remains round 15; round 18 is not independently validated.

## Results

| Round | Original-domain hold | Expanded-domain hold |
|---:|---:|---:|
| Initial | 45.31% | 17.97% |
| 3 | 99.02% | 78.71% |
| 6 | 99.61% | 95.51% |
| 9 | 99.80% | 98.44% |
| 12 | 100.00% | 98.83% |
| 15 | 100.00% | 99.22% |

Independent evaluation subsequently established:

- 1024/1024 successful 10-second episodes.
- 256/256 successful 60-second episodes.
- Continuous precision hold between 10 and 60 seconds in all
  extended student episodes.
- Approximately 0.82 cm mean settled lateral position error
  under the expanded reset domain.

## Interpretation

The combined representation and curriculum intervention enabled
near-teacher-level control without updating the perception network
during final DAgger training.

The separate contributions of the spatial attention features,
explicit pose predictions, expanded curriculum, weighted imitation
objective and additional training have not been isolated.

## Canonical artifacts

- checkpoints/reference/pose_attention/checkpoint.pt
- checkpoints/reference/pose_attention/policy.ts
- checkpoints/reference/visual_baseline/ppo_checkpoint.pt
- checkpoints/reference/representations/relative_pose.pt
- results/reference/independent_validation/
- configs/distillation/reference_pose_attention.json

## Original implementation

train_pose_attention_resume_dagger.py

Original experiment directory:

experiments/state_bottleneck/pose_attention_dagger_resume_20261124T040606Z

## Canonical implementation

- rl/pose_attention_runtime.py
- rl/train_pose_attention_dagger.py
- evaluation/evaluate_controller.py
- controllers/pose_attention_policy.py
- models/spatial_pose_attention.py

