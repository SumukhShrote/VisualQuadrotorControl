# Visual Quadrotor Learning — Research Timeline

This document records the research progression and the
main decisions that led to the reference visual controller.

Historical results are not necessarily controlled
head-to-head comparisons.

## Phase 1 — Privileged-state teacher

Period: June 2026.

Established a privileged-state reinforcement learning
controller for quadrotor hovering.

Reference checkpoint:

`logs/rsl_rl/CTBR_Hover_Teacher/2026-06-25_12-41-52/model_749.pt`

The teacher reliably completed approximately 499 control
steps under the earlier 10-second evaluation protocol.

## Phase 2 — Visual behavioral cloning and TCN distillation

Period: June–July 2026.

Investigated a temporal convolutional student operating on
32 frames of ordered marker corners.

Behavioral cloning with approximately 500,000 transitions
achieved a mean episode length of approximately 122 steps.

Sequential DAgger improved mean episode length to
approximately 250 steps at two million transitions.

A historical checkpoint achieved 38.15% full-episode
survival across 2,000 evaluations.

Auxiliary physical-state supervision improved offline
teacher-action prediction but did not consistently
produce stable closed-loop control.

Decision:
Better imitation accuracy alone was insufficient.

## Phase 3 — Alternative visual RL and representation learning

Period: July–November 2026.

Investigated asymmetric PPO, teacher-initialized PPO,
auxiliary supervision, visual observation alternatives
and state reconstruction.

Earlier Dreamer experiments belong to a separate
world-model research track and should retain their own
historical records.

## Phase 4 — Geometric observability and temporal attention

Period: November 19–20, 2026.

Collected approximately 100,000 student-controlled samples
across five curriculum levels.

The mature-history dataset contained 80,287 samples:

- Training: 57,377.
- Validation: 12,146.
- Test: 10,764.

Trained multi-timescale Transformers to reconstruct
position, gravity, marker direction, distance, linear
velocity and angular velocity.

Investigated MSE, Huber, masked and unmasked supervision.

Compared learned reconstruction with calibrated
geometric reconstruction.

On the earlier matched evaluation dataset:

| Quantity | Geometry RMSE | Transformer RMSE |
|---|---:|---:|
| Position | 0.000122 m | 0.0799 m |
| Linear velocity | 0.0300 m/s | 0.1892 m/s |
| Angular velocity | 0.2939 rad/s | 0.4397 rad/s |

Decision:
The observations contained useful physical information,
but the learned representations did not recover it with
the accuracy of calibrated geometry.

## Phase 5 — Transformer policy and action-history experiments

Period: November 20–21, 2026.

Investigated direct action prediction, supervised
representation pretraining and joint action/state
supervision.

The pretrained actor was initialized from the plain Huber
Transformer checkpoint.

Compared zero, one and eight previous executed actions.

Subsequent K=8 DAgger experiments produced a much stronger
visual controller. A historical evaluation achieved
126/128 survival and 63/128 precision holds.

Decision:
Action history was retained, but precise hovering remained
unresolved.

## Phase 6 — PPO fine-tuning

Period: November 21–23, 2026.

Investigated:

- Asymmetric privileged critics.
- Critic warmup.
- Conservative actor learning rates.
- Imitation anchoring.
- KL-based update constraints.
- Longer-horizon GAE.
- Near-hover reset mixtures.
- Increased stable-hover reward weights.
- Continuing-hover reward objectives.

The experiments did not consistently improve deterministic
precision hovering.

Local update-direction probes revealed cases where stochastic
return improved while deterministic precision declined.

Decision:
Stop further blind PPO tuning and diagnose the physical
failure mode directly.

## Phase 7 — Frozen representation and equilibrium diagnosis

Date: November 23, 2026.

Analyzed recorded visual features, actions, simulator state
and closed-loop trajectories.

The controller often stabilized at an incorrect physical
position while achieving very low velocity.

Across 126 surviving flights:

- Mean settled position offset: approximately 11.77 cm.
- Mean lateral error: approximately 10.99 cm.
- Mean settled speed: approximately 0.0125 m/s.
- Position variability around equilibrium: approximately 0.72 cm.

The controller compensated for lateral displacement through
yaw rotation, keeping the marker approximately centered.

Geometric analysis showed a correlation near -0.99991
between estimated yaw and lateral displacement.

Incorrect-task diagnostic archives were invalidated.
Only the `_distill` archives are accepted as reference
diagnostic evidence.

Decision:
Improve explicit geometric information rather than
continue tuning the controller objective.

## Phase 8 — Independently supervised spatial pose attention

Date: November 24, 2026.

Introduced spatial attention over the four current marker
corners.

Trained the branch using analytic projection with
independently sampled position and orientation, including
deliberately yaw-compensated configurations.

Archived simulator pose-estimation results:

| Quantity | RMSE |
|---|---:|
| Longitudinal position | 3.66 mm |
| Lateral position | 1.13 mm |
| Vertical position | 2.53 mm |
| Yaw | 0.063 degrees |

These estimates are not a controlled comparison against
the earlier temporal estimator because the datasets and
reference-frame conventions differ.

## Phase 9 — Expanded curriculum and frozen-encoder DAgger

Date: November 24, 2026.

Expanded initial conditions to independently vary position,
orientation and velocity, including targeted compensation
scenarios.

Verified recoverability using the privileged teacher.

Integrated the spatial features and explicit pose predictions
into the existing controller using zero-initialized new
input connections.

Froze all representation branches and trained the expanded
policy head through DAgger.

Selected round 15 after 1,474,560 newly collected transitions.

## Phase 10 — Independent reference evaluation

Date: November 24, 2026.

Independent 10-second evaluation:

- Two reset domains.
- Two seeds.
- 1,024 total student episodes.
- 100% survival and precision hold.

Independent 60-second evaluation:

- Two reset domains.
- One independent seed.
- 256 total student episodes.
- 100% continuous precision hold between 10 and 60 seconds.

Expanded-domain settled lateral position error:

- Previous controller: 37.42 cm.
- Updated controller: 0.82 cm.
- Privileged teacher: 0.65 cm.

The combined pose-representation and curriculum intervention
established near-teacher-level simulated visual control.

## Phase 11 — Repository consolidation

Date: November 24, 2026.

- Created independent recovery snapshot.
- Indexed 363 historical artifact groups.
- Verified critical checkpoint hashes.
- Organized canonical reference assets.
- Extracted reusable temporal and spatial model modules.
- Verified zero numerical difference against legacy models
  and the successful TorchScript export.

No new training was performed during refactoring.

## Remaining documentation requirements

The timeline above is a summary, not a complete individual
experiment registry.

Exact configurations, dates, seeds and outcomes of historical
runs must be recovered from the archived experiment records.

The current results do not establish real-camera robustness,
cross-robot dynamics transfer or world-model planning.
