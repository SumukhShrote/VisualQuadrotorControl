

## Final spatial-attention controller

Best checkpoint: round 15.

- Final training-script evaluation: 5119/5120 survival.
- Final two-second precision holds: 5119/5120.
- Spatial pose branch trained from random initialization.
- Temporal representation inherited from fresh DAgger round 2.
- Representation branches frozen during final head training.
- Intermediate student PPO fine-tuning omitted.
- Eager/export maximum numerical discrepancy: 0.0.

## Independent validation

Curriculum level 1.0, original and expanded reset distributions.

### 10-second evaluation

| Policy | Domain | Survival | Holds | Position (m) | Lateral (m) | Speed (m/s) |
|---|---|---:|---:|---:|---:|---:|
| candidate | original | 512/512 | 512/512 | 0.02670587 | 0.00664255 | 0.00043961 |
| candidate | expanded | 511/512 | 511/512 | 0.02611022 | 0.00675644 | 0.00044364 |
| student | original | 512/512 | 512/512 | 0.03024809 | 0.00839687 | 0.00051239 |
| student | expanded | 510/512 | 510/512 | 0.02953572 | 0.00847320 | 0.00050983 |
| teacher | original | 512/512 | 512/512 | 0.02693984 | 0.00670368 | 0.00029153 |
| teacher | expanded | 512/512 | 512/512 | 0.02634177 | 0.00679284 | 0.00029610 |

### 60-second evaluation

| Policy | Domain | Survival | Holds | Position (m) | Lateral (m) | Speed (m/s) |
|---|---|---:|---:|---:|---:|---:|
| candidate | original | 128/128 | 128/128 | 0.02675213 | 0.00658623 | 0.00038138 |
| candidate | expanded | 128/128 | 128/128 | 0.02699756 | 0.00657211 | 0.00038032 |
| student | original | 128/128 | 128/128 | 0.03020147 | 0.00835079 | 0.00048241 |
| student | expanded | 128/128 | 128/128 | 0.03053909 | 0.00834752 | 0.00048745 |
| teacher | original | 128/128 | 128/128 | 0.02705533 | 0.00672032 | 0.00020882 |
| teacher | expanded | 128/128 | 128/128 | 0.02728595 | 0.00667812 | 0.00021810 |

All three controllers maintained continuous precision after 10 seconds in every 60-second evaluation episode.

## Conclusions

The consolidated training pipeline reproduced high-precision visual hover without historical visual-policy initialization or intermediate student PPO fine-tuning.

The privileged teacher and original supervised dataset were reused. The improvements combine spatial pose attention and expanded-reset curriculum training; they are not an isolated architecture ablation.

The results use analytic marker corners in simulation. Real-camera robustness, hardware transfer, and cross-robot adaptation remain unvalidated.

## End-to-end representation DAgger

All four control-relevant representation branches were jointly
optimized with the controller head during final DAgger.

- Same fresh K8 initialization and pretrained spatial branch.
- Representation learning rate: 1e-5.
- Head learning rate: 1e-4.
- 24 DAgger rounds.
- 512 parallel environments.
- 192 collection steps per round.
- 180 optimization updates per round.
- Batch size: 2048.
- Replay capacity: 180000.
- Nonzero gradients and parameter changes verified for all branches.

The original checkpoint-selection criterion selected round 18.
Round 24 tied on survival and precision holds but achieved better
settled positional accuracy, so it was selected for independent
evaluation before inspecting independent test results.

Training completed all 24 rounds. The original final export failed
its numerical equivalence check, preventing the completion marker.
Both checkpoints were subsequently recovered and exported using an
inference-only model copy without retraining.

Independent results:

| Metric | Frozen R15 | Joint R24 |
|---|---:|---:|
| 10s survival | 1023/1024 | 1023/1024 |
| 60s survival | 256/256 | 256/256 |
| Continuous precision after 10s | 100% | 100% |
| 60s original position error (m) | 0.02675213 | 0.02383385 |
| 60s expanded position error (m) | 0.02699756 | 0.02387425 |
| 60s expanded lateral error (m) | 0.00657211 | 0.00781232 |
| 60s expanded settled speed (m/s) | 0.00038032 | 0.00050031 |
| Worst expanded error after 10s (m) | 0.04486740 | 0.03785690 |

Joint training improves overall position accuracy while slightly
worsening lateral accuracy and residual velocity.

Physical-state decoding after joint optimization remains untested.
All results use analytic marker corners in simulation.
