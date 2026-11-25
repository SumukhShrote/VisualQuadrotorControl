# Research and Implementation Changelog

## November 24, 2026 — Reference visual controller

- Added independently supervised spatial pose attention.
- Introduced an expanded reset curriculum.
- Trained the controller head using frozen-feature DAgger.
- Selected round 15 as the reference controller.
- Achieved 1024/1024 successful independent 10-second episodes.
- Achieved 256/256 successful independent 60-second episodes.
- Established continuous precision hold after 10 seconds.
- Extracted canonical model, checkpoint and evaluation modules.
- Verified zero numerical error against the original model exports.
- Reproduced closed-loop performance across 512 episodes.

## November 23, 2026 — Physical failure diagnosis

- Identified lateral displacement compensated by yaw rotation.
- Analyzed frozen visual representations and physical state.
- Identified discrepancies between stochastic PPO return and
  deterministic precision-hover performance.
- Discontinued blind PPO hyperparameter tuning.

## November 20–23, 2026 — Visual controller optimization

- Investigated action-history conditioning.
- Trained Transformer-based DAgger controllers.
- Investigated asymmetric PPO, imitation anchoring, critic warmup,
  extended GAE horizons and stable-hover reward modifications.
- Retained representative baselines and negative experimental results.

## November 19, 2026 — Structured visual representation

- Established supervised physical-state reconstruction benchmarks.
- Compared learned state estimation against geometric reconstruction.
- Introduced independent multi-timescale Transformer branches.

## June–July 2026 — Initial visual control experiments

- Established the privileged PPO teacher.
- Investigated corners-only behavioral cloning.
- Developed temporal convolutional student policies.
- Investigated sequential DAgger and auxiliary state supervision.
- Established historical visual-control baselines.

## Repository consolidation

- Separated reusable architectures from experimental scripts.
- Preserved selected reference and baseline checkpoints.
- Organized research records and evaluation outputs.
- Permanently retired the old Dreamer implementation and artifacts.
- Removed redundant checkpoints and generated caches.
- Retained existing Git history.

See docs/research/RESEARCH_TIMELINE.md for research context and
docs/cleanup/ for repository modification provenance.

- Separated neural architectures into `networks/`; `models/` now contains simulation assets only. Network source bytes and parameter names are unchanged.
