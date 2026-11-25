# Dreamer Research Track Retirement

Date: November 24, 2026.

The old Dreamer implementation, Dreamer-specific simulation environments and historical Dreamer artifacts covered by the approved deletion plan were permanently removed.

Only four historical checkpoints, totaling approximately 0.483 GiB, were verified to have byte-identical copies in the separate QuadcopterDreamer repository.

Exact checkpoint-level reproducibility of many earlier Dreamer experiments was intentionally discontinued.

The Dreamer-dependent scratch asymmetric PPO environment was retired, but its reusable policy was retained in controllers/corners_asymmetric_policy.py.

The four standard Gym environment registrations and the relocated asymmetric policy passed import validation.

The separate QuadcopterDreamer repository was not modified.

See docs/cleanup/dreamer_removal.jsonl for deletion provenance.
