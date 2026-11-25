# Research Artifact Retention

The canonical reference checkpoints and selected historical baselines are retained.

Raw evaluation tensors were organized under results/historical_evaluations/raw_tensors, preserving their original relative directory structure.

Intermediate numbered student-training snapshots were pruned while retaining the earliest and latest snapshots and known run-specific checkpoint references.

Exact resumption from every historical intermediate training iteration is no longer supported.

Historical commands may contain original artifact locations. The relocation journal records their new locations.

No training or simulation was performed during this operation.

Detailed decisions: docs/cleanup/final_organization.jsonl
