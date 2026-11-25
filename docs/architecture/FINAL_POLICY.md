# Temporal-Spatial Visual Quadrotor Controller

## Reference controller

The reference controller is the selected round-15 frozen-encoder
DAgger checkpoint from November 24, 2026.

Environment:
`Isaac-QuadrotorHoverStudentDistill-v0`

Control frequency: 50 Hz.

## Observation contract

- 32 consecutive frames of eight normalized marker-corner coordinates.
- Eight previous executed four-dimensional CTBR actions.
- No IMU or privileged physical state at deployment.
- No explicit PnP computation at deployment.

The current observation contract is visual plus action history,
rather than the original strictly corners-only contract.

## Temporal representation

Three independent Transformer branches:

| Branch | Window | Feature dimension | Supervised target |
|---|---:|---:|---|
| Geometry | 8 | 64 | Position, gravity, marker direction, distance |
| Linear velocity | 32 | 64 | Body-frame linear velocity |
| Angular velocity | 8 | 64 | Body-frame angular velocity |

Each temporal branch contains:

- Linear input projection: 8 to 64.
- Learned temporal positional embeddings.
- Two Transformer encoder layers.
- Four attention heads per layer.
- Feedforward dimension 128.
- GELU activations.
- Dropout 0.05.
- Learned-query multihead attention pooling.
- Residual addition of the most recent encoded token.
- Layer normalization.
- Supervised decoder: 64 to 96 to target dimension.

Only the internal 64-dimensional features from each temporal
branch are supplied to the final policy.

The explicit predictions of the original temporal branches
are not supplied to the policy.

## Spatial pose representation

Spatial self-attention over the four ordered corners of
the most recent visual observation.

Each corner token contains nine features derived from its
coordinates, centered and scale-normalized shape, marker
center, apparent size and opposite-edge differences.

Architecture:

- Embedding: 9 to 64.
- Learned spatial positional embeddings.
- Single-head spatial self-attention.
- QKV projection: 64 to 192.
- Attention residual and layer normalization.
- Feedforward network: 64 to 128 to 64.
- Flattened spatial readout: 256 to 128 to 64.
- Pose decoder: 64 to 64 to 5.

Supervised targets:

    [x, y, z, sin(yaw), cos(yaw) - 1]

Normalization scales:

    [0.3, 0.3, 0.3, 0.25, 0.1]

Both the 64-dimensional spatial representation and its five
normalized physical predictions are supplied to the controller.

If the latest visual observation is entirely zero, the
spatial features and explicit predictions are masked to zero.

## Final feature ordering

| Feature | Dimensions | Indices |
|---|---:|---|
| Geometry | 64 | 0:64 |
| Linear velocity | 64 | 64:128 |
| Angular velocity | 64 | 128:192 |
| Action history | 32 | 192:224 |
| Spatial pose features | 64 | 224:288 |
| Normalized pose predictions | 5 | 288:293 |

The controller MLP is:

    293 -> 256 -> 128 -> 4

Hidden activation: GELU.
Output activation: Tanh.

## Training

The original temporal representation was pretrained using
supervised physical-state reconstruction.

The original pretrained actor references the plain Huber
Transformer checkpoint.

Subsequent action-supervised imitation experiments allowed
the temporal representation to be fine-tuned.

The spatial pose branch was independently supervised using
synthetic marker observations with independently sampled
positions and orientations.

During final DAgger training, all four representation
branches were frozen. Only the expanded policy head was
optimized.

## Parameter count

Total stored parameters: 555,965.
Trainable parameters during final DAgger: 108,676.
Frozen parameters: 447,289.

The total includes retained modules whose outputs are not
used in the final action computation.

## Canonical implementation

- `models/temporal_transformer.py`
- `models/spatial_pose_attention.py`
- `controllers/visual_policy.py`
- `controllers/pose_attention_policy.py`
- `utils/checkpoint_utils.py`

## Refactoring validation

On November 24, 2026, the canonical model was compared
against the original implementation and TorchScript export.

All eight numerical equivalence comparisons returned
exactly zero maximum absolute error on the tested inputs.

This establishes model-level numerical equivalence on
the tested CPU inputs. A post-refactoring closed-loop
simulation reproduction gate remains separate.

Evidence:

- `docs/cleanup/model_equivalence.json`
- `tests/check_reference_model_equivalence.py`
