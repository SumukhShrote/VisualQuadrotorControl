"""Numerical equivalence tests for canonical visual models."""

from pathlib import Path
import importlib.util
import sys
import json

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.temporal_transformer import TemporalEstimator
from models.spatial_pose_attention import PoseAttention
from controllers.visual_policy import Policy
from controllers.pose_attention_policy import Augmented

SB = ROOT / "experiments/state_bottleneck"
REF = ROOT / "checkpoints/reference"

torch.manual_seed(20261124)
torch.set_num_threads(4)

ATOL = 1e-5
RTOL = 1e-5


def load_module(name, path):

    spec = importlib.util.spec_from_file_location(
        name, path
    )

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module

    spec.loader.exec_module(module)

    return module


def load_checkpoint(path):

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def compare(name, reference, candidate):

    error = (
        reference - candidate
    ).abs()

    print(
        f"[EQUIVALENCE] {name}: "
        f"max_abs={error.max().item():.9e}, "
        f"mean_abs={error.mean().item():.9e}"
    )

    torch.testing.assert_close(
        candidate,
        reference,
        atol=ATOL,
        rtol=RTOL,
    )

    print("[PASS]", name)

    return error.max().item()


# ============================================================
# INPUTS
# ============================================================

# Random inputs exercise the numerical implementation.
# Recorded inputs exercise the actual observation distribution.

corners = torch.rand(128, 256)
actions = torch.rand(128, 8, 4) * 2 - 1

inputs = {
    "random": (corners, actions)
}

recorded_path = (
    ROOT
    / "results/diagnostics/frozen_representation"
    / "seed_9513_distill/frozen_features.npz"
)

if recorded_path.is_file():

    with np.load(recorded_path) as data:

        print("[RECORDED DATA KEYS]", list(data.keys()))

        corner_key = next(
            (
                k for k in ("corners", "corners256", "observations")
                if k in data
            ),
            None,
        )

        action_key = next(
            (
                k for k in ("actions8", "action_history")
                if k in data
            ),
            None,
        )

        if corner_key and action_key:

            c = torch.from_numpy(
                data[corner_key][:512]
            ).float().reshape(-1, 256)

            h = torch.from_numpy(
                data[action_key][:512]
            ).float().reshape(-1, 8, 4)

            inputs["recorded"] = (c, h)

            print("[PASS] Recorded inputs loaded")

        else:

            print(
                "[WARN] Recorded input key names require inspection"
            )


# ============================================================
# ORIGINAL SUPERVISED ESTIMATOR
# ============================================================

legacy_estimator_module = load_module(
    "legacy_temporal_transformer",
    SB / "supervised_multiscale_transformer.py",
)

original_estimator = (
    legacy_estimator_module.TemporalEstimator()
).eval()

canonical_estimator = TemporalEstimator().eval()

# Same weights in both implementations.
canonical_estimator.load_state_dict(
    original_estimator.state_dict(),
    strict=True,
)

results = {}

with torch.no_grad():

    for label, (c, h) in inputs.items():

        results[f"estimator_{label}"] = compare(
            f"Supervised estimator / {label}",
            original_estimator(c.reshape(-1, 32, 8)),
            canonical_estimator(c.reshape(-1, 32, 8)),
        )


# ============================================================
# ORIGINAL K=8 CONTROLLER
# ============================================================

legacy_policy_module = load_module(
    "legacy_action_history",
    SB / "launch_action_history_study.py",
)

legacy = legacy_policy_module.Policy(8).eval()

canonical = Policy(8).eval()

checkpoint = load_checkpoint(
    REF / "visual_baseline/ppo_checkpoint.pt"
)

actor_state = {
    k[6:]: v

    for k, v in checkpoint["model_state_dict"].items()

    if k.startswith("actor.")
}

legacy.load_state_dict(
    actor_state,
    strict=True,
)

canonical.load_state_dict(
    actor_state,
    strict=True,
)

reference_old = torch.jit.load(
    str(REF / "visual_baseline/policy.ts"),
    map_location="cpu",
).eval()

with torch.no_grad():

    for label, (c, h) in inputs.items():

        results[f"base_{label}"] = compare(
            f"Original K=8 policy / {label}",
            legacy(c, h),
            canonical(c, h),
        )

        results[f"base_export_{label}"] = compare(
            f"Original TorchScript / {label}",
            reference_old(c, h),
            canonical(c, h),
        )


# ============================================================
# FINAL AUGMENTED CONTROLLER
# ============================================================

final_checkpoint = load_checkpoint(
    REF / "pose_attention/checkpoint.pt"
)

print()
print("[FINAL CHECKPOINT KEYS]")

for key, value in final_checkpoint.items():

    if isinstance(value, dict):

        print(
            key,
            "dictionary keys:",
            list(value.keys())[:8],
        )

    elif torch.is_tensor(value):

        print(key, tuple(value.shape))

    else:

        print(key, type(value).__name__)


model = Augmented(
    canonical,
    PoseAttention(),
).eval()

loaded = False

# Handle a complete augmented model state dictionary.

for key in (
    "model_state_dict",
    "state_dict",
    "model",
):

    state = final_checkpoint.get(key)

    if not isinstance(state, dict):
        continue

    if set(state.keys()) == set(model.state_dict().keys()):

        model.load_state_dict(
            state,
            strict=True,
        )

        loaded = True

        print("[PASS] Complete augmented checkpoint loaded")

        break

# Handle checkpoints that store only the added pose branch
# and expanded controller head.

if not loaded:

    for pose_key in (
        "pose",
        "pose_state_dict",
        "pose_model_state_dict",
    ):

        for head_key in (
            "head",
            "head_state_dict",
            "policy_head",
        ):

            ps = final_checkpoint.get(pose_key)
            hs = final_checkpoint.get(head_key)

            if not (
                isinstance(ps, dict)
                and isinstance(hs, dict)
            ):
                continue

            if (
                set(ps) != set(model.pose.state_dict())
                or set(hs) != set(model.head.state_dict())
            ):
                continue

            model.pose.load_state_dict(
                ps, strict=True
            )

            model.head.load_state_dict(
                hs, strict=True
            )

            loaded = True

            print(
                "[PASS] Pose and head loaded:",
                pose_key, head_key,
            )

            break

        if loaded:
            break

if not loaded:

    raise RuntimeError(
        "Unsupported final checkpoint format. "
        "Inspect the checkpoint keys printed above."
    )

reference_final = torch.jit.load(
    str(REF / "pose_attention/policy.ts"),
    map_location="cpu",
).eval()

model.eval()

with torch.no_grad():

    for label, (c, h) in inputs.items():

        features = model.features(c, h)

        assert features.shape[-1] == 293

        results[f"final_{label}"] = compare(
            f"Final augmented policy / {label}",
            reference_final(c, h),
            model(c, h),
        )


# ============================================================
# CHECK TRAINABILITY
# ============================================================

assert not any(
    p.requires_grad
    for p in model.base.parameters()
)

assert not any(
    p.requires_grad
    for p in model.pose.parameters()
)

assert all(
    p.requires_grad
    for p in model.head.parameters()
)

print("[PASS] Frozen representation contract preserved")

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print("Total parameters:", total)
print("Trainable parameters:", trainable)

# ============================================================
# SAVE EQUIVALENCE REPORT
# ============================================================

report = {
    "passed": True,
    "absolute_tolerance": ATOL,
    "relative_tolerance": RTOL,
    "input_sets": list(inputs),
    "maximum_absolute_errors": results,
    "total_parameters": total,
    "trainable_parameters": trainable,
}

out = ROOT / "docs/cleanup/model_equivalence.json"

out.write_text(
    json.dumps(report, indent=2) + "\n"
)

print()
print("=" * 80)
print("ALL MODEL EQUIVALENCE TESTS PASSED")
print("=" * 80)
