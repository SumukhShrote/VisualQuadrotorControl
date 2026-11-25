"""Checkpoint loading for reference visual quadrotor controllers."""

from pathlib import Path
import hashlib
import json

import torch

from networks.temporal_transformer import TemporalEstimator
from networks.spatial_pose_attention import PoseAttention
from controllers.visual_policy import Policy, Actor
from controllers.pose_attention_policy import Augmented


ROOT = Path(__file__).resolve().parents[1]

REFERENCE = ROOT / "checkpoints/reference"


def sha256(path):
    """Calculate a file's SHA-256 digest."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_reference_artifacts():
    """Verify all canonical reference artifacts."""

    manifest = json.loads(
        (REFERENCE / "manifest.json").read_text()
    )

    for item in manifest:

        path = ROOT / item["canonical_path"]

        if not path.is_file():
            raise FileNotFoundError(path)

        actual = sha256(path)

        if actual != item["sha256"]:
            raise RuntimeError(
                f"Reference artifact hash mismatch: {path}"
            )

    return len(manifest)


def load_local_checkpoint(path, device="cpu"):
    """Load a trusted local research checkpoint."""

    return torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )


def load_pretrained_actor(
    checkpoint=None,
    device="cpu",
):
    """Load the original pretrained visual imitation actor."""

    checkpoint = Path(
        checkpoint
        or REFERENCE / "visual_baseline/pretrained_actor.pt"
    )

    data = load_local_checkpoint(checkpoint)

    model = Actor(TemporalEstimator)

    model.load_state_dict(
        data["actor_state_dict"],
        strict=True,
    )

    return model.to(device).eval()


def load_visual_baseline(
    checkpoint=None,
    device="cpu",
    frozen=True,
):
    """Reconstruct the established K=8 PPO visual baseline."""

    checkpoint = Path(
        checkpoint
        or REFERENCE / "visual_baseline/ppo_checkpoint.pt"
    )

    data = load_local_checkpoint(checkpoint)

    state = data["model_state_dict"]

    if data.get("k") == 8 and "head.0.weight" in state:
        actor_state = state
    else:
        actor_state = {
            key[6:]: value
            for key, value in state.items()
            if key.startswith("actor.")
        }

    model = Policy(8)

    model.load_state_dict(
        actor_state,
        strict=True,
    )

    model = model.to(device).eval()

    if frozen:
        model.requires_grad_(False)

    return model


def load_reference_controller(
    checkpoint=None,
    base_checkpoint=None,
    device="cpu",
    train_head=False,
):
    """Reconstruct the final temporal-spatial visual controller.

    The three temporal branches and spatial pose branch are frozen.
    The expanded control head can optionally be made trainable.
    """

    checkpoint = Path(
        checkpoint
        or REFERENCE / "pose_attention/checkpoint.pt"
    )

    data = load_local_checkpoint(checkpoint)

    base = load_visual_baseline(
        checkpoint=base_checkpoint,
        device=device,
        frozen=True,
    )

    if "base_state" in data:
        base.load_state_dict(data["base_state"], strict=True)

    pose = PoseAttention().to(device)

    pose.load_state_dict(
        data["pose"],
        strict=True,
    )

    model = Augmented(base, pose).to(device)

    model.head.load_state_dict(
        data["head"],
        strict=True,
    )

    model.eval()

    model.base.requires_grad_(False)
    model.pose.requires_grad_(False)
    model.head.requires_grad_(train_head)

    return model


def load_deployment_policy(
    checkpoint=None,
    device="cpu",
):
    """Load the standalone exported deployment policy."""

    checkpoint = Path(
        checkpoint
        or REFERENCE / "pose_attention/policy.ts"
    )

    return torch.jit.load(
        str(checkpoint),
        map_location=device,
    ).eval()
