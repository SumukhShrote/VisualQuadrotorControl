from __future__ import annotations

import argparse

import copy
import os
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import importlib.util

MODEL_FILE = Path(
    "envs/aruco_hover/agents/"
    "vision_student_teacher_probe_ready.py"
).resolve()

model_spec = importlib.util.spec_from_file_location(
    "vision_student_teacher_probe_ready_offline",
    MODEL_FILE,
)

if model_spec is None or model_spec.loader is None:
    raise ImportError(
        f"Could not load model module from {MODEL_FILE}"
    )

model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)

VisionStudentTeacher = model_module.VisionStudentTeacher


SEED = 7300

DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "TCN_PROBE_DATA",
        "dagger_data/tcn_state_probe_v1/"
        "smoke_student_level0/"
        "round_00_shard_0000.pt",
    )
)

CHECKPOINT_PATH = Path(
    "checkpoints/corners_tcn_student_v1/"
    "stage3_dagger2m/"
    "stage3_dagger2m_e70.pt"
)

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        "TCN_PROBE_OUTPUT",
        "eval_logs/corners_tcn_student_v1/"
        "tcn_state_probe_smoke",
    )
)

TARGET_NAMES = [
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "omega_x",
    "omega_y",
    "omega_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_episode_split(
    environment_ids: torch.Tensor,
    episode_ids: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split complete environment/episode groups, never individual windows."""

    groups = torch.stack(
        [
            environment_ids.to(torch.int64),
            episode_ids.to(torch.int64),
        ],
        dim=1,
    )

    unique_groups, inverse = torch.unique(
        groups,
        dim=0,
        return_inverse=True,
    )

    num_groups = unique_groups.shape[0]

    if num_groups < 3:
        raise RuntimeError(
            f"Need at least three complete episodes, found {num_groups}."
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        num_groups,
        generator=generator,
    )

    num_train = max(1, int(round(0.70 * num_groups)))
    num_val = max(1, int(round(0.15 * num_groups)))

    if num_train + num_val >= num_groups:
        num_val = 1
        num_train = num_groups - 2

    train_groups = permutation[:num_train]
    val_groups = permutation[num_train:num_train + num_val]
    test_groups = permutation[num_train + num_val:]

    train_mask = torch.isin(inverse, train_groups)
    val_mask = torch.isin(inverse, val_groups)
    test_mask = torch.isin(inverse, test_groups)

    print(
        "[SPLIT]",
        f"groups={num_groups}",
        f"train_groups={len(train_groups)}",
        f"val_groups={len(val_groups)}",
        f"test_groups={len(test_groups)}",
    )

    print(
        "[SPLIT]",
        f"train_rows={int(train_mask.sum())}",
        f"val_rows={int(val_mask.sum())}",
        f"test_rows={int(test_mask.sum())}",
    )

    return train_mask, val_mask, test_mask


class LinearProbe(nn.Module):
    def __init__(self, latent_dim: int, target_dim: int):
        super().__init__()
        self.network = nn.Linear(latent_dim, target_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class NonlinearProbe(nn.Module):
    def __init__(self, latent_dim: int, target_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, target_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


@torch.no_grad()
def encode_in_batches(
    policy: VisionStudentTeacher,
    observations: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> torch.Tensor:
    outputs = []

    for start in range(0, observations.shape[0], batch_size):
        batch = observations[start:start + batch_size].to(device)

        latent = policy.encode_temporal_state(batch)

        outputs.append(latent.cpu())

    return torch.cat(outputs, dim=0)


def train_probe(
    name: str,
    probe: nn.Module,
    train_latent: torch.Tensor,
    train_target: torch.Tensor,
    val_latent: torch.Tensor,
    val_target: torch.Tensor,
    device: torch.device,
    learning_rate: float,
    max_epochs: int = 300,
    patience: int = 30,
) -> nn.Module:
    probe = probe.to(device)

    dataset = TensorDataset(
        train_latent,
        train_target,
    )

    loader = DataLoader(
        dataset,
        batch_size=min(256, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=1.0e-5,
    )

    best_val = math.inf
    best_state = None
    stale_epochs = 0

    val_latent_device = val_latent.to(device)
    val_target_device = val_target.to(device)

    for epoch in range(1, max_epochs + 1):
        probe.train()

        train_loss_sum = 0.0
        train_rows = 0

        for latent_batch, target_batch in loader:
            latent_batch = latent_batch.to(device)
            target_batch = target_batch.to(device)

            prediction = probe(latent_batch)

            loss = F.smooth_l1_loss(
                prediction,
                target_batch,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                probe.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            rows = latent_batch.shape[0]
            train_loss_sum += float(loss.item()) * rows
            train_rows += rows

        probe.eval()

        with torch.no_grad():
            val_prediction = probe(val_latent_device)
            val_loss = F.smooth_l1_loss(
                val_prediction,
                val_target_device,
            ).item()

        train_loss = train_loss_sum / max(train_rows, 1)

        if (
            epoch == 1
            or epoch % 20 == 0
            or val_loss < best_val
        ):
            print(
                f"[{name}] "
                f"epoch={epoch:03d} "
                f"train={train_loss:.7f} "
                f"val={val_loss:.7f}"
            )

        if val_loss < best_val - 1.0e-7:
            best_val = val_loss
            best_state = copy.deepcopy(
                probe.state_dict()
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(
                f"[{name}] Early stopping at epoch {epoch}; "
                f"best_val={best_val:.7f}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"{name}: no valid checkpoint was produced."
        )

    probe.load_state_dict(best_state)
    probe.eval()

    return probe


def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    error = prediction - target

    rmse = torch.sqrt(
        torch.mean(error.square(), dim=0)
    )

    mae = torch.mean(
        error.abs(),
        dim=0,
    )

    bias = torch.mean(
        error,
        dim=0,
    )

    target_mean = target.mean(
        dim=0,
        keepdim=True,
    )

    residual_sum = torch.sum(
        error.square(),
        dim=0,
    )

    total_sum = torch.sum(
        (target - target_mean).square(),
        dim=0,
    )

    r2 = 1.0 - residual_sum / total_sum.clamp_min(1.0e-12)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
    }


def print_metrics(
    title: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    metrics = calculate_metrics(
        prediction,
        target,
    )

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    print(
        f"{'target':14s}"
        f"{'RMSE':>12s}"
        f"{'MAE':>12s}"
        f"{'bias':>12s}"
        f"{'R2':>12s}"
    )

    for index, target_name in enumerate(TARGET_NAMES):
        print(
            f"{target_name:14s}"
            f"{metrics['rmse'][index].item():12.6f}"
            f"{metrics['mae'][index].item():12.6f}"
            f"{metrics['bias'][index].item():12.6f}"
            f"{metrics['r2'][index].item():12.6f}"
        )

    groups = {
        "position": slice(0, 3),
        "linear_velocity": slice(3, 6),
        "angular_velocity": slice(6, 9),
        "gravity": slice(9, 12),
    }

    print()
    print("Grouped means")

    for group_name, group_slice in groups.items():
        print(
            f"{group_name:20s} "
            f"RMSE={metrics['rmse'][group_slice].mean().item():.6f} "
            f"R2={metrics['r2'][group_slice].mean().item():.6f}"
        )


@torch.no_grad()
def predict_physical_units(
    probe: nn.Module,
    latent: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    prediction_normalized = probe(
        latent.to(device)
    ).cpu()

    return (
        prediction_normalized * target_std
        + target_mean
    )


def make_ablation_observations(
    observations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sequence = observations.reshape(-1, 32, 8)

    newest_repeated = (
        sequence[:, -1:, :]
        .repeat(1, 32, 1)
        .reshape(-1, 256)
    )

    reversed_history = torch.flip(
        sequence,
        dims=[1],
    ).reshape(-1, 256)

    generator = torch.Generator().manual_seed(SEED + 99)
    permutation = torch.randperm(
        32,
        generator=generator,
    )

    shuffled_history = sequence[
        :,
        permutation,
        :,
    ].reshape(-1, 256)

    return {
        "normal": observations,
        "newest_repeated": newest_repeated,
        "reversed": reversed_history,
        "shuffled": shuffled_history,
    }


def print_ablation_summary(
    probe_name: str,
    probe: nn.Module,
    policy: VisionStudentTeacher,
    test_observations: torch.Tensor,
    test_target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> None:
    print()
    print("=" * 100)
    print(f"{probe_name}: TEMPORAL ABLATIONS")
    print("=" * 100)

    print(
        f"{'input':20s}"
        f"{'vel_RMSE':>14s}"
        f"{'vel_R2':>14s}"
        f"{'omega_RMSE':>14s}"
        f"{'omega_R2':>14s}"
    )

    ablations = make_ablation_observations(
        test_observations
    )

    for ablation_name, ablation_obs in ablations.items():
        latent = encode_in_batches(
            policy,
            ablation_obs,
            device,
        )

        prediction = predict_physical_units(
            probe,
            latent,
            target_mean,
            target_std,
            device,
        )

        metrics = calculate_metrics(
            prediction,
            test_target,
        )

        velocity_rmse = metrics["rmse"][3:6].mean().item()
        velocity_r2 = metrics["r2"][3:6].mean().item()

        omega_rmse = metrics["rmse"][6:9].mean().item()
        omega_r2 = metrics["r2"][6:9].mean().item()

        print(
            f"{ablation_name:20s}"
            f"{velocity_rmse:14.6f}"
            f"{velocity_r2:14.6f}"
            f"{omega_rmse:14.6f}"
            f"{omega_r2:14.6f}"
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen probes on a corners-TCN checkpoint."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Probe shard containing observations and teacher_observations.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to save the trained probe.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = args.data
    output_dir = args.output

    seed_everything(SEED)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("[INFO] Device:", device)
    print("[INFO] Data:", data_path)
    print("[INFO] Checkpoint:", CHECKPOINT_PATH)

    data = torch.load(
        data_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "observations",
        "teacher_observations",
        "environment_ids",
        "episode_ids",
        "episode_steps",
    }

    missing = sorted(
        required_keys - set(data.keys())
    )

    if missing:
        raise KeyError(
            f"Probe shard is missing keys: {missing}"
        )

    observations = data["observations"].float()
    teacher_observations = (
        data["teacher_observations"].float()
    )

    environment_ids = data["environment_ids"].long()
    episode_ids = data["episode_ids"].long()
    episode_steps = data["episode_steps"].long()

    # Use only fully populated histories for this first test.
    mature = episode_steps >= 31

    observations = observations[mature]
    teacher_observations = teacher_observations[mature]
    environment_ids = environment_ids[mature]
    episode_ids = episode_ids[mature]

    # Predict position, linear velocity, angular velocity, and gravity.
    targets = teacher_observations[:, 0:12]

    print(
        "[INFO] Mature rows retained:",
        observations.shape[0],
    )

    train_mask, val_mask, test_mask = make_episode_split(
        environment_ids,
        episode_ids,
        seed=SEED,
    )

    target_mean = targets[train_mask].mean(
        dim=0
    )

    target_std = targets[train_mask].std(
        dim=0
    ).clamp_min(1.0e-6)

    targets_normalized = (
        targets - target_mean
    ) / target_std

    policy = VisionStudentTeacher(
        num_student_obs=256,
        num_teacher_obs=20,
        num_actions=4,
        student_hidden_dims=[256, 256],
        teacher_hidden_dims=[64, 64],
        activation="elu",
        init_noise_std=0.01,
        history_length=32,
        token_dim=8,
        temporal_channels=128,
        use_gru=False,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "student_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain student_state_dict. "
            f"Keys: {list(checkpoint.keys())}"
        )

    policy.student.load_state_dict(
        checkpoint["student_state_dict"],
        strict=True,
    )

    policy.eval()

    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    print("[PASS] Loaded and froze epoch-70 student.")

    latent = encode_in_batches(
        policy,
        observations,
        device,
    )

    print(
        "[INFO] Latent shape:",
        tuple(latent.shape),
    )

    train_latent = latent[train_mask]
    val_latent = latent[val_mask]
    test_latent = latent[test_mask]

    train_target = targets_normalized[train_mask]
    val_target = targets_normalized[val_mask]
    test_target_physical = targets[test_mask]

    linear_probe = train_probe(
        name="linear",
        probe=LinearProbe(
            latent_dim=128,
            target_dim=12,
        ),
        train_latent=train_latent,
        train_target=train_target,
        val_latent=val_latent,
        val_target=val_target,
        device=device,
        learning_rate=3.0e-3,
    )

    nonlinear_probe = train_probe(
        name="nonlinear",
        probe=NonlinearProbe(
            latent_dim=128,
            target_dim=12,
        ),
        train_latent=train_latent,
        train_target=train_target,
        val_latent=val_latent,
        val_target=val_target,
        device=device,
        learning_rate=1.0e-3,
    )

    linear_prediction = predict_physical_units(
        linear_probe,
        test_latent,
        target_mean,
        target_std,
        device,
    )

    nonlinear_prediction = predict_physical_units(
        nonlinear_probe,
        test_latent,
        target_mean,
        target_std,
        device,
    )

    print_metrics(
        "FROZEN EPOCH-70 TCN — LINEAR PROBE",
        linear_prediction,
        test_target_physical,
    )

    print_metrics(
        "FROZEN EPOCH-70 TCN — NONLINEAR PROBE",
        nonlinear_prediction,
        test_target_physical,
    )

    print_ablation_summary(
        probe_name="LINEAR PROBE",
        probe=linear_probe,
        policy=policy,
        test_observations=observations[test_mask],
        test_target=test_target_physical,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )

    print_ablation_summary(
        probe_name="NONLINEAR PROBE",
        probe=nonlinear_probe,
        policy=policy,
        test_observations=observations[test_mask],
        test_target=test_target_physical,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )

    output = {
        "linear_probe_state_dict":
            linear_probe.cpu().state_dict(),

        "nonlinear_probe_state_dict":
            nonlinear_probe.cpu().state_dict(),

        "target_mean":
            target_mean,

        "target_std":
            target_std,

        "target_names":
            TARGET_NAMES,

        "seed":
            SEED,

        "data_path":
            str(data_path),

        "student_checkpoint":
            str(CHECKPOINT_PATH),
    }

    output_path = output_dir / "frozen_probe_smoke.pt"

    torch.save(
        output,
        output_path,
    )

    print()
    print("[DONE] Saved probe checkpoint:", output_path)


if __name__ == "__main__":
    main()
