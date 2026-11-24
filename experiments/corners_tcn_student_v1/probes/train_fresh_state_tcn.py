from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 7300

DEFAULT_DATA = Path(
    "dagger_data/tcn_state_probe_v1/"
    "pooled_student_epoch70_all_levels.pt"
)

DEFAULT_OUTPUT = Path(
    "eval_logs/corners_tcn_student_v1/"
    "fresh_state_supervised_tcn"
)

MODEL_FILE = Path(
    "envs/aruco_hover/agents/"
    "vision_student_teacher_probe_ready.py"
).resolve()

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


def load_model_class():
    spec = importlib.util.spec_from_file_location(
        "vision_student_teacher_probe_ready_offline",
        MODEL_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {MODEL_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.VisionStudentTeacher


VisionStudentTeacher = load_model_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a fresh corners-only TCN directly on privileged "
            "state-estimation targets."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3.0e-4,
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_episode_split(
    environment_ids: torch.Tensor,
    episode_ids: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            f"Need at least three episode groups, found {num_groups}."
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        num_groups,
        generator=generator,
    )

    num_train = max(1, int(round(0.70 * num_groups)))
    num_val = max(1, int(round(0.15 * num_groups)))

    if num_train + num_val >= num_groups:
        num_train = num_groups - 2
        num_val = 1

    train_groups = permutation[:num_train]
    val_groups = permutation[
        num_train:num_train + num_val
    ]
    test_groups = permutation[
        num_train + num_val:
    ]

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


class FreshTCNStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.policy = VisionStudentTeacher(
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
        )

        self.state_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 12),
        )

    def forward(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.policy.encode_temporal_state(
            observations
        )

        return self.state_head(latent)


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

    r2 = 1.0 - residual_sum / total_sum.clamp_min(
        1.0e-12
    )

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

    for index, name in enumerate(TARGET_NAMES):
        print(
            f"{name:14s}"
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

    for name, section in groups.items():
        print(
            f"{name:20s} "
            f"RMSE={metrics['rmse'][section].mean().item():.6f} "
            f"R2={metrics['r2'][section].mean().item():.6f}"
        )


@torch.no_grad()
def predict_in_batches(
    model: nn.Module,
    observations: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> torch.Tensor:
    model.eval()
    predictions = []

    for start in range(
        0,
        observations.shape[0],
        batch_size,
    ):
        batch = observations[
            start:start + batch_size
        ].to(device)

        prediction_normalized = model(batch).cpu()

        prediction = (
            prediction_normalized * target_std
            + target_mean
        )

        predictions.append(prediction)

    return torch.cat(predictions, dim=0)


def make_temporal_ablations(
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

    generator = torch.Generator().manual_seed(
        SEED + 99
    )

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


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("[INFO] Device:", device)
    print("[INFO] Data:", args.data)
    print("[INFO] Output:", args.output)

    data = torch.load(
        args.data,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "observations",
        "teacher_observations",
        "environment_ids",
        "episode_ids",
        "episode_steps",
        "curriculum_levels",
    }

    missing = sorted(required - set(data.keys()))

    if missing:
        raise KeyError(
            f"Dataset is missing keys: {missing}"
        )

    observations = data["observations"].float()
    teacher_observations = (
        data["teacher_observations"].float()
    )
    environment_ids = data["environment_ids"].long()
    episode_ids = data["episode_ids"].long()
    episode_steps = data["episode_steps"].long()
    curriculum_levels = (
        data["curriculum_levels"].float()
    )

    mature = episode_steps >= 31

    observations = observations[mature]
    teacher_observations = teacher_observations[mature]
    environment_ids = environment_ids[mature]
    episode_ids = episode_ids[mature]
    curriculum_levels = curriculum_levels[mature]

    # Match the frozen probe exactly:
    # position, linear velocity, angular velocity, gravity.
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

    target_mean = targets[train_mask].mean(dim=0)
    target_std = targets[train_mask].std(
        dim=0
    ).clamp_min(1.0e-6)

    normalized_targets = (
        targets - target_mean
    ) / target_std

    train_dataset = TensorDataset(
        observations[train_mask],
        normalized_targets[train_mask],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_observations = observations[val_mask].to(
        device
    )
    val_targets = normalized_targets[val_mask].to(
        device
    )

    model = FreshTCNStateModel().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1.0e-6,
    )

    best_val_loss = math.inf
    best_state = None
    stale_epochs = 0
    patience = 20

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_rows = 0

        for obs_batch, target_batch in train_loader:
            obs_batch = obs_batch.to(
                device,
                non_blocking=True,
            )
            target_batch = target_batch.to(
                device,
                non_blocking=True,
            )

            prediction = model(obs_batch)

            loss = F.smooth_l1_loss(
                prediction,
                target_batch,
                beta=0.5,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            rows = obs_batch.shape[0]
            total_loss += float(loss.item()) * rows
            total_rows += rows

        model.eval()

        with torch.no_grad():
            val_prediction = model(
                val_observations
            )

            val_loss = F.smooth_l1_loss(
                val_prediction,
                val_targets,
                beta=0.5,
            ).item()

        train_loss = total_loss / max(total_rows, 1)

        scheduler.step(val_loss)

        learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"[TRAIN] "
            f"epoch={epoch:03d} "
            f"train={train_loss:.7f} "
            f"val={val_loss:.7f} "
            f"lr={learning_rate:.2e}"
        )

        if val_loss < best_val_loss - 1.0e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(
                f"[TRAIN] Early stopping at epoch {epoch}; "
                f"best_val={best_val_loss:.7f}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(best_state)
    model.eval()

    test_observations = observations[test_mask]
    test_targets = targets[test_mask]
    test_levels = curriculum_levels[test_mask]

    pooled_prediction = predict_in_batches(
        model,
        test_observations,
        target_mean,
        target_std,
        device,
    )

    print_metrics(
        "FRESH STATE-SUPERVISED TCN — POOLED TEST",
        pooled_prediction,
        test_targets,
    )

    print()
    print("=" * 100)
    print("FRESH STATE-SUPERVISED TCN — PER-LEVEL TEST")
    print("=" * 100)

    for level in (
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
    ):
        level_mask = torch.isclose(
            test_levels,
            torch.tensor(level),
            atol=1.0e-6,
        )

        rows = int(level_mask.sum())

        if rows == 0:
            print(
                f"[WARNING] Level {level:.2f}: no test rows"
            )
            continue

        prediction = pooled_prediction[level_mask]
        target = test_targets[level_mask]

        metrics = calculate_metrics(
            prediction,
            target,
        )

        print(
            f"level={level:.2f} "
            f"rows={rows:5d} "
            f"velocity_RMSE="
            f"{metrics['rmse'][3:6].mean().item():.6f} "
            f"velocity_R2="
            f"{metrics['r2'][3:6].mean().item():.6f} "
            f"omega_RMSE="
            f"{metrics['rmse'][6:9].mean().item():.6f} "
            f"omega_R2="
            f"{metrics['r2'][6:9].mean().item():.6f}"
        )

    print()
    print("=" * 100)
    print("FRESH STATE-SUPERVISED TCN — TEMPORAL ABLATIONS")
    print("=" * 100)

    print(
        f"{'input':20s}"
        f"{'vel_RMSE':>14s}"
        f"{'vel_R2':>14s}"
        f"{'omega_RMSE':>14s}"
        f"{'omega_R2':>14s}"
    )

    for name, ablated_observations in (
        make_temporal_ablations(
            test_observations
        ).items()
    ):
        prediction = predict_in_batches(
            model,
            ablated_observations,
            target_mean,
            target_std,
            device,
        )

        metrics = calculate_metrics(
            prediction,
            test_targets,
        )

        print(
            f"{name:20s}"
            f"{metrics['rmse'][3:6].mean().item():14.6f}"
            f"{metrics['r2'][3:6].mean().item():14.6f}"
            f"{metrics['rmse'][6:9].mean().item():14.6f}"
            f"{metrics['r2'][6:9].mean().item():14.6f}"
        )

    output_path = args.output / "fresh_state_tcn.pt"

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "target_names": TARGET_NAMES,
            "seed": SEED,
            "data_path": str(args.data),
            "best_val_loss": best_val_loss,
            "architecture": {
                "history_length": 32,
                "token_dim": 8,
                "temporal_channels": 128,
                "state_head": [128, 256, 256, 12],
            },
        },
        output_path,
    )

    print()
    print("[DONE] Saved:", output_path)


if __name__ == "__main__":
    main()
