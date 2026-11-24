from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_DATA = Path(
    "dagger_data/tcn_state_probe_v1/"
    "pooled_student_epoch70_all_levels.pt"
)

DEFAULT_CHECKPOINT = Path(
    "eval_logs/corners_tcn_student_v1/"
    "fresh_state_supervised_tcn/"
    "fresh_state_tcn.pt"
)

DEFAULT_OUTPUT = Path(
    "eval_logs/corners_tcn_student_v1/"
    "fresh_state_supervised_tcn/"
    "temporal_lag_results.pt"
)

MODEL_FILE = Path(
    "envs/aruco_hover/agents/"
    "vision_student_teacher_probe_ready.py"
).resolve()


def load_vision_model_class():
    spec = importlib.util.spec_from_file_location(
        "vision_student_teacher_probe_ready_offline",
        MODEL_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {MODEL_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.VisionStudentTeacher


VisionStudentTeacher = load_vision_model_class()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure temporal lag in fresh TCN state predictions."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max_offset",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
    )

    return parser.parse_args()


@torch.no_grad()
def predict_in_batches(
    model: nn.Module,
    observations: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
    batch_size: int,
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

        normalized_prediction = model(batch).cpu()

        prediction = (
            normalized_prediction * target_std
            + target_mean
        )

        predictions.append(prediction)

    return torch.cat(predictions, dim=0)


def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    error = prediction - target

    rmse = torch.sqrt(
        error.square().mean(dim=0)
    )

    target_mean = target.mean(
        dim=0,
        keepdim=True,
    )

    residual_sum = error.square().sum(dim=0)

    total_sum = (
        target - target_mean
    ).square().sum(dim=0)

    r2 = 1.0 - (
        residual_sum
        / total_sum.clamp_min(1.0e-12)
    )

    return {
        "rmse": rmse,
        "r2": r2,
    }


def build_same_sample_lag_pairs(
    environment_ids: torch.Tensor,
    episode_ids: torch.Tensor,
    episode_steps: torch.Tensor,
    offsets: list[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Select prediction rows for which every requested target offset exists.

    Therefore every offset is evaluated using exactly the same prediction
    rows, and no comparison crosses an episode boundary.
    """

    groups: dict[
        tuple[int, int],
        list[int],
    ] = defaultdict(list)

    for row in range(len(environment_ids)):
        key = (
            int(environment_ids[row]),
            int(episode_ids[row]),
        )

        groups[key].append(row)

    prediction_indices: list[int] = []

    target_indices: dict[int, list[int]] = {
        offset: []
        for offset in offsets
    }

    duplicate_step_groups = 0

    for group_rows in groups.values():
        sorted_rows = sorted(
            group_rows,
            key=lambda index: int(
                episode_steps[index]
            ),
        )

        step_to_row: dict[int, int] = {}

        for row in sorted_rows:
            step = int(episode_steps[row])

            if step in step_to_row:
                duplicate_step_groups += 1
                continue

            step_to_row[step] = row

        for step, prediction_row in step_to_row.items():
            if not all(
                step + offset in step_to_row
                for offset in offsets
            ):
                continue

            prediction_indices.append(
                prediction_row
            )

            for offset in offsets:
                target_indices[offset].append(
                    step_to_row[step + offset]
                )

    if duplicate_step_groups:
        print(
            "[WARNING] Duplicate episode-step rows skipped:",
            duplicate_step_groups,
        )

    if not prediction_indices:
        raise RuntimeError(
            "No rows have the full requested temporal neighborhood."
        )

    prediction_tensor = torch.tensor(
        prediction_indices,
        dtype=torch.long,
    )

    target_tensors = {
        offset: torch.tensor(
            indices,
            dtype=torch.long,
        )
        for offset, indices in target_indices.items()
    }

    expected_rows = len(prediction_indices)

    for offset, indices in target_tensors.items():
        if len(indices) != expected_rows:
            raise RuntimeError(
                f"Offset {offset} has inconsistent row count."
            )

    return prediction_tensor, target_tensors


def grouped_values(
    metrics: dict[str, torch.Tensor],
) -> dict[str, float]:
    return {
        "velocity_rmse":
            metrics["rmse"][3:6].mean().item(),

        "velocity_r2":
            metrics["r2"][3:6].mean().item(),

        "omega_rmse":
            metrics["rmse"][6:9].mean().item(),

        "omega_r2":
            metrics["r2"][6:9].mean().item(),
    }


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(args.data)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    args.output.parent.mkdir(
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
    print("[INFO] Checkpoint:", args.checkpoint)
    print("[INFO] Maximum offset:", args.max_offset)

    data = torch.load(
        args.data,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
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

    # Use the same mature-history criterion as previous experiments.
    mature = episode_steps >= 31

    observations = observations[mature]
    teacher_observations = teacher_observations[mature]
    environment_ids = environment_ids[mature]
    episode_ids = episode_ids[mature]
    episode_steps = episode_steps[mature]
    curriculum_levels = curriculum_levels[mature]

    targets = teacher_observations[:, 0:12]

    print(
        "[INFO] Mature rows retained:",
        len(observations),
    )

    model = FreshTCNStateModel().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    target_mean = checkpoint[
        "target_mean"
    ].float()

    target_std = checkpoint[
        "target_std"
    ].float()

    predictions = predict_in_batches(
        model=model,
        observations=observations,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        batch_size=args.batch_size,
    )

    print(
        "[INFO] Prediction shape:",
        tuple(predictions.shape),
    )

    offsets = list(
        range(
            -args.max_offset,
            args.max_offset + 1,
        )
    )

    (
        prediction_indices,
        target_indices,
    ) = build_same_sample_lag_pairs(
        environment_ids=environment_ids,
        episode_ids=episode_ids,
        episode_steps=episode_steps,
        offsets=offsets,
    )

    selected_predictions = predictions[
        prediction_indices
    ]

    selected_levels = curriculum_levels[
        prediction_indices
    ]

    print(
        "[INFO] Same-sample lag rows:",
        len(prediction_indices),
    )

    print()
    print("=" * 104)
    print("FRESH STATE-TCN TEMPORAL-LAG SWEEP")
    print("=" * 104)

    print(
        "Definition: prediction at t is compared with truth at "
        "t + target_offset."
    )

    print(
        "A best negative offset means the prediction resembles "
        "an earlier state and therefore lags."
    )

    print(
        "A best positive offset means the prediction resembles "
        "a future state."
    )

    print()

    print(
        f"{'target_offset':>14s}"
        f"{'rows':>10s}"
        f"{'vel_RMSE':>14s}"
        f"{'vel_R2':>14s}"
        f"{'omega_RMSE':>14s}"
        f"{'omega_R2':>14s}"
    )

    results: dict[int, dict[str, object]] = {}

    for offset in offsets:
        shifted_targets = targets[
            target_indices[offset]
        ]

        metrics = calculate_metrics(
            selected_predictions,
            shifted_targets,
        )

        grouped = grouped_values(metrics)

        results[offset] = {
            **grouped,
            "rmse_per_axis": metrics["rmse"],
            "r2_per_axis": metrics["r2"],
        }

        print(
            f"{offset:14d}"
            f"{len(prediction_indices):10d}"
            f"{grouped['velocity_rmse']:14.6f}"
            f"{grouped['velocity_r2']:14.6f}"
            f"{grouped['omega_rmse']:14.6f}"
            f"{grouped['omega_r2']:14.6f}"
        )

    best_velocity_offset = max(
        offsets,
        key=lambda offset:
            float(results[offset]["velocity_r2"]),
    )

    best_omega_offset = max(
        offsets,
        key=lambda offset:
            float(results[offset]["omega_r2"]),
    )

    print()
    print("=" * 104)
    print("BEST OFFSETS")
    print("=" * 104)

    print(
        "Best linear-velocity target offset:",
        best_velocity_offset,
        "steps",
        f"R2={results[best_velocity_offset]['velocity_r2']:.6f}",
        f"RMSE={results[best_velocity_offset]['velocity_rmse']:.6f}",
    )

    print(
        "Best angular-velocity target offset:",
        best_omega_offset,
        "steps",
        f"R2={results[best_omega_offset]['omega_r2']:.6f}",
        f"RMSE={results[best_omega_offset]['omega_rmse']:.6f}",
    )

    print()
    print("Angular-velocity per-axis R2 at offset 0:")

    zero_r2 = results[0]["r2_per_axis"]

    print(
        f"omega_x={zero_r2[6].item():.6f} "
        f"omega_y={zero_r2[7].item():.6f} "
        f"omega_z={zero_r2[8].item():.6f}"
    )

    print(
        f"Angular-velocity per-axis R2 at best offset "
        f"{best_omega_offset}:"
    )

    best_r2 = results[
        best_omega_offset
    ]["r2_per_axis"]

    print(
        f"omega_x={best_r2[6].item():.6f} "
        f"omega_y={best_r2[7].item():.6f} "
        f"omega_z={best_r2[8].item():.6f}"
    )

    print()
    print("=" * 104)
    print("PER-LEVEL ANGULAR-LAG SUMMARY")
    print("=" * 104)

    for level in (
        0.00,
        0.25,
        0.50,
        0.75,
        1.00,
    ):
        level_mask = torch.isclose(
            selected_levels,
            torch.tensor(level),
            atol=1.0e-6,
        )

        rows = int(level_mask.sum())

        if rows == 0:
            print(
                f"level={level:.2f} rows=0"
            )
            continue

        level_predictions = selected_predictions[
            level_mask
        ]

        level_results = {}

        for offset in offsets:
            shifted_targets = targets[
                target_indices[offset]
            ][level_mask]

            metrics = calculate_metrics(
                level_predictions,
                shifted_targets,
            )

            level_results[offset] = grouped_values(
                metrics
            )

        level_best = max(
            offsets,
            key=lambda offset:
                level_results[offset]["omega_r2"],
        )

        print(
            f"level={level:.2f} "
            f"rows={rows:5d} "
            f"offset0_omega_R2="
            f"{level_results[0]['omega_r2']:.6f} "
            f"best_offset={level_best:+d} "
            f"best_omega_R2="
            f"{level_results[level_best]['omega_r2']:.6f}"
        )

    torch.save(
        {
            "offsets": offsets,
            "results": results,
            "best_velocity_offset":
                best_velocity_offset,
            "best_omega_offset":
                best_omega_offset,
            "prediction_rows":
                prediction_indices,
            "data_path":
                str(args.data),
            "checkpoint_path":
                str(args.checkpoint),
            "offset_definition": (
                "prediction_at_t_vs_truth_at_t_plus_offset"
            ),
        },
        args.output,
    )

    print()
    print("[DONE] Saved:", args.output)


if __name__ == "__main__":
    main()
