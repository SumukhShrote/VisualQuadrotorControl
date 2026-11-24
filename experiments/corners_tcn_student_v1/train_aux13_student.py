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


SEED = 7500

DEFAULT_DATA = Path(
    "dagger_data/tcn_state_probe_v1/"
    "pooled_student_epoch70_all_levels.pt"
)

DEFAULT_CHECKPOINT = Path(
    "checkpoints/corners_tcn_student_v1/"
    "stage3_dagger2m/"
    "stage3_dagger2m_e70.pt"
)

DEFAULT_OUTPUT = Path(
    "checkpoints/corners_tcn_student_v1/"
    "stage4_aux13_ft/"
    "stage4_aux13_ft_best.pt"
)

MODEL_FILE = Path(
    "envs/aruco_hover/agents/"
    "vision_student_teacher_probe_ready.py"
).resolve()

# Teacher observation:
# 0:3   position error
# 3:6   linear velocity
# 6:9   angular velocity          -- deliberately excluded
# 9:12  projected gravity
# 12:15 marker direction
# 15:19 previous action           -- deliberately excluded
# 19    marker distance
AUX_TARGET_INDICES = [
    0, 1, 2,
    3, 4, 5,
    9, 10, 11,
    12, 13, 14,
    19,
]

ACTION_NAMES = [
    "collective_thrust",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
]


def load_policy_class():
    spec = importlib.util.spec_from_file_location(
        "vision_student_teacher_probe_ready_offline",
        MODEL_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model file: {MODEL_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.VisionStudentTeacher


VisionStudentTeacher = load_policy_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the epoch-70 corners TCN using teacher-action "
            "imitation plus a training-only 13-D auxiliary state loss."
        )
    )

    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--tcn_lr", type=float, default=3.0e-5)
    parser.add_argument("--action_lr", type=float, default=1.0e-4)
    parser.add_argument("--aux_lr", type=float, default=3.0e-4)

    parser.add_argument(
        "--aux_weight",
        type=float,
        default=0.10,
    )

    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_episode_split(
    environment_ids: torch.Tensor,
    episode_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = torch.stack(
        [
            environment_ids.long(),
            episode_ids.long(),
        ],
        dim=1,
    )

    unique_groups, inverse = torch.unique(
        groups,
        dim=0,
        return_inverse=True,
    )

    num_groups = unique_groups.shape[0]

    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(
        num_groups,
        generator=generator,
    )

    num_train = int(round(0.70 * num_groups))
    num_val = int(round(0.15 * num_groups))

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


class AuxiliaryStateHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, len(AUX_TARGET_INDICES)),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


def build_policy(device: torch.device) -> VisionStudentTeacher:
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

    return policy


@torch.no_grad()
def predict_actions(
    policy: VisionStudentTeacher,
    observations: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    policy.eval()
    outputs = []

    for start in range(0, len(observations), batch_size):
        batch = observations[
            start:start + batch_size
        ].to(device)

        outputs.append(
            policy.act_inference(batch).cpu()
        )

    return torch.cat(outputs, dim=0)


def action_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    error = prediction - target

    rmse = torch.sqrt(
        error.square().mean(dim=0)
    )

    mae = error.abs().mean(dim=0)

    centered = target - target.mean(
        dim=0,
        keepdim=True,
    )

    r2 = 1.0 - (
        error.square().sum(dim=0)
        / centered.square().sum(dim=0).clamp_min(1.0e-12)
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def print_action_metrics(
    title: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics = action_metrics(prediction, target)

    print()
    print("=" * 92)
    print(title)
    print("=" * 92)

    print(
        f"{'action':20s}"
        f"{'RMSE':>12s}"
        f"{'MAE':>12s}"
        f"{'R2':>12s}"
    )

    for index, name in enumerate(ACTION_NAMES):
        print(
            f"{name:20s}"
            f"{metrics['rmse'][index].item():12.6f}"
            f"{metrics['mae'][index].item():12.6f}"
            f"{metrics['r2'][index].item():12.6f}"
        )

    print(
        f"{'MEAN':20s}"
        f"{metrics['rmse'].mean().item():12.6f}"
        f"{metrics['mae'].mean().item():12.6f}"
        f"{metrics['r2'].mean().item():12.6f}"
    )

    return metrics


@torch.no_grad()
def validation_losses(
    policy: VisionStudentTeacher,
    aux_head: AuxiliaryStateHead,
    observations: torch.Tensor,
    teacher_actions: torch.Tensor,
    normalized_aux_targets: torch.Tensor,
    mature: torch.Tensor,
    action_scale: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[float, float]:
    policy.eval()
    aux_head.eval()

    action_sum = 0.0
    action_rows = 0

    aux_sum = 0.0
    aux_rows = 0

    for start in range(0, len(observations), batch_size):
        obs_batch = observations[
            start:start + batch_size
        ].to(device)

        action_batch = teacher_actions[
            start:start + batch_size
        ].to(device)

        aux_batch = normalized_aux_targets[
            start:start + batch_size
        ].to(device)

        mature_batch = mature[
            start:start + batch_size
        ].to(device)

        latent = policy.encode_temporal_state(obs_batch)

        action_prediction = torch.tanh(
            policy.vision_student(latent)
        )

        auxiliary_prediction = aux_head(latent)

        action_loss = (
            (
                (action_prediction - action_batch)
                / action_scale
            ).square().mean()
        )

        rows = len(obs_batch)
        action_sum += float(action_loss.item()) * rows
        action_rows += rows

        if mature_batch.any():
            auxiliary_loss = F.smooth_l1_loss(
                auxiliary_prediction[mature_batch],
                aux_batch[mature_batch],
                beta=0.5,
            )

            mature_rows = int(mature_batch.sum())
            aux_sum += (
                float(auxiliary_loss.item())
                * mature_rows
            )
            aux_rows += mature_rows

    return (
        action_sum / max(action_rows, 1),
        aux_sum / max(aux_rows, 1),
    )


def main() -> None:
    args = parse_args()
    seed_everything(SEED)

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
    print("[INFO] Initial checkpoint:", args.checkpoint)
    print("[INFO] Output checkpoint:", args.output)
    print("[INFO] Auxiliary weight:", args.aux_weight)
    print("[INFO] Auxiliary indices:", AUX_TARGET_INDICES)

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

    required = {
        "observations",
        "teacher_observations",
        "teacher_actions",
        "environment_ids",
        "episode_ids",
        "episode_steps",
        "curriculum_levels",
    }

    missing = sorted(required - set(data.keys()))

    if missing:
        raise KeyError(f"Dataset is missing keys: {missing}")

    observations = data["observations"].float()
    teacher_observations = (
        data["teacher_observations"].float()
    )
    teacher_actions = data["teacher_actions"].float()

    environment_ids = data["environment_ids"].long()
    episode_ids = data["episode_ids"].long()
    episode_steps = data["episode_steps"].long()
    curriculum_levels = data["curriculum_levels"].float()

    mature = episode_steps >= 31

    auxiliary_targets = teacher_observations[
        :,
        AUX_TARGET_INDICES,
    ]

    train_mask, val_mask, test_mask = make_episode_split(
        environment_ids,
        episode_ids,
    )

    mature_train = train_mask & mature

    auxiliary_mean = auxiliary_targets[
        mature_train
    ].mean(dim=0)

    auxiliary_std = auxiliary_targets[
        mature_train
    ].std(dim=0).clamp_min(1.0e-6)

    normalized_auxiliary_targets = (
        auxiliary_targets - auxiliary_mean
    ) / auxiliary_std

    action_scale = teacher_actions[
        train_mask
    ].std(dim=0).clamp_min(0.05).to(device)

    policy = build_policy(device)

    if "student_state_dict" not in checkpoint:
        raise KeyError(
            "Initial checkpoint is missing student_state_dict."
        )

    policy.student.load_state_dict(
        checkpoint["student_state_dict"],
        strict=True,
    )

    print("[PASS] Strictly loaded epoch-70 student.")

    # Teacher parameters and action-noise parameters are not optimized.
    for parameter in policy.teacher.parameters():
        parameter.requires_grad_(False)

    auxiliary_head = AuxiliaryStateHead().to(device)

    baseline_prediction = predict_actions(
        policy,
        observations[test_mask],
        device,
    )

    baseline_metrics = print_action_metrics(
        "BASELINE EPOCH-70 — HELD-OUT TEACHER ACTIONS",
        baseline_prediction,
        teacher_actions[test_mask],
    )

    train_dataset = TensorDataset(
        observations[train_mask],
        teacher_actions[train_mask],
        normalized_auxiliary_targets[train_mask],
        mature[train_mask],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": policy.tcn.parameters(),
                "lr": args.tcn_lr,
            },
            {
                "params": policy.vision_student.parameters(),
                "lr": args.action_lr,
            },
            {
                "params": auxiliary_head.parameters(),
                "lr": args.aux_lr,
            },
        ],
        weight_decay=1.0e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1.0e-6,
    )

    best_val_action = math.inf
    best_student_state = None
    best_aux_state = None
    best_epoch = 0
    stale_epochs = 0
    patience = 18

    val_observations = observations[val_mask]
    val_actions = teacher_actions[val_mask]
    val_auxiliary = normalized_auxiliary_targets[val_mask]
    val_mature = mature[val_mask]

    for epoch in range(1, args.epochs + 1):
        policy.train()
        auxiliary_head.train()

        total_action_loss = 0.0
        total_auxiliary_loss = 0.0
        total_rows = 0
        total_mature_rows = 0

        for (
            observation_batch,
            action_batch,
            auxiliary_batch,
            mature_batch,
        ) in train_loader:
            observation_batch = observation_batch.to(
                device,
                non_blocking=True,
            )
            action_batch = action_batch.to(
                device,
                non_blocking=True,
            )
            auxiliary_batch = auxiliary_batch.to(
                device,
                non_blocking=True,
            )
            mature_batch = mature_batch.to(
                device,
                non_blocking=True,
            )

            latent = policy.encode_temporal_state(
                observation_batch
            )

            action_prediction = torch.tanh(
                policy.vision_student(latent)
            )

            auxiliary_prediction = auxiliary_head(latent)

            action_loss = (
                (
                    (action_prediction - action_batch)
                    / action_scale
                ).square().mean()
            )

            if mature_batch.any():
                auxiliary_loss = F.smooth_l1_loss(
                    auxiliary_prediction[mature_batch],
                    auxiliary_batch[mature_batch],
                    beta=0.5,
                )
            else:
                auxiliary_loss = torch.zeros(
                    (),
                    device=device,
                )

            loss = (
                action_loss
                + args.aux_weight * auxiliary_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(policy.tcn.parameters())
                + list(policy.vision_student.parameters())
                + list(auxiliary_head.parameters()),
                max_norm=1.0,
            )

            optimizer.step()

            rows = len(observation_batch)
            mature_rows = int(mature_batch.sum())

            total_action_loss += (
                float(action_loss.item()) * rows
            )
            total_auxiliary_loss += (
                float(auxiliary_loss.item())
                * mature_rows
            )

            total_rows += rows
            total_mature_rows += mature_rows

        (
            val_action_loss,
            val_auxiliary_loss,
        ) = validation_losses(
            policy=policy,
            aux_head=auxiliary_head,
            observations=val_observations,
            teacher_actions=val_actions,
            normalized_aux_targets=val_auxiliary,
            mature=val_mature,
            action_scale=action_scale,
            device=device,
        )

        scheduler.step(val_action_loss)

        train_action_loss = (
            total_action_loss / max(total_rows, 1)
        )

        train_auxiliary_loss = (
            total_auxiliary_loss
            / max(total_mature_rows, 1)
        )

        print(
            f"[TRAIN] "
            f"epoch={epoch:03d} "
            f"action={train_action_loss:.7f} "
            f"aux={train_auxiliary_loss:.7f} "
            f"val_action={val_action_loss:.7f} "
            f"val_aux={val_auxiliary_loss:.7f} "
            f"tcn_lr={optimizer.param_groups[0]['lr']:.2e} "
            f"action_lr={optimizer.param_groups[1]['lr']:.2e}"
        )

        if val_action_loss < best_val_action - 1.0e-6:
            best_val_action = val_action_loss
            best_student_state = copy.deepcopy(
                policy.student.state_dict()
            )
            best_aux_state = copy.deepcopy(
                auxiliary_head.state_dict()
            )
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(
                f"[TRAIN] Early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}, "
                f"best_val_action={best_val_action:.7f}"
            )
            break

    if best_student_state is None:
        raise RuntimeError("No valid checkpoint was produced.")

    policy.student.load_state_dict(
        best_student_state,
        strict=True,
    )

    auxiliary_head.load_state_dict(
        best_aux_state,
        strict=True,
    )

    final_prediction = predict_actions(
        policy,
        observations[test_mask],
        device,
    )

    final_metrics = print_action_metrics(
        "STAGE-4 AUX13 STUDENT — HELD-OUT TEACHER ACTIONS",
        final_prediction,
        teacher_actions[test_mask],
    )

    print()
    print("=" * 92)
    print("PER-LEVEL HELD-OUT ACTION RESULTS")
    print("=" * 92)

    test_levels = curriculum_levels[test_mask]
    test_targets = teacher_actions[test_mask]

    for level in (0.00, 0.25, 0.50, 0.75, 1.00):
        level_mask = torch.isclose(
            test_levels,
            torch.tensor(level),
            atol=1.0e-6,
        )

        rows = int(level_mask.sum())

        if rows == 0:
            print(f"level={level:.2f} rows=0")
            continue

        metrics = action_metrics(
            final_prediction[level_mask],
            test_targets[level_mask],
        )

        print(
            f"level={level:.2f} "
            f"rows={rows:5d} "
            f"mean_RMSE={metrics['rmse'].mean().item():.6f} "
            f"mean_R2={metrics['r2'].mean().item():.6f}"
        )

    output = {
        "student_state_dict": {
            key: value.detach().cpu()
            for key, value in best_student_state.items()
        },
        "auxiliary_state_head_dict": {
            key: value.detach().cpu()
            for key, value in best_aux_state.items()
        },
        "history_length": 32,
        "token_dim": 8,
        "temporal_channels": 128,
        "source_checkpoint": str(args.checkpoint),
        "data_path": str(args.data),
        "best_epoch": best_epoch,
        "best_val_action_loss": best_val_action,
        "auxiliary_weight": args.aux_weight,
        "auxiliary_target_indices": AUX_TARGET_INDICES,
        "auxiliary_target_mean": auxiliary_mean,
        "auxiliary_target_std": auxiliary_std,
        "training_semantics": (
            "epoch70 initialization; teacher-action imitation on all rows; "
            "training-only normalized 13D state auxiliary loss on mature rows"
        ),
        "baseline_test_mean_rmse":
            baseline_metrics["rmse"].mean().item(),
        "baseline_test_mean_r2":
            baseline_metrics["r2"].mean().item(),
        "final_test_mean_rmse":
            final_metrics["rmse"].mean().item(),
        "final_test_mean_r2":
            final_metrics["r2"].mean().item(),
        "seed": SEED,
    }

    torch.save(output, args.output)

    print()
    print("[DONE] Saved deployable student checkpoint:", args.output)
    print(
        "[RESULT]",
        f"baseline_R2={baseline_metrics['r2'].mean().item():.6f}",
        f"final_R2={final_metrics['r2'].mean().item():.6f}",
        f"baseline_RMSE={baseline_metrics['rmse'].mean().item():.6f}",
        f"final_RMSE={final_metrics['rmse'].mean().item():.6f}",
    )


if __name__ == "__main__":
    main()
