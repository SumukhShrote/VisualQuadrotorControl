from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


SEED = 7400

DATA_PATH = Path(
    "dagger_data/tcn_state_probe_v1/"
    "pooled_student_epoch70_all_levels.pt"
)

OUTPUT_PATH = Path(
    "eval_logs/corners_tcn_student_v1/"
    "teacher_action_dependency_results.pt"
)

ACTION_NAMES = [
    "collective_thrust",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_episode_split(
    environment_ids: torch.Tensor,
    episode_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = torch.stack(
        [environment_ids.long(), episode_ids.long()],
        dim=1,
    )

    _, inverse = torch.unique(
        groups,
        dim=0,
        return_inverse=True,
    )

    num_groups = int(inverse.max()) + 1

    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(
        num_groups,
        generator=generator,
    )

    num_train = int(round(0.70 * num_groups))
    num_val = int(round(0.15 * num_groups))

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
        f"train_rows={int(train_mask.sum())}",
        f"val_rows={int(val_mask.sum())}",
        f"test_rows={int(test_mask.sum())}",
    )

    return train_mask, val_mask, test_mask


class ActionPredictor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def calculate_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    error = prediction - target

    rmse = torch.sqrt(
        error.square().mean(dim=0)
    )

    mae = error.abs().mean(dim=0)

    target_centered = (
        target - target.mean(dim=0, keepdim=True)
    )

    r2 = 1.0 - (
        error.square().sum(dim=0)
        / target_centered.square().sum(dim=0).clamp_min(1.0e-12)
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def train_predictor(
    name: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    feature_mean = features[train_mask].mean(dim=0)
    feature_std = features[train_mask].std(
        dim=0
    ).clamp_min(1.0e-6)

    normalized_features = (
        features - feature_mean
    ) / feature_std

    target_mean = targets[train_mask].mean(dim=0)
    target_std = targets[train_mask].std(
        dim=0
    ).clamp_min(1.0e-6)

    normalized_targets = (
        targets - target_mean
    ) / target_std

    dataset = TensorDataset(
        normalized_features[train_mask],
        normalized_targets[train_mask],
    )

    loader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = ActionPredictor(
        input_dim=features.shape[1]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=1.0e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1.0e-6,
    )

    val_features = normalized_features[val_mask].to(device)
    val_targets = normalized_targets[val_mask].to(device)

    best_loss = math.inf
    best_state = None
    stale_epochs = 0

    for epoch in range(1, 101):
        model.train()

        total_loss = 0.0
        total_rows = 0

        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)

            prediction = model(feature_batch)

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

            rows = feature_batch.shape[0]
            total_loss += float(loss.item()) * rows
            total_rows += rows

        model.eval()

        with torch.no_grad():
            val_prediction = model(val_features)

            val_loss = F.smooth_l1_loss(
                val_prediction,
                val_targets,
                beta=0.5,
            ).item()

        scheduler.step(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{name}] "
                f"epoch={epoch:03d} "
                f"train={total_loss / total_rows:.7f} "
                f"val={val_loss:.7f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_loss < best_loss - 1.0e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(
                model.state_dict()
            )
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= 15:
            print(
                f"[{name}] Early stopping; "
                f"best_val={best_loss:.7f}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"{name}: no valid checkpoint produced"
        )

    model.load_state_dict(best_state)
    model.eval()

    return (
        model,
        feature_mean,
        feature_std,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    outputs = []

    normalized = (
        features - feature_mean
    ) / feature_std

    for start in range(0, len(features), 4096):
        batch = normalized[
            start:start + 4096
        ].to(device)

        output = model(batch).cpu()
        outputs.append(output)

    return (
        torch.cat(outputs, dim=0)
        * target_std
        + target_mean
    )


def print_metrics(
    name: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    metrics = calculate_metrics(
        prediction,
        target,
    )

    print()
    print("=" * 94)
    print(name)
    print("=" * 94)

    print(
        f"{'action':20s}"
        f"{'RMSE':>12s}"
        f"{'MAE':>12s}"
        f"{'R2':>12s}"
    )

    for index, action_name in enumerate(ACTION_NAMES):
        print(
            f"{action_name:20s}"
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


def main() -> None:
    seed_everything(SEED)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("[INFO] Device:", device)
    print("[INFO] Data:", DATA_PATH)

    data = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "teacher_observations",
        "teacher_actions",
        "student_actions",
        "environment_ids",
        "episode_ids",
        "episode_steps",
    }

    missing = sorted(required - set(data.keys()))

    if missing:
        raise KeyError(
            f"Missing dataset keys: {missing}"
        )

    teacher_obs = data[
        "teacher_observations"
    ].float()

    teacher_actions = data[
        "teacher_actions"
    ].float()

    student_actions = data[
        "student_actions"
    ].float()

    environment_ids = data[
        "environment_ids"
    ].long()

    episode_ids = data[
        "episode_ids"
    ].long()

    episode_steps = data[
        "episode_steps"
    ].long()

    mature = episode_steps >= 31

    teacher_obs = teacher_obs[mature]
    teacher_actions = teacher_actions[mature]
    student_actions = student_actions[mature]
    environment_ids = environment_ids[mature]
    episode_ids = episode_ids[mature]

    print("[INFO] Mature rows:", len(teacher_obs))

    train_mask, val_mask, test_mask = make_episode_split(
        environment_ids,
        episode_ids,
    )

    # Teacher observation:
    # 0:3   position error
    # 3:6   linear velocity
    # 6:9   angular velocity
    # 9:12  projected gravity
    # 12:15 marker direction
    # 15:19 previous action
    # 19    distance

    feature_sets = {
        "FULL_20D": list(range(20)),

        "NO_PREVIOUS_ACTION_16D":
            list(range(15)) + [19],

        "NO_ANGULAR_VELOCITY_17D":
            list(range(6)) + list(range(9, 20)),

        "NO_OMEGA_NO_PREVIOUS_ACTION_13D":
            list(range(6))
            + list(range(9, 15))
            + [19],

        "PREVIOUS_ACTION_ONLY_4D":
            list(range(15, 19)),
    }

    target_mean = teacher_actions[train_mask].mean(dim=0)
    target_std = teacher_actions[train_mask].std(
        dim=0
    ).clamp_min(1.0e-6)

    results = {}

    # Existing epoch-70 student baseline.
    results["EPOCH70_CORNERS_STUDENT"] = print_metrics(
        "EPOCH-70 CORNERS-TCN STUDENT",
        student_actions[test_mask],
        teacher_actions[test_mask],
    )

    for name, indices in feature_sets.items():
        features = teacher_obs[:, indices]

        print()
        print(
            f"[TRAIN] {name}: "
            f"input_dim={features.shape[1]}"
        )

        (
            model,
            feature_mean,
            feature_std,
        ) = train_predictor(
            name=name,
            features=features,
            targets=teacher_actions,
            train_mask=train_mask,
            val_mask=val_mask,
            device=device,
        )

        prediction = predict(
            model=model,
            features=features[test_mask],
            feature_mean=feature_mean,
            feature_std=feature_std,
            target_mean=target_mean,
            target_std=target_std,
            device=device,
        )

        results[name] = print_metrics(
            name,
            prediction,
            teacher_actions[test_mask],
        )

    print()
    print("=" * 94)
    print("TEACHER-ACTION DEPENDENCY SUMMARY")
    print("=" * 94)

    print(
        f"{'model':38s}"
        f"{'mean_RMSE':>14s}"
        f"{'mean_R2':>14s}"
    )

    for name, metrics in results.items():
        print(
            f"{name:38s}"
            f"{metrics['rmse'].mean().item():14.6f}"
            f"{metrics['r2'].mean().item():14.6f}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "results": results,
            "seed": SEED,
            "data_path": str(DATA_PATH),
            "test_rows": int(test_mask.sum()),
        },
        OUTPUT_PATH,
    )

    print()
    print("[DONE] Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
