import argparse
import copy
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path.cwd()

DATA = ROOT / "experiments/state_bottleneck/data/pooled_student_epoch70_all_levels.pt"

AUX_SCRIPT = ROOT / "experiments/corners_tcn_student_v1/train_aux13_student.py"

AUX_CKPT = ROOT / (
    "checkpoints/corners_tcn_student_v1/"
    "stage4_aux13_ft/stage4_aux13_ft_best.pt"
)

OUT = ROOT / "experiments/state_bottleneck/results"

# Position error, linear velocity, gravity, marker direction, distance
INDICES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 19]


def load_historical_module():
    spec = importlib.util.spec_from_file_location(
        "historical_aux13", AUX_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data(module):
    data = torch.load(
        DATA, map_location="cpu", weights_only=False
    )

    print("[DATA KEYS]", list(data.keys()), flush=True)

    states = data["teacher_observations"].float()[:, INDICES]
    actions = data["teacher_actions"].float()

    train, val, test = module.make_episode_split(
        data["environment_ids"].long(),
        data["episode_ids"].long(),
    )

    print(
        f"[DATA] total={len(states)} "
        f"train={int(train.sum())} "
        f"val={int(val.sum())} "
        f"test={int(test.sum())}",
        flush=True,
    )

    assert states.shape[1] == 13
    assert actions.shape[1] == 4

    return data, states, actions, train, val, test


class OracleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 4),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict_batched(model, x, device):
    predictions = []

    for batch in x.split(4096):
        predictions.append(model(batch.to(device)).cpu())

    return torch.cat(predictions)


def metrics(pred, target):
    rmse = (pred - target).square().mean(0).sqrt()

    r2 = 1.0 - (
        (pred - target).square().sum(0)
        /
        (target - target.mean(0))
        .square().sum(0).clamp_min(1e-8)
    )

    return rmse, r2


def train_oracle(module, device, epochs):
    _, states, actions, train, val, test = load_data(module)

    mean = states[train].mean(0)
    std = states[train].std(0).clamp_min(1e-5)

    x = ((states - mean) / std).clamp(-15, 15)

    action_scale = (
        actions[train].std(0).clamp_min(0.05).to(device)
    )

    loader = DataLoader(
        TensorDataset(x[train], actions[train]),
        batch_size=2048,
        shuffle=True,
        pin_memory=True,
    )

    model = OracleMLP().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-5,
    )

    best_loss = float("inf")
    best_state = None
    stale = 0

    OUT.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            prediction = model(xb)

            loss = (
                ((prediction - yb) / action_scale)
                .square().mean()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():
            pred = predict_batched(model, x[val], device)

            val_loss = (
                ((pred - actions[val]) /
                 action_scale.cpu())
                .square().mean().item()
            )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0

            torch.save(
                {
                    "model": {
                        k: v.detach().cpu()
                        for k, v in best_state.items()
                    },
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_indices": INDICES,
                    "epoch": epoch,
                    "validation_loss": best_loss,
                },
                OUT / "oracle13_teacher_bc.pt",
            )
        else:
            stale += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[ORACLE] epoch={epoch} "
                f"val_loss={val_loss:.6f} "
                f"best={best_loss:.6f}",
                flush=True,
            )

        if stale >= 12:
            break

    model.load_state_dict(best_state)
    model.eval()

    pred = predict_batched(model, x[test], device)

    rmse, r2 = metrics(pred, actions[test])

    print("\n========== ORACLE HELD-OUT RESULTS ==========")
    print("Per-action RMSE:", rmse.tolist())
    print("Per-action R2:", r2.tolist())
    print("Mean action R2:", r2.mean().item())
    print("[SAVED]", OUT / "oracle13_teacher_bc.pt")


@torch.no_grad()
def evaluate_state_estimator(module, device):
    data, states, _, _, _, test = load_data(module)

    checkpoint = torch.load(
        AUX_CKPT, map_location="cpu", weights_only=False
    )

    policy = module.build_policy(device)

    policy.student.load_state_dict(
        checkpoint["student_state_dict"],
        strict=True,
    )

    policy.eval()

    aux_head = module.AuxiliaryStateHead().to(device)

    aux_head.load_state_dict(
        checkpoint["auxiliary_state_head_dict"],
        strict=True,
    )

    aux_head.eval()

    candidates = [
        "observations",
        "student_observations",
        "corner_histories",
        "corner_history",
    ]

    obs_key = next(
        (
            k for k in candidates
            if k in data
            and isinstance(data[k], torch.Tensor)
            and data[k].ndim == 2
            and data[k].shape[1] == 256
        ),
        None,
    )

    if obs_key is None:
        obs_key = next(
            (
                k for k, v in data.items()
                if isinstance(v, torch.Tensor)
                and v.shape == (len(states), 256)
            ),
            None,
        )

    if obs_key is None:
        raise KeyError("Cannot locate 256-D corner histories")

    mature_test = (
        test & (data["episode_steps"].long() >= 31)
    )

    observations = data[obs_key][mature_test].float()
    target = states[mature_test]

    mean = checkpoint["auxiliary_target_mean"].to(device)
    std = checkpoint["auxiliary_target_std"].to(device)

    predictions = []

    for batch in observations.split(4096):
        latent = policy.encode_temporal_state(
            batch.to(device)
        )

        estimate = aux_head(latent) * std + mean
        predictions.append(estimate.cpu())

    prediction = torch.cat(predictions)

    rmse, r2 = metrics(prediction, target)

    labels = [
        "position_x", "position_y", "position_z",
        "velocity_x", "velocity_y", "velocity_z",
        "gravity_x", "gravity_y", "gravity_z",
        "marker_dir_x", "marker_dir_y", "marker_dir_z",
        "marker_distance",
    ]

    print("\n========== STATE ESTIMATION RESULTS ==========")
    print("Held-out mature samples:", len(target))

    for label, error, score in zip(
        labels, rmse.tolist(), r2.tolist()
    ):
        print(
            f"{label:>18s}: "
            f"RMSE={error:.5f}  R2={score:.4f}"
        )

    print("\nMean R2:", r2.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job",
        required=True,
        choices=["oracle", "state-eval"],
    )
    parser.add_argument("--epochs", type=int, default=70)

    args = parser.parse_args()

    torch.manual_seed(7500)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("[START]", args.job, device, flush=True)

    module = load_historical_module()

    if args.job == "oracle":
        train_oracle(module, device, args.epochs)
    else:
        evaluate_state_estimator(module, device)


if __name__ == "__main__":
    main()
