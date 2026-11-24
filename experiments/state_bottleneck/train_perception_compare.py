"""Matched state-only TCN vs Transformer experiment."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

P = argparse.ArgumentParser()
P.add_argument("--model", choices=("tcn", "transformer"), required=True)
P.add_argument("--epochs", type=int, default=20)
P.add_argument("--batch", type=int, default=512)
P.add_argument("--seed", type=int, default=20261118)
P.add_argument("--lr", type=float, default=3e-4)
P.add_argument("--tag", required=True)
A = P.parse_args()

if A.epochs < 1 or A.batch < 3 or A.lr <= 0:
    raise ValueError("Invalid training hyperparameters")

torch.set_num_threads(4)
torch.manual_seed(A.seed)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[2]

SOURCES = [
    ROOT / "experiments/state_bottleneck/data/pooled_student_epoch70_all_levels.pt",
    ROOT / "experiments/state_bottleneck/data/perception_teacher_l0/round_100_shard_0000.pt",
    ROOT / "experiments/state_bottleneck/data/perception_mixed_l0/round_101_shard_0000.pt",
]

STATE_INDICES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 19]

OUT = ROOT / "experiments/state_bottleneck/perception_compare" / A.tag

if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite: {OUT}")

x_parts, y_parts = [], []
domain_parts, split_parts = [], []
step_parts, level_parts = [], []

for domain, path in enumerate(SOURCES):
    data = torch.load(path, map_location="cpu", weights_only=False)

    x = data["observations"].float().reshape(-1, 32, 8)
    y = data["teacher_observations"].float()[:, STATE_INDICES]

    steps = data["episode_steps"].long().reshape(-1)
    env = data["environment_ids"].long().reshape(-1)
    episode = data["episode_ids"].long().reshape(-1)

    if domain == 0:
        level = data["curriculum_levels"].float().reshape(-1)
    else:
        level = torch.zeros(len(x))

    if y.shape != (len(x), 13):
        raise RuntimeError(f"Invalid label shape in {path}")

    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError(f"Non-finite data in {path}")

    # Entire episodes must belong to only one split.
    groups, inverse = torch.unique(
        torch.stack((env, episode), dim=1),
        dim=0,
        return_inverse=True,
    )

    generator = torch.Generator().manual_seed(A.seed + 37 * domain)
    permutation = torch.randperm(len(groups), generator=generator)

    n_train = int(0.80 * len(groups))
    n_val = int(0.10 * len(groups))

    group_split = torch.full((len(groups),), 2, dtype=torch.long)
    group_split[permutation[:n_train]] = 0
    group_split[permutation[n_train:n_train+n_val]] = 1

    split = group_split[inverse]

    print(
        f"[DATA] source={domain} rows={len(x)} "
        f"episodes={len(groups)} "
        f"train={(split==0).sum().item()} "
        f"val={(split==1).sum().item()} "
        f"test={(split==2).sum().item()}",
        flush=True,
    )

    x_parts.append(x)
    y_parts.append(y)
    domain_parts.append(torch.full((len(x),), domain))
    split_parts.append(split)
    step_parts.append(steps)
    level_parts.append(level)

X = torch.cat(x_parts).to(DEVICE)
Y = torch.cat(y_parts).to(DEVICE)
DOMAIN = torch.cat(domain_parts).to(DEVICE)
SPLIT = torch.cat(split_parts).to(DEVICE)
STEP = torch.cat(step_parts).to(DEVICE)
LEVEL = torch.cat(level_parts).to(DEVICE)

# Compute normalization statistics using training episodes only.
MEAN = Y[SPLIT == 0].mean(0)
STD = Y[SPLIT == 0].std(0).clamp_min(0.02)

Y_NORMALIZED = (Y - MEAN) / STD

TRAIN_POOLS = [
    torch.where((SPLIT == 0) & (DOMAIN == d))[0]
    for d in range(3)
]

if any(len(pool) == 0 for pool in TRAIN_POOLS):
    raise RuntimeError("Missing source in training split")


class FrameEncoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )

    def forward(self, x):
        return self.net(x)


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(width, width, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, dilation=dilation)

    def forward(self, x):
        h = F.gelu(self.conv1(F.pad(x, (self.pad, 0))))
        h = self.conv2(F.pad(h, (self.pad, 0)))
        return (x + h) * (2 ** -0.5)


class StateEstimator(nn.Module):
    def __init__(self, kind, width=128):
        super().__init__()

        self.kind = kind
        self.frame = FrameEncoder(width)

        if kind == "tcn":
            self.temporal = nn.Sequential(
                *(CausalBlock(width, d) for d in (1, 2, 4, 8))
            )
        else:
            self.pos = nn.Parameter(torch.zeros(1, 32, width))
            nn.init.normal_(self.pos, std=0.02)

            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=256,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

            self.temporal = nn.TransformerEncoder(
                layer,
                num_layers=4,
                enable_nested_tensor=False,
            )

        self.head = nn.Sequential(
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 13),
        )

    def forward(self, x):
        h = self.frame(x)

        if self.kind == "tcn":
            z = self.temporal(h.transpose(1, 2))[:, :, -1]
        else:
            z = self.temporal(h + self.pos)[:, -1]

        return self.head(z)


MODEL = StateEstimator(A.model).to(DEVICE)

OPTIMIZER = torch.optim.AdamW(
    MODEL.parameters(),
    lr=A.lr,
    weight_decay=1e-4,
)

print(
    f"[MODEL] type={A.model} "
    f"parameters={sum(p.numel() for p in MODEL.parameters())} "
    f"device={DEVICE}",
    flush=True,
)


@torch.no_grad()
def evaluate(indices):
    if len(indices) == 0:
        return float("nan")

    MODEL.eval()
    total = 0.0

    for ids in indices.split(1024):
        total += F.mse_loss(
            MODEL(X[ids]),
            Y_NORMALIZED[ids],
            reduction="sum",
        ).item()

    return total / (len(indices) * 13)


@torch.no_grad()
def physical_report(name, indices):
    if len(indices) == 0:
        print(f"[TEST] {name} empty")
        return

    MODEL.eval()

    sum_squared = torch.zeros(13, device=DEVICE)
    baseline_squared = torch.zeros(13, device=DEVICE)

    for ids in indices.split(1024):
        prediction = MODEL(X[ids]) * STD + MEAN

        sum_squared += (
            (prediction - Y[ids]) ** 2
        ).sum(0)

        # Constant-state predictor as a sanity-check baseline.
        baseline_squared += (
            (MEAN - Y[ids]) ** 2
        ).sum(0)

    rmse = (sum_squared / len(indices)).sqrt().cpu().tolist()
    baseline = (
        baseline_squared / len(indices)
    ).sqrt().cpu().tolist()

    print(f"[TEST] {name} n={len(indices)}", flush=True)

    groups = [
        ("position", slice(0, 3)),
        ("velocity", slice(3, 6)),
        ("gravity", slice(6, 9)),
        ("direction", slice(9, 12)),
        ("distance", slice(12, 13)),
    ]

    for group, sl in groups:
        print(
            f"  {group:9s} "
            f"rmse={[round(v, 5) for v in rmse[sl]]} "
            f"constant_rmse={[round(v, 5) for v in baseline[sl]]}",
            flush=True,
        )


OUT.mkdir(parents=True)

train_rows = sum(len(pool) for pool in TRAIN_POOLS)
updates_per_epoch = math.ceil(train_rows / A.batch)

best_validation = float("inf")

for epoch in range(1, A.epochs + 1):
    MODEL.train()

    running_loss = 0.0

    for _ in range(updates_per_epoch):
        # Balanced sampling across the three sources.
        ids = torch.cat([
            pool[
                torch.randint(
                    len(pool),
                    (A.batch // 3 + (k < A.batch % 3),),
                    device=DEVICE,
                )
            ]
            for k, pool in enumerate(TRAIN_POOLS)
        ])

        OPTIMIZER.zero_grad(set_to_none=True)

        loss = F.mse_loss(
            MODEL(X[ids]),
            Y_NORMALIZED[ids],
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")

        loss.backward()

        nn.utils.clip_grad_norm_(
            MODEL.parameters(),
            max_norm=1.0,
        )

        OPTIMIZER.step()
        running_loss += loss.item()

    # Equal-weight validation across trajectory sources.
    validation = [
        evaluate(torch.where((SPLIT == 1) & (DOMAIN == d))[0])
        for d in range(3)
    ]

    score = sum(validation) / 3

    if score < best_validation:
        best_validation = score

        torch.save(
            {
                "model_state_dict": MODEL.state_dict(),
                "target_mean": MEAN.cpu(),
                "target_std": STD.cpu(),
                "state_indices": STATE_INDICES,
                "kind": A.model,
                "epoch": epoch,
                "validation_normalized_mse": score,
                "seed": A.seed,
            },
            OUT / "best.pt",
        )

    print(
        f"[EPOCH] {epoch:02d}/{A.epochs} "
        f"train_mse={running_loss/updates_per_epoch:.5f} "
        f"val_student={validation[0]:.5f} "
        f"val_teacher={validation[1]:.5f} "
        f"val_mixed={validation[2]:.5f} "
        f"best_val={best_validation:.5f}",
        flush=True,
    )


# Evaluate the selected checkpoint on untouched test episodes.
checkpoint = torch.load(
    OUT / "best.pt",
    map_location=DEVICE,
    weights_only=False,
)
MODEL.load_state_dict(checkpoint["model_state_dict"])

print("[BEST] Loaded for episode-held-out evaluation", flush=True)

# Geometric reconstruction requires current and previous visible corners.
visible = (
    (X[:, -2:, :] >= 0)
    & (X[:, -2:, :] <= 1)
).all(dim=(1, 2))

for domain, name in enumerate(("student", "teacher", "mixed")):
    test = (SPLIT == 2) & (DOMAIN == domain)

    physical_report(
        name + "_all",
        torch.where(test)[0],
    )

    mature_visible = (
        test
        & (LEVEL.abs() < 1e-6)
        & (STEP >= 31)
        & visible
    )

    physical_report(
        name + "_level0_mature_visible",
        torch.where(mature_visible)[0],
    )

    early = (
        test
        & (LEVEL.abs() < 1e-6)
        & (STEP < 31)
    )

    physical_report(
        name + "_level0_early",
        torch.where(early)[0],
    )

print(f"[DONE] {A.model} checkpoint={OUT / 'best.pt'}", flush=True)
