"""Matched synthetic-pixel corners vs oracle-mask perception ablation.

Four configurations:
  corners at 64 / 256
  oracle marker masks at 64 / 256

Masks are rasterized from simulator corner coordinates, not rendered RGB.
This tests representation and pixel sampling, NOT real detector accuracy.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--representation", choices=["corners", "mask"], required=True)
parser.add_argument("--resolution", type=int, choices=[64, 256], required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=20261118)
parser.add_argument("--tag", required=True)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_num_threads(4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = Path(__file__).resolve().parents[2]

out = (
    root / "experiments/state_bottleneck/perception_2x2"
    / args.tag
)

if out.exists():
    raise FileExistsError(f"Existing run preserved: {out}")

sources = [
    root / "experiments/state_bottleneck/data/"
    "pooled_student_epoch70_all_levels.pt",

    root / "experiments/state_bottleneck/data/"
    "perception_teacher_l0/round_100_shard_0000.pt",

    root / "experiments/state_bottleneck/data/"
    "perception_mixed_l0/round_101_shard_0000.pt",
]

state_indices = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 19]

xs, ys, splits = [], [], []
domains, levels, episode_steps = [], [], []

for d, path in enumerate(sources):
    data = torch.load(path, map_location="cpu", weights_only=False)

    x = data["observations"].float().reshape(-1, 32, 8)
    y = data["teacher_observations"].float()[:, state_indices]

    groups, inverse = torch.unique(
        torch.stack(
            [
                data["environment_ids"].long(),
                data["episode_ids"].long(),
            ],
            dim=1,
        ),
        dim=0,
        return_inverse=True,
    )

    generator = torch.Generator().manual_seed(args.seed + 37 * d)
    permutation = torch.randperm(len(groups), generator=generator)

    n_train = int(0.8 * len(groups))
    n_val = int(0.1 * len(groups))

    group_split = torch.full((len(groups),), 2, dtype=torch.long)
    group_split[permutation[:n_train]] = 0
    group_split[permutation[n_train:n_train + n_val]] = 1

    split = group_split[inverse]

    xs.append(x)
    ys.append(y)
    splits.append(split)

    domains.append(torch.full((len(x),), d, dtype=torch.long))

    levels.append(
        data["curriculum_levels"].float()
        if d == 0 else torch.zeros(len(x))
    )

    episode_steps.append(data["episode_steps"].long())

    print(
        f"[DATA] domain={d} rows={len(x)} "
        f"episodes={len(groups)} "
        f"train={(split == 0).sum().item()} "
        f"val={(split == 1).sum().item()} "
        f"test={(split == 2).sum().item()}",
        flush=True,
    )


X = torch.cat(xs).to(device)
Y = torch.cat(ys).to(device)

SPLIT = torch.cat(splits).to(device)
DOMAIN = torch.cat(domains).to(device)
LEVEL = torch.cat(levels).to(device)
STEP = torch.cat(episode_steps).to(device)

mean = Y[SPLIT == 0].mean(0)
std = Y[SPLIT == 0].std(0).clamp_min(0.02)
Y_NORMALIZED = (Y - mean) / std

train_pools = [
    torch.where((SPLIT == 0) & (DOMAIN == d))[0]
    for d in range(3)
]

val_pools = [
    torch.where((SPLIT == 1) & (DOMAIN == d))[0]
    for d in range(3)
]

visible_last_two = (
    (X[:, -2:, :] >= 0)
    & (X[:, -2:, :] <= 1)
).all(dim=(1, 2))

matched = (
    (SPLIT == 2)
    & (DOMAIN == 0)
    & (LEVEL.abs() < 1e-6)
    & (STEP >= 31)
    & visible_last_two
)

matched_ids = torch.where(matched)[0]

print("[MATCHED TEST]", len(matched_ids), flush=True)

if len(matched_ids) != 1252:
    raise RuntimeError("Episode split differs from previous benchmark")


def pixel_quantize(x):
    """Model a pixel-center corner measurement, not subpixel detection."""

    valid = ((x >= 0) & (x <= 1)).all(-1, keepdim=True)

    q = (
        torch.floor(x * args.resolution)
        .clamp(0, args.resolution - 1)
        + 0.5
    ) / args.resolution

    return torch.where(valid, q, torch.full_like(q, -1.0))


def rasterize_mask(x):
    """Rasterize ideal quadrilateral silhouettes at native resolution."""

    batch, length, _ = x.shape
    p = x.reshape(-1, 4, 2)

    valid = ((p >= 0) & (p <= 1)).all(dim=(1, 2))

    coords = (
        torch.arange(
            args.resolution,
            device=x.device,
            dtype=x.dtype,
        ) + 0.5
    ) / args.resolution

    grid_y = coords[None, :, None]
    grid_x = coords[None, None, :]

    area = (
        (p[:, 1, 0] - p[:, 0, 0])
        * (p[:, 2, 1] - p[:, 0, 1])
        -
        (p[:, 1, 1] - p[:, 0, 1])
        * (p[:, 2, 0] - p[:, 0, 0])
    )

    positive = area[:, None, None] >= 0

    inside = torch.ones(
        (batch * length, args.resolution, args.resolution),
        device=x.device,
        dtype=torch.bool,
    )

    for k in range(4):
        a = p[:, k]
        edge = p[:, (k + 1) % 4] - a

        cross = (
            edge[:, 0, None, None]
            * (grid_y - a[:, 1, None, None])
            -
            edge[:, 1, None, None]
            * (grid_x - a[:, 0, None, None])
        )

        inside &= torch.where(
            positive, cross >= 0, cross <= 0
        )

    inside &= valid[:, None, None]

    return inside[:, None].float()


class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        incoming = 1

        for channels in (8, 16, 32, 64):
            layers.extend([
                nn.Conv2d(incoming, channels, 3, stride=2, padding=1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
            ])
            incoming = channels

        self.net = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CausalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()

        self.padding = 2 * dilation

        self.conv1 = nn.Conv1d(
            128, 128, 3, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            128, 128, 3, dilation=dilation
        )

    def forward(self, x):
        h = F.gelu(
            self.conv1(
                F.pad(x, (self.padding, 0))
            )
        )

        h = self.conv2(
            F.pad(h, (self.padding, 0))
        )

        return (x + h) * 0.70710678


class StateEstimator(nn.Module):
    def __init__(self):
        super().__init__()

        if args.representation == "corners":
            self.frame = nn.Sequential(
                nn.Linear(8, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
            )
        else:
            self.frame = FrameCNN()

        self.temporal = nn.Sequential(
            *(CausalBlock(d) for d in (1, 2, 4, 8))
        )

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 13),
        )

    def forward(self, x):
        if args.representation == "corners":
            h = self.frame(pixel_quantize(x))
        else:
            b, t, _ = x.shape

            masks = rasterize_mask(x)
            h = self.frame(masks).reshape(b, t, 128)

        features = self.temporal(
            h.transpose(1, 2)
        )[:, :, -1]

        return self.head(features)


model = StateEstimator().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
)

print(
    f"[CONFIG] representation={args.representation} "
    f"resolution={args.resolution} "
    f"parameters={sum(p.numel() for p in model.parameters())} "
    f"device={device}",
    flush=True,
)

# All experiments use the same effective batch and update count.
effective_batch = 48

microbatch = (
    48 if args.representation == "corners"
    else 16 if args.resolution == 64
    else 8
)


@torch.no_grad()
def evaluate(indices, limit=256):
    model.eval()

    if len(indices) > limit:
        generator = torch.Generator().manual_seed(args.seed + 123)

        chosen = torch.randperm(
            len(indices),
            generator=generator,
        )[:limit].to(indices.device)

        indices = indices[chosen]

    squared_error = torch.zeros(13, device=device)
    normalized_total = 0.0

    for ids in indices.split(microbatch):
        prediction = model(X[ids]) * std + mean
        difference = prediction - Y[ids]

        squared_error += difference.square().sum(0)

        normalized_total += F.mse_loss(
            (prediction - mean) / std,
            Y_NORMALIZED[ids],
            reduction="sum",
        ).item()

    rmse = (squared_error / len(indices)).sqrt().cpu().tolist()
    normalized_mse = normalized_total / (len(indices) * 13)

    return normalized_mse, rmse


out.mkdir(parents=True)

best = float("inf")

for update in range(1, args.steps + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    selections = []

    for pool in train_pools:
        sampled = torch.randint(
            len(pool),
            (effective_batch // 3,),
            device=device,
        )
        selections.append(pool[sampled])

    batch_ids = torch.cat(selections)

    running_loss = 0.0

    for ids in batch_ids.split(microbatch):
        loss = F.mse_loss(
            model(X[ids]),
            Y_NORMALIZED[ids],
        )

        loss = loss * (len(ids) / effective_batch)

        loss.backward()
        running_loss += loss.detach().item()

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if update == 1 or update % 100 == 0 or update == args.steps:
        validation = [
            evaluate(pool)[0]
            for pool in val_pools
        ]

        score = sum(validation) / 3

        if score < best:
            best = score

            torch.save({
                "model_state_dict": model.state_dict(),
                "target_mean": mean.cpu(),
                "target_std": std.cpu(),
                "state_indices": state_indices,
                "representation": args.representation,
                "resolution": args.resolution,
                "update": update,
                "validation_loss": best,
                "sensor_type": "synthetic_oracle_raster",
            }, out / "best.pt")

        print(
            f"[UPDATE] {update}/{args.steps} "
            f"train={running_loss:.6f} "
            f"val_student={validation[0]:.6f} "
            f"val_teacher={validation[1]:.6f} "
            f"val_mixed={validation[2]:.6f} "
            f"best={best:.6f}",
            flush=True,
        )


checkpoint = torch.load(
    out / "best.pt",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

print(
    f"[BEST] update={checkpoint['update']} "
    f"validation={best:.6f}",
    flush=True,
)

test_sets = [
    ("student_level0_mature_visible", matched_ids),
    (
        "teacher_heldout",
        torch.where((SPLIT == 2) & (DOMAIN == 1))[0],
    ),
    (
        "mixed_heldout",
        torch.where((SPLIT == 2) & (DOMAIN == 2))[0],
    ),
]

for name, ids in test_sets:
    limit = 2000 if name.startswith("student") else 512

    normalized, rmse = evaluate(ids, limit=limit)

    print(
        f"[TEST] {name} n={min(len(ids), limit)} "
        f"normalized_mse={normalized:.6f}",
        flush=True,
    )

    for label, sl in (
        ("position", slice(0, 3)),
        ("velocity", slice(3, 6)),
        ("gravity", slice(6, 9)),
        ("direction", slice(9, 12)),
        ("distance", slice(12, 13)),
    ):
        print(
            f"  {label}: "
            f"{[round(v, 5) for v in rmse[sl]]}",
            flush=True,
        )

print("[DONE]", out / "best.pt", flush=True)
