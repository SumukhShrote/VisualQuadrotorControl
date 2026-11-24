"""Test whether a single ArUco frame is sufficient for learned pose estimation."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=1024)
parser.add_argument("--seed", type=int, default=20261118)
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.set_num_threads(4)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ROOT = Path(__file__).resolve().parents[2]

SOURCES = [
    ROOT / "experiments/state_bottleneck/data/pooled_student_epoch70_all_levels.pt",
    ROOT / "experiments/state_bottleneck/data/perception_teacher_l0/round_100_shard_0000.pt",
    ROOT / "experiments/state_bottleneck/data/perception_mixed_l0/round_101_shard_0000.pt",
]

OUT = ROOT / "experiments/state_bottleneck/perception_compare/frame_pose_mlp"

if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite existing run: {OUT}")

# Position, projected gravity, marker direction, hover-target distance.
# No velocity, angular velocity, or actions.
TARGET_INDICES = [0, 1, 2, 9, 10, 11, 12, 13, 14, 19]

xs, ys = [], []
splits, domains = [], []
steps_all, levels_all = [], []
visible_all, pair_visible_all = [], []

for domain, path in enumerate(SOURCES):
    data = torch.load(path, map_location="cpu", weights_only=False)

    history = data["observations"].float().reshape(-1, 32, 8)

    x = history[:, -1]
    y = data["teacher_observations"].float()[:, TARGET_INDICES]

    visible = ((x >= 0) & (x <= 1)).all(dim=1)
    visible &= torch.isfinite(x).all(dim=1)

    last_two = history[:, -2:, :]
    pair_visible = ((last_two >= 0) & (last_two <= 1)).all(dim=(1, 2))
    pair_visible &= torch.isfinite(last_two).all(dim=(1, 2))

    steps = data["episode_steps"].long()

    if domain == 0:
        levels = data["curriculum_levels"].float()
    else:
        levels = torch.zeros(len(x))

    groups, inverse = torch.unique(
        torch.stack([
            data["environment_ids"].long(),
            data["episode_ids"].long(),
        ], dim=1),
        dim=0,
        return_inverse=True,
    )

    # Exactly matches the previous TCN/Transformer splits.
    generator = torch.Generator().manual_seed(args.seed + 37 * domain)
    permutation = torch.randperm(len(groups), generator=generator)

    n_train = int(0.8 * len(groups))
    n_val = int(0.1 * len(groups))

    group_split = torch.full((len(groups),), 2, dtype=torch.long)
    group_split[permutation[:n_train]] = 0
    group_split[permutation[n_train:n_train+n_val]] = 1

    split = group_split[inverse]

    print(
        f"[DATA] domain={domain} rows={len(x)} "
        f"episodes={len(groups)} "
        f"visible={visible.sum().item()}",
        flush=True,
    )

    xs.append(x)
    ys.append(y)
    splits.append(split)
    domains.append(torch.full((len(x),), domain))
    steps_all.append(steps)
    levels_all.append(levels)
    visible_all.append(visible)
    pair_visible_all.append(pair_visible)

X = torch.cat(xs).to(DEVICE)
Y = torch.cat(ys).to(DEVICE)
SPLIT = torch.cat(splits).to(DEVICE)
DOMAIN = torch.cat(domains).to(DEVICE)
STEP = torch.cat(steps_all).to(DEVICE)
LEVEL = torch.cat(levels_all).to(DEVICE)
VISIBLE = torch.cat(visible_all).to(DEVICE)
PAIR_VISIBLE = torch.cat(pair_visible_all).to(DEVICE)

train_mask = (SPLIT == 0) & VISIBLE

X_MEAN = X[train_mask].mean(0)
X_STD = X[train_mask].std(0).clamp_min(0.01)

Y_MEAN = Y[train_mask].mean(0)
Y_STD = Y[train_mask].std(0).clamp_min(0.02)

X = (X - X_MEAN) / X_STD
Y_NORM = (Y - Y_MEAN) / Y_STD


class FramePoseMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


model = FramePoseMLP().to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=1e-4
)

print(
    "[MODEL] frame_pose_mlp "
    f"parameters={sum(p.numel() for p in model.parameters())}",
    flush=True,
)

train_pools = [
    torch.where(train_mask & (DOMAIN == d))[0]
    for d in range(3)
]

if any(len(p) == 0 for p in train_pools):
    raise RuntimeError("Empty domain training pool")


@torch.no_grad()
def validation_loss(domain):
    ids = torch.where(
        (SPLIT == 1) & (DOMAIN == domain) & VISIBLE
    )[0]

    if len(ids) == 0:
        raise RuntimeError("Empty validation domain")

    model.eval()

    total = 0.0

    for batch in ids.split(1024):
        total += F.mse_loss(
            model(X[batch]),
            Y_NORM[batch],
            reduction="sum",
        ).item()

    return total / (len(ids) * 10)


@torch.no_grad()
def report(name, mask):
    ids = torch.where(mask)[0]

    if len(ids) == 0:
        print(f"[TEST] {name}: empty")
        return

    model.eval()

    squared = torch.zeros(10, device=DEVICE)
    baseline_squared = torch.zeros(10, device=DEVICE)

    for batch in ids.split(1024):
        prediction = model(X[batch]) * Y_STD + Y_MEAN

        squared += (
            prediction - Y[batch]
        ).square().sum(0)

        baseline_squared += (
            Y_MEAN - Y[batch]
        ).square().sum(0)

    rmse = (squared / len(ids)).sqrt().cpu()
    baseline = (baseline_squared / len(ids)).sqrt().cpu()

    print(f"[TEST] {name} n={len(ids)}", flush=True)

    groups = {
        "position": slice(0, 3),
        "gravity": slice(3, 6),
        "direction": slice(6, 9),
        "distance": slice(9, 10),
    }

    for key, sl in groups.items():
        values = [round(v, 6) for v in rmse[sl].tolist()]
        constant = [round(v, 6) for v in baseline[sl].tolist()]

        print(
            f"  {key:9s} rmse={values} "
            f"constant_rmse={constant}",
            flush=True,
        )


OUT.mkdir(parents=True)

best = float("inf")

num_updates = math.ceil(train_mask.sum().item() / args.batch)

for epoch in range(1, args.epochs + 1):
    model.train()
    running = 0.0

    for _ in range(num_updates):
        sample_ids = []

        for domain, pool in enumerate(train_pools):
            count = args.batch // 3 + int(domain < args.batch % 3)

            chosen = pool[
                torch.randint(
                    len(pool),
                    (count,),
                    device=DEVICE,
                )
            ]

            sample_ids.append(chosen)

        batch = torch.cat(sample_ids)

        optimizer.zero_grad(set_to_none=True)

        loss = F.mse_loss(
            model(X[batch]),
            Y_NORM[batch],
        )

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()
        running += loss.item()

    val = [validation_loss(d) for d in range(3)]
    score = sum(val) / 3

    if score < best:
        best = score

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_mean": X_MEAN.cpu(),
            "input_std": X_STD.cpu(),
            "target_mean": Y_MEAN.cpu(),
            "target_std": Y_STD.cpu(),
            "target_indices": TARGET_INDICES,
            "epoch": epoch,
            "validation_loss": score,
        }, OUT / "best.pt")

    print(
        f"[EPOCH] {epoch:02d}/{args.epochs} "
        f"train={running/num_updates:.6f} "
        f"val_student={val[0]:.6f} "
        f"val_teacher={val[1]:.6f} "
        f"val_mixed={val[2]:.6f} "
        f"best={best:.6f}",
        flush=True,
    )

checkpoint = torch.load(
    OUT / "best.pt", map_location=DEVICE, weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

print(
    f"[BEST] epoch={checkpoint['epoch']}",
    flush=True,
)

for domain, name in enumerate(("student", "teacher", "mixed")):
    test = (SPLIT == 2) & (DOMAIN == domain) & VISIBLE

    report(name + "_all_visible", test)

    matched = (
        test
        & PAIR_VISIBLE
        & (STEP >= 31)
        & (LEVEL.abs() < 1e-6)
    )

    report(name + "_level0_mature_visible", matched)

    if domain == 0:
        print(
            "[MATCHED GEOMETRY COUNT]",
            matched.sum().item(),
            flush=True,
        )

        if matched.sum().item() != 1252:
            raise RuntimeError(
                "Test split does not match the previous experiment"
            )

print(f"[DONE] checkpoint={OUT / 'best.pt'}", flush=True)
