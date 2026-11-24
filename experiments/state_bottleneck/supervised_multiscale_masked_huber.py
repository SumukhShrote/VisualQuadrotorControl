#!/usr/bin/env python3
"""Episode-disjoint supervised corners-to-state observability experiment."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# Indices into the reconstructed 16-dimensional target.
GROUPS = {
    "position_m": [0, 1, 2],
    "linear_velocity_mps": [3, 4, 5],
    "angular_velocity_radps": [6, 7, 8],
    "projected_gravity": [9, 10, 11],
    "marker_direction": [12, 13, 14],
    "marker_distance_m": [15],
}

# Indices into the simulator's 20-dimensional teacher observation.
# Excludes previous actions [15:19], which are not sensor inputs.
TARGET_IDX = list(range(15)) + [19]



class TemporalAttentionBranch(nn.Module):
    """Independent attention encoder for one state group."""

    def __init__(
        self,
        window,
        output_dim,
        width=64,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()

        self.window = window

        self.input_projection = nn.Linear(12, width)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, window, width)
        )

        nn.init.normal_(
            self.position_embedding,
            std=0.02,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=width * 2,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.query = nn.Parameter(
            torch.zeros(1, 1, width)
        )

        nn.init.normal_(self.query, std=0.02)

        self.pool = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(width)

        self.output = nn.Sequential(
            nn.Linear(width, 96),
            nn.GELU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, history):
        # Strictly restrict each branch to its assigned
        # temporal window. No privileged inputs.
        x = history[:, -self.window:, :]

        # Derive per-corner visibility from existing observations.
        pairs = x.reshape(x.shape[0], self.window, 4, 2)
        missing = (pairs < -0.99).all(dim=-1)
        validity = (~missing).to(dtype=x.dtype)

        x = torch.cat((x, validity), dim=-1)

        z = (
            self.input_projection(x)
            + self.position_embedding
        )

        z = self.encoder(z)

        query = self.query.expand(
            history.shape[0], -1, -1
        )

        pooled, _ = self.pool(
            query,
            z,
            z,
            need_weights=False,
        )

        # Preserve direct access to newest visual geometry.
        features = self.norm(
            pooled[:, 0, :] + z[:, -1, :]
        )

        return self.output(features)


class TemporalEstimator(nn.Module):
    """
    Supervised multi-timescale state estimator.

    Inputs:
        [batch, 32, 8] ArUco corner history.

    Outputs:
        [batch, 16] reconstructed physical state.

    Branches:
        Geometry:  8 frames -> position, gravity,
                    marker direction, hover distance.
        Linear:   32 frames -> linear velocity.
        Angular:   8 frames -> angular velocity.

    No actions, IMU, ground-truth state, or other
    privileged information enters the estimator.
    """

    def __init__(self):
        super().__init__()

        self.geometry = TemporalAttentionBranch(
            window=8,
            output_dim=10,
        )

        self.linear_velocity = TemporalAttentionBranch(
            window=32,
            output_dim=3,
        )

        self.angular_velocity = TemporalAttentionBranch(
            window=8,
            output_dim=3,
        )

    def forward(self, observations):
        if (
            observations.ndim != 3
            or observations.shape[1:] != (32, 8)
        ):
            raise RuntimeError(
                f"Expected [B,32,8], got "
                f"{tuple(observations.shape)}"
            )

        geom = self.geometry(observations)
        linear = self.linear_velocity(observations)
        angular = self.angular_velocity(observations)

        # Match the exact 16-D target ordering used by
        # our existing supervised reconstruction experiment.
        return torch.cat([
            geom[:, 0:3],    # Position
            linear,         # Linear velocity
            angular,        # Angular velocity
            geom[:, 3:6],   # Projected gravity
            geom[:, 6:9],   # Marker direction
            geom[:, 9:10],  # Hover distance
        ], dim=-1)

def get_array(data, key, dtype=None):
    if key not in data:
        raise KeyError(
            f"Missing {key!r}; available: {sorted(data)}"
        )

    a = data[key]

    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    else:
        a = np.asarray(a)

    if dtype is not None:
        a = a.astype(dtype, copy=False)

    return a


def build_masks(level, env, ep, rng):
    # Group by curriculum level, environment ID, and episode ID.
    # Never split samples from one episode across train/test.
    triples = np.column_stack((
        np.rint(level * 1000).astype(np.int64),
        env.astype(np.int64),
        ep.astype(np.int64),
    ))

    groups, assignment = np.unique(
        triples, axis=0, return_inverse=True
    )

    group_split = np.full(
        len(groups), -1, dtype=np.int64
    )

    for lev in np.unique(groups[:, 0]):
        ids = np.flatnonzero(groups[:, 0] == lev)
        rng.shuffle(ids)

        n = len(ids)

        if n < 3:
            raise ValueError(
                f"Only {n} episode groups at level {lev}"
            )

        nv = max(1, int(round(n * 0.15)))
        nt = max(1, int(round(n * 0.15)))

        if nv + nt >= n:
            raise ValueError(
                "Insufficient episode groups"
            )

        group_split[ids[:nv]] = 1
        group_split[ids[nv:nv + nt]] = 2
        group_split[ids[nv + nt:]] = 0

    return group_split[assignment], groups, group_split


def metrics(truth, pred):
    ans = {}

    for name, idx in GROUPS.items():
        a = truth[:, idx]
        b = pred[:, idx]
        err = b - a

        rmse = math.sqrt(
            float(np.mean(err ** 2))
        )

        denom = float(np.sum(
            (a - a.mean(axis=0, keepdims=True)) ** 2
        ))

        r2 = 1 - float(
            np.sum(err ** 2)
        ) / max(denom, 1e-12)

        err_norm = np.linalg.norm(err, axis=1)

        ans[name] = {
            "rmse_component": rmse,
            "r2": r2,
            "error_norm_p50": float(
                np.quantile(err_norm, 0.50)
            ),
            "error_norm_p95": float(
                np.quantile(err_norm, 0.95)
            ),
        }

        if name in (
            "projected_gravity",
            "marker_direction",
        ):
            norm_a = np.linalg.norm(
                a, axis=1
            ).clip(1e-6)

            norm_b = np.linalg.norm(
                b, axis=1
            ).clip(1e-6)

            cosine = np.clip(
                (a * b).sum(axis=1)
                / (norm_a * norm_b),
                -1,
                1,
            )

            ans[name]["angle_error_deg_p50"] = float(
                np.median(
                    np.degrees(np.arccos(cosine))
                )
            )

    return ans


def predict(model, x, mean, std, device, batch=4096):
    model.eval()
    outputs = []

    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.from_numpy(
                x[start:start + batch]
            ).to(device)

            z = model(xb)

            outputs.append(
                (
                    z.cpu().numpy() * std + mean
                ).astype(np.float32)
            )

    return np.concatenate(outputs)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        default=(
            "dagger_data/tcn_state_probe_v1/"
            "pooled_student_epoch70_all_levels.pt"
        ),
    )

    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20261119)

    ap.add_argument(
        "--histories",
        type=int,
        nargs="+",
        default=[1, 8, 32],
    )

    args = ap.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    rng = np.random.default_rng(args.seed)

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA is unavailable")

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(
        "[DATA] Loading:", args.dataset,
        flush=True,
    )

    data = torch.load(
        args.dataset,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dictionary, got {type(data)}"
        )

    x = get_array(
        data, "observations", np.float32
    )

    if x.ndim == 2 and x.shape[1] == 256:
        x = x.reshape(-1, 32, 8)

    if x.ndim != 3 or x.shape[1:] != (32, 8):
        raise ValueError(
            f"Unexpected corners shape: {x.shape}"
        )

    teacher = get_array(
        data, "teacher_observations", np.float32
    )

    if (
        teacher.ndim != 2
        or teacher.shape[1] != 20
    ):
        raise ValueError(
            f"Unexpected teacher shape: {teacher.shape}"
        )

    level = get_array(
        data, "curriculum_levels", np.float64
    ).reshape(-1)

    env = get_array(
        data, "environment_ids"
    ).reshape(-1)

    ep = get_array(
        data, "episode_ids"
    ).reshape(-1)

    steps = get_array(
        data, "episode_steps"
    ).reshape(-1)

    if not all(
        len(z) == len(x)
        for z in (teacher, level, env, ep, steps)
    ):
        raise ValueError("Dataset row alignment failure")

    # Require fully populated temporal histories.
    valid = (
        (steps >= 31)
        & np.isfinite(x).all(axis=(1, 2))
        & np.isfinite(teacher).all(axis=1)
    )

    print(
        "[DATA] Total:", len(valid),
        "Mature/finite:", int(valid.sum()),
        flush=True,
    )

    x = x[valid]
    teacher = teacher[valid]
    level = level[valid]
    env = env[valid]
    ep = ep[valid]

    y = np.ascontiguousarray(
        teacher[:, TARGET_IDX]
    )

    x = np.ascontiguousarray(x)

    split, groups, group_split = build_masks(
        level, env, ep, rng
    )

    train_ids, val_ids, test_ids = [
        np.flatnonzero(split == s)
        for s in [0, 1, 2]
    ]

    print(
        "[SPLIT] Samples train/val/test:",
        *(len(q) for q in (
            train_ids, val_ids, test_ids
        )),
        flush=True,
    )

    print(
        "[SPLIT] Episode groups train/val/test:",
        *(
            int((group_split == s).sum())
            for s in [0, 1, 2]
        ),
        flush=True,
    )

    print(
        "[DATA] Test level distribution:",
        {
            str(k): int(
                (level[test_ids] == k).sum()
            )
            for k in np.unique(level[test_ids])
        },
        flush=True,
    )

    print(
        "[DATA] Test invalid current corners:",
        float(np.mean(
            np.all(
                x[test_ids, -1, :] < -0.99,
                axis=1,
            )
        )),
        flush=True,
    )

    # Normalization statistics come from training episodes only.
    y_mean = y[train_ids].mean(
        axis=0
    ).astype(np.float32)

    y_std = y[train_ids].std(
        axis=0
    ).clip(1e-3).astype(np.float32)

    protocol = {
        "dataset": str(args.dataset),
        "seed": args.seed,
        "histories": args.histories,
        "target_indices": TARGET_IDX,
        "groups": GROUPS,
        "samples": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "episode_groups": {
            str(s): int(
                (group_split == s).sum()
            )
            for s in [0, 1, 2]
        },
        "mature_history_min_episode_step": 31,
        "train_label_mean": y_mean.tolist(),
        "train_label_std": y_std.tolist(),
    }

    with (output / "protocol.json").open("w") as f:
        json.dump(protocol, f, indent=2)

    all_results = []

    # Constant-state baseline.
    constant_pred = np.broadcast_to(
        y_mean, y[test_ids].shape
    )

    baseline = metrics(
        y[test_ids], constant_pred
    )

    print(
        "[BASELINE] Train-mean constant:",
        " ".join(
            f"{k}: R2={v['r2']:.3f}"
            for k, v in baseline.items()
        ),
        flush=True,
    )

    for name, values in baseline.items():
        all_results.append({
            "history": 0,
            "subset": "constant_train_mean",
            "n": len(test_ids),
            "variable": name,
            **values,
        })

    for H in args.histories:
        if not 1 <= H <= 32:
            raise ValueError(
                f"Invalid history length: {H}"
            )

        print(
            f"\n========== FIT H={H} ==========",
            flush=True,
        )

        model = TemporalEstimator().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
            )
        )

        tx = torch.from_numpy(
            np.ascontiguousarray(
                x[train_ids, -H:]
            )
        )

        ty = torch.from_numpy(
            np.ascontiguousarray(
                (y[train_ids] - y_mean) / y_std
            )
        )

        vx = torch.from_numpy(
            np.ascontiguousarray(
                x[val_ids, -H:]
            )
        ).to(device)

        vy = torch.from_numpy(
            np.ascontiguousarray(
                (y[val_ids] - y_mean) / y_std
            )
        ).to(device)

        loader = DataLoader(
            TensorDataset(tx, ty),
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        best = float("inf")
        best_weights = None
        patience = 0

        for epoch in range(args.epochs):
            model.train()

            total = 0.0
            n = 0

            for bx, by in loader:
                bx = bx.to(
                    device, non_blocking=True
                )
                by = by.to(
                    device, non_blocking=True
                )

                pred = model(bx)
                loss = F.smooth_l1_loss(pred, by)

                optimizer.zero_grad(
                    set_to_none=True
                )
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )

                optimizer.step()

                total += float(
                    loss.detach()
                ) * len(bx)

                n += len(bx)

            model.eval()

            with torch.no_grad():
                val_loss = sum(
                    float(
                        F.smooth_l1_loss(
                            model(
                                vx[k:k + args.batch]
                            ),
                            vy[k:k + args.batch],
                            reduction="sum",
                        )
                    )
                    for k in range(
                        0, len(vx), args.batch
                    )
                ) / (
                    len(vx) * vy.shape[1]
                )

            scheduler.step(val_loss)

            if val_loss < best - 1e-5:
                best = val_loss

                best_weights = {
                    k: v.detach().cpu().clone()
                    for k, v
                    in model.state_dict().items()
                }

                patience = 0
            else:
                patience += 1

            if (
                epoch == 0
                or (epoch + 1) % 5 == 0
                or patience >= 7
            ):
                print(
                    f"[FIT H={H}] "
                    f"epoch={epoch+1:02d} "
                    f"train_objective={total/n:.5f} "
                    f"val_objective={val_loss:.5f} "
                    f"best={best:.5f}",
                    flush=True,
                )

            if patience >= 7 and epoch >= 9:
                break

        model.load_state_dict(best_weights)

        torch.save(
            {
                "model_state_dict": best_weights,
                "history_length": H,
                "target_indices": TARGET_IDX,
                "target_mean": y_mean,
                "target_std": y_std,
            },
            output / f"multiscale_transformer_H{H}.pt",
        )

        p = predict(
            model,
            np.ascontiguousarray(
                x[test_ids, -H:]
            ),
            y_mean,
            y_std,
            device,
        )

        m = metrics(
            y[test_ids], p
        )

        visible_last = ~np.all(
            x[test_ids, -1, :] < -0.99,
            axis=1,
        )

        subsets = [
            (
                "all",
                np.ones(
                    len(test_ids),
                    dtype=bool,
                ),
            ),
            (
                "last_frame_visible",
                visible_last,
            ),
            (
                "last_frame_all_invalid",
                ~visible_last,
            ),
        ]

        subsets += [
            (
                f"level_{lv:.2f}",
                level[test_ids] == lv,
            )
            for lv in np.unique(level[test_ids])
        ]

        for label, mask in subsets:
            if mask.sum() < 20:
                continue

            mm = metrics(
                y[test_ids][mask],
                p[mask],
            )

            for name, values in mm.items():
                all_results.append({
                    "history": H,
                    "subset": label,
                    "n": int(mask.sum()),
                    "variable": name,
                    **values,
                })

        print(
            f"[TEST H={H}]",
            " ".join(
                f"{k}: "
                f"RMSE={v['rmse_component']:.4f}, "
                f"R2={v['r2']:.3f}, "
                f"P95norm={v['error_norm_p95']:.4f}"
                for k, v in m.items()
            ),
            flush=True,
        )

        if H == 32:
            # Preserve the current frame.
            # Shuffle only the preceding observations.
            perm = np.concatenate((
                rng.permutation(H - 1),
                np.array([H - 1]),
            ))

            p_shuffle = predict(
                model,
                np.ascontiguousarray(
                    x[test_ids, -H:][:, perm]
                ),
                y_mean,
                y_std,
                device,
            )

            ms = metrics(
                y[test_ids], p_shuffle
            )

            print(
                "[ABLATION] Shuffled earlier frames:",
                " ".join(
                    f"{k}: R2={v['r2']:.3f}"
                    for k, v in ms.items()
                ),
                flush=True,
            )

            # Perturb only valid corner coordinates.
            # Units are those of the normalized input.
            x_noise = x[
                test_ids, -H:
            ].copy()

            valid_corners = x_noise > -0.99

            eps = rng.normal(
                0, 0.005,
                size=x_noise.shape,
            ).astype(np.float32)

            x_noise += eps * valid_corners

            p_noise = predict(
                model,
                np.ascontiguousarray(x_noise),
                y_mean,
                y_std,
                device,
            )

            mn = metrics(
                y[test_ids], p_noise
            )

            print(
                "[ABLATION] Corner noise sigma=0.005:",
                " ".join(
                    f"{k}: R2={v['r2']:.3f}"
                    for k, v in mn.items()
                ),
                flush=True,
            )

            for label, mm in [
                ("shuffled_history", ms),
                ("noise_sigma_0.005", mn),
            ]:
                for name, values in mm.items():
                    all_results.append({
                        "history": H,
                        "subset": label,
                        "n": len(test_ids),
                        "variable": name,
                        **values,
                    })

    with (
        output / "metrics.csv"
    ).open("w", newline="") as f:

        fieldnames = sorted(
            set().union(
                *(
                    set(row)
                    for row in all_results
                )
            )
        )

        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    print(
        "\n[DONE] Results:",
        output / "metrics.csv",
        flush=True,
    )


if __name__ == "__main__":
    main()
