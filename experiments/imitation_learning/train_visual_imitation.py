#!/usr/bin/env python3
"""Matched, corners-only supervised action learning on two GPUs."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


from controllers.visual_policy import Actor, DeployPolicy

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/runs"
HIST = REPO / "results/historical_experiments/state_bottleneck"
DATA = REPO / "datasets/state_estimation/pooled_student_epoch70_all_levels.pt"
PROTOCOL = HIST / (
    "state_reconstruction_20261119T195844Z/protocol.json"
)
SOURCE = REPO / "rl/pretrain_state_estimator.py"
PRETRAINED = REPO / "checkpoints/reference/representations/multiscale_plain_huber.pt"
SEED = 20261119


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module






def load_dataset():
    data = torch.load(
        DATA,
        map_location="cpu",
        weights_only=False,
    )

    def arr(key):
        value = data[key]
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    x = arr("observations").reshape(-1, 256)
    privileged = arr("teacher_observations")
    actions = arr("teacher_actions")

    valid = (
        (arr("episode_steps") >= 31)
        & np.isfinite(x).all(axis=1)
        & np.isfinite(privileged).all(axis=1)
    )

    assert np.isfinite(actions[valid]).all(), (
        "Non-finite teacher actions"
    )

    assert np.max(np.abs(actions[valid])) <= 1.0001, (
        "Teacher actions are not normalized to [-1,1]"
    )

    protocol = json.loads(PROTOCOL.read_text())

    probe = load_module(
        "observability_probe_for_actor",
        REPO / "experiments/representation_learning/observability_probe.py",
    )

    split, _, _ = probe.build_masks(
        arr("curriculum_levels")[valid],
        arr("environment_ids")[valid],
        arr("episode_ids")[valid],
        np.random.default_rng(
            int(protocol["seed"])
        ),
    )

    sizes = [
        int((split == k).sum())
        for k in (0, 1, 2)
    ]

    assert sizes == [57377, 12146, 10764], (
        f"Unexpected dataset split: {sizes}"
    )

    state = privileged[valid][
        :, list(range(15)) + [19]
    ]

    return (
        torch.from_numpy(
            x[valid].astype(np.float32)
        ),
        torch.from_numpy(
            state.astype(np.float32)
        ),
        torch.from_numpy(
            actions[valid].astype(np.float32)
        ),
        split,
        arr("newest_visible_corner_count")[valid],
    )


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    predictions = []

    for batch in x.split(2048):
        actions, _ = model(batch.to(device))
        predictions.append(actions.cpu())

    return torch.cat(predictions).numpy()


def evaluate(model, x, targets, states, visible, device):
    predictions = predict(model, x, device)

    targets = targets.numpy()
    states = states.numpy()

    near_hover = (
        (np.linalg.norm(states[:, :3], axis=1) < 0.2)
        & (np.linalg.norm(states[:, 3:6], axis=1) < 0.3)
        & (np.linalg.norm(states[:, 6:9], axis=1) < 1.0)
    )

    groups = {
        "all": np.ones(
            len(targets), dtype=bool
        ),
        "four_corners": visible == 4,
        "all_corners_missing": visible == 0,
        "near_hover": near_hover,
        "high_angular_speed": (
            np.linalg.norm(
                states[:, 6:9], axis=1
            ) >= 3.0
        ),
    }

    results = {}

    for name, mask in groups.items():
        if not mask.any():
            continue

        error = predictions[mask] - targets[mask]

        denominator = np.sum(
            (
                targets[mask]
                - targets[mask].mean(axis=0)
            ) ** 2
        )

        results[name] = {
            "n": int(mask.sum()),
            "rmse": float(
                np.sqrt(np.mean(error ** 2))
            ),
            "mae": float(
                np.mean(np.abs(error))
            ),
            "r2": float(
                1.0
                - np.sum(error ** 2)
                / max(float(denominator), 1e-12)
            ),
            "p95_action_l2_error": float(
                np.quantile(
                    np.linalg.norm(error, axis=1),
                    0.95,
                )
            ),
            "per_channel_rmse": (
                np.sqrt(
                    np.mean(error ** 2, axis=0)
                ).tolist()
            ),
        }

    return results


def fit(args):
    global PRETRAINED, PROTOCOL
    if args.pretrained:
        PRETRAINED = Path(args.pretrained).resolve(strict=True)
    if args.protocol:
        PROTOCOL = Path(args.protocol).resolve(strict=True)
    assert torch.cuda.is_available(), (
        "CUDA is unavailable"
    )

    torch.set_num_threads(4)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda:0")

    module = load_module(
        "multiscale_transformer_for_actor",
        SOURCE,
    )

    model = Actor(
        module.TemporalEstimator
    )

    checkpoint = torch.load(
        PRETRAINED,
        map_location="cpu",
        weights_only=False,
    )

    if args.variant != "direct":
        model.estimator.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        # Verify our feature extraction reproduces
        # the pretrained estimator's predictions.
        model.eval()

        with torch.no_grad():
            sample = torch.rand(4, 32, 8)

            _, reconstructed = model(
                sample.reshape(4, 256)
            )

            reference = model.estimator(sample)

            error = (
                reconstructed - reference
            ).abs().max().item()

            assert error < 1e-5, (
                f"Pretrained-state mismatch: {error}"
            )

    x, state, action, split, visibility = (
        load_dataset()
    )

    train_mask = split == 0
    val_mask = split == 1
    test_mask = split == 2

    x_train = x[train_mask].to(device)
    y_train = state[train_mask].to(device)
    a_train = action[train_mask].to(device)

    x_val = x[val_mask].to(device)
    y_val = state[val_mask].to(device)
    a_val = action[val_mask].to(device)

    target_mean = torch.as_tensor(
        checkpoint["target_mean"],
        dtype=torch.float32,
        device=device,
    )

    target_std = torch.as_tensor(
        checkpoint["target_std"],
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-6)

    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.estimator.parameters()
                ),
                "lr": 2e-4,
            },
            {
                "params": model.policy.parameters(),
                "lr": 3e-4,
            },
        ],
        weight_decay=1e-4,
    )

    aux_weight = (
        0.10 if args.variant == "joint" else 0.0
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)

    config = {
        "variant": args.variant,
        "seed": SEED,
        "training_samples": int(train_mask.sum()),
        "validation_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "auxiliary_loss_weight": aux_weight,
        "input": "32x8 ArUco corners",
        "output": "4 normalized CTBR actions",
        "gpu": torch.cuda.get_device_name(0),
        "dataset": str(DATA),
        "pretraining_checkpoint": str(PRETRAINED),
        "split_protocol": str(PROTOCOL),
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2)
    )

    best_val = float("inf")
    history = []

    print(
        f"[START] {args.variant} "
        f"train={len(x_train)} "
        f"val={len(x_val)} "
        f"test={int(test_mask.sum())}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED + epoch
        )

        permutation = torch.randperm(
            len(x_train),
            generator=generator,
            device=device,
        )

        total_action = 0.0
        total_state = 0.0
        total_samples = 0

        for ids in permutation.split(args.batch):
            predicted_action, predicted_state = (
                model(x_train[ids])
            )

            action_loss = F.mse_loss(
                predicted_action,
                a_train[ids],
            )

            normalized_state = (
                y_train[ids] - target_mean
            ) / target_std

            state_loss = F.smooth_l1_loss(
                predicted_state,
                normalized_state,
            )

            loss = (
                action_loss
                + aux_weight * state_loss
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            count = len(ids)

            total_action += (
                action_loss.detach().item() * count
            )

            total_state += (
                state_loss.detach().item() * count
            )

            total_samples += count

        model.eval()

        val_action_sum = 0.0
        val_state_sum = 0.0

        with torch.inference_mode():
            for xx, aa, yy in zip(
                x_val.split(args.batch),
                a_val.split(args.batch),
                y_val.split(args.batch),
            ):
                predicted_action, predicted_state = (
                    model(xx)
                )

                val_action_sum += F.mse_loss(
                    predicted_action,
                    aa,
                    reduction="sum",
                ).item()

                val_state_sum += F.smooth_l1_loss(
                    predicted_state,
                    (yy - target_mean) / target_std,
                    reduction="sum",
                ).item()

        val_action_mse = (
            val_action_sum / (len(x_val) * 4)
        )

        val_state_huber = (
            val_state_sum / (len(x_val) * 16)
        )

        row = {
            "epoch": epoch,
            "train_action_mse": (
                total_action / total_samples
            ),
            "train_state_huber": (
                total_state / total_samples
            ),
            "val_action_mse": val_action_mse,
            "val_state_huber": val_state_huber,
        }

        history.append(row)

        # All three policies select checkpoints using
        # validation ACTION loss, never test loss.
        if val_action_mse < best_val:
            best_val = val_action_mse

            torch.save(
                {
                    "actor_state_dict": {
                        key: value.detach().cpu()
                        for key, value
                        in model.state_dict().items()
                    },
                    "variant": args.variant,
                    "epoch": epoch,
                    "val_action_mse": best_val,
                    "auxiliary_weight": aux_weight,
                    "state_target_mean": (
                        target_mean.cpu()
                    ),
                    "state_target_std": (
                        target_std.cpu()
                    ),
                    "input_shape": [32, 8],
                    "action_dim": 4,
                },
                out / "policy.pt",
            )

        if (
            epoch <= 2
            or epoch % 5 == 0
            or epoch == args.epochs
        ):
            print(
                f"[{args.variant}] "
                f"epoch={epoch:02d}/{args.epochs} "
                f"train_action={row['train_action_mse']:.6f} "
                f"val_action={val_action_mse:.6f} "
                f"best={best_val:.6f}",
                flush=True,
            )

    with (out / "history.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0]),
        )

        writer.writeheader()
        writer.writerows(history)

    best_checkpoint = torch.load(
        out / "policy.pt",
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint["actor_state_dict"]
    )

    model.to(device)

    metrics = evaluate(
        model,
        x[test_mask],
        action[test_mask],
        state[test_mask],
        visibility[test_mask],
        device,
    )

    (out / "offline_test.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(
        "[TEST]",
        args.variant,
        json.dumps(metrics),
        flush=True,
    )

    # Export a simple deployment interface:
    # policy(corners256) -> actions4.
    deployment_policy = DeployPolicy(
        model.cpu().eval()
    )

    try:
        traced = torch.jit.trace(
            deployment_policy,
            torch.zeros(2, 256),
            check_trace=False,
        )

        with torch.inference_mode():
            check_input = torch.rand(7, 256)

            difference = (
                traced(check_input)
                - deployment_policy(check_input)
            ).abs().max().item()

        assert difference < 1e-4, (
            f"TorchScript export mismatch: {difference}"
        )

        traced.save(
            str(out / "policy.ts")
        )

        print(
            "[EXPORTED]",
            out / "policy.ts",
            flush=True,
        )

    except Exception as exc:
        print(
            "[WARN] TorchScript export failed:",
            repr(exc),
            flush=True,
        )

    print(
        f"[DONE] {args.variant} "
        f"best_epoch={best_checkpoint['epoch']} "
        f"best_val_action_mse={best_val:.6f}",
        flush=True,
    )


def suite(args):
    assert torch.cuda.device_count() >= 2, (
        "At least two visible CUDA GPUs are required"
    )

    run = ROOT / (
        "action_transformer_suite_"
        + time.strftime(
            "%Y%m%dT%H%M%SZ",
            time.gmtime(),
        )
    )

    run.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        "[SUITE]",
        run,
        "GPUs:",
        torch.cuda.device_count(),
        flush=True,
    )

    def begin(variant, gpu):
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
        )

        log = (
            run / f"{variant}.log"
        ).open("w")

        command = [
            sys.executable,
            "-u",
            __file__,
            "--variant",
            variant,
            "--out",
            str(run / variant),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
        ]

        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        return process, log

    process_a, log_a = begin("direct", 0)
    process_b, log_b = begin("pretrained", 1)

    print(
        "[GPU 0] Direct imitation",
        flush=True,
    )

    print(
        "[GPU 1] State-pretrained imitation",
        flush=True,
    )

    result_a = process_a.wait()
    log_a.close()

    print(
        "[GPU 0] Direct completed:",
        result_a,
        flush=True,
    )

    if result_a == 0:
        process_c, log_c = begin("joint", 0)

        print(
            "[GPU 0] Joint state/action supervision",
            flush=True,
        )
    else:
        process_c, log_c = None, None

    result_b = process_b.wait()
    log_b.close()

    if process_c is not None:
        result_c = process_c.wait()
        log_c.close()
    else:
        result_c = -1

    statuses = {
        "direct": result_a,
        "pretrained": result_b,
        "joint": result_c,
    }

    print(
        "\n========== TRAINING STATUS ==========",
        flush=True,
    )

    print(statuses, flush=True)

    print(
        "\n========== OFFLINE COMPARISON ==========",
        flush=True,
    )

    for variant in (
        "direct",
        "pretrained",
        "joint",
    ):
        metrics_path = (
            run / variant / "offline_test.json"
        )

        if not metrics_path.is_file():
            print(
                variant,
                "NO METRICS",
                flush=True,
            )
            continue

        metrics = json.loads(
            metrics_path.read_text()
        )

        overall = metrics["all"]

        near = metrics.get(
            "near_hover", {}
        )

        missing = metrics.get(
            "all_corners_missing", {}
        )

        print(
            f"{variant:12s} "
            f"overall_RMSE={overall['rmse']:.5f} "
            f"near_hover_RMSE={near.get('rmse', float('nan')):.5f} "
            f"missing_RMSE={missing.get('rmse', float('nan')):.5f}",
            flush=True,
        )

        print(
            "  Checkpoint:",
            run / variant / "policy.pt",
            flush=True,
        )

        print(
            "  Deployable policy:",
            run / variant / "policy.ts",
            flush=True,
        )

    print(
        "\n========== EXISTING EVALUATOR INTERFACE ==========",
        flush=True,
    )

    evaluator = REPO / "evaluation/evaluate_state_bottleneck.py"

    if evaluator.is_file():
        lines = evaluator.read_text().splitlines()

        for i, line in enumerate(lines):
            if any(
                token in line
                for token in (
                    "add_argument(",
                    "student_checkpoint",
                    "student_model",
                    "load_state_dict",
                    "torch.jit.load",
                )
            ):
                print(
                    f"{i+1}: {line.strip()}",
                    flush=True,
                )

    print(
        "\n[NOTE] These are offline imitation metrics, "
        "not closed-loop hovering results.",
        flush=True,
    )

    if any(code != 0 for code in statuses.values()):
        raise RuntimeError(
            f"Training failed for at least one variant. "
            f"Check logs in {run}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--suite",
        action="store_true",
    )

    parser.add_argument(
        "--variant",
        choices=(
            "direct",
            "pretrained",
            "joint",
        ),
    )

    parser.add_argument("--out")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--protocol", default=None)

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1024,
    )

    args = parser.parse_args()

    if args.suite:
        suite(args)
    else:
        if not args.variant or not args.out:
            parser.error(
                "--variant and --out are required"
            )

        fit(args)
