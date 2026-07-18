"""Train a fresh 256-D corners-only DAgger student."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_vision_student_teacher_class():
    """Load the policy definition without registering Isaac Lab tasks.

    Importing ``envs.aruco_hover.agents`` normally executes ``envs/__init__.py``
    and the Gym environment-registration package. Offline supervised training
    requires only the pure-PyTorch policy definition.
    """

    module_path = (
        Path(__file__).resolve().parents[1]
        / "envs"
        / "aruco_hover"
        / "agents"
        / "vision_student_teacher.py"
    )

    if not module_path.is_file():
        raise FileNotFoundError(
            f"Policy module does not exist: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "_corners_vision_student_teacher",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not construct an import specification for {module_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return module.VisionStudentTeacher
    except AttributeError as error:
        raise ImportError(
            f"VisionStudentTeacher is missing from {module_path}"
        ) from error


VisionStudentTeacher = _load_vision_student_teacher_class()

ACTION_NAMES = ("thrust", "roll", "pitch", "yaw")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_shard(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a dictionary")

    required = {
        "observations",
        "teacher_actions",
        "schema_version",
        "sample_count",
        "history_length",
        "token_dim",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"{path}: missing keys {sorted(missing)}")

    observations = data["observations"]
    actions = data["teacher_actions"]
    rows = int(data["sample_count"])

    if observations.shape != (rows, 256):
        raise ValueError(f"{path}: observations have shape {tuple(observations.shape)}")
    if actions.shape != (rows, 4):
        raise ValueError(f"{path}: actions have shape {tuple(actions.shape)}")
    if observations.dtype != torch.float32 or actions.dtype != torch.float32:
        raise TypeError(f"{path}: observations and actions must be float32")
    if data["schema_version"] != "corners_dagger_v1":
        raise ValueError(f"{path}: unexpected schema {data['schema_version']!r}")
    if int(data["history_length"]) != 32 or int(data["token_dim"]) != 8:
        raise ValueError(f"{path}: expected a 32 x 8 history")
    if not bool(torch.isfinite(observations).all()):
        raise ValueError(f"{path}: non-finite observations")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError(f"{path}: non-finite actions")
    if not bool(
        (
            observations.eq(-1.0)
            | (observations.ge(0.0) & observations.le(1.0))
        ).all()
    ):
        raise ValueError(f"{path}: corner values outside {{-1}} union [0, 1]")
    if float(actions.min()) < -1.00001 or float(actions.max()) > 1.00001:
        raise ValueError(f"{path}: actions outside [-1, 1]")

    return observations, actions, data


def build_model(device: str) -> VisionStudentTeacher:
    return VisionStudentTeacher(
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


def cosine_lr(epoch: int, epochs: int, maximum: float, minimum: float) -> float:
    if epochs == 1:
        return maximum
    progress = epoch / (epochs - 1)
    return minimum + 0.5 * (maximum - minimum) * (
        1.0 + math.cos(math.pi * progress)
    )


def set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_checkpoint", type=Path, required=True)
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--validation_batch_size", type=int, default=2048)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--min_learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--max_samples_per_shard", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch_size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation_fraction must be in (0, 1)")
    if args.manifest is not None and not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    last_checkpoint = args.output_checkpoint.with_name(
        f"{args.output_checkpoint.stem}_last{args.output_checkpoint.suffix}"
    )
    existing = [
        path
        for path in (args.output_checkpoint, last_checkpoint, args.metrics_csv)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing files: {existing}")

    shard_paths = sorted(args.data_dir.rglob("round_*_shard_*.pt"))
    if not shard_paths:
        raise RuntimeError(f"No shards found recursively under {args.data_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    splits = []
    shard_records = []
    for index, path in enumerate(shard_paths):
        observations, _, metadata = load_shard(path)
        rows = observations.shape[0]
        if args.max_samples_per_shard is not None:
            rows = min(rows, args.max_samples_per_shard)
        validation_rows = max(1, round(rows * args.validation_fraction))
        validation_rows = min(validation_rows, rows - 1)
        generator = torch.Generator().manual_seed(args.seed + 1009 * index)
        permutation = torch.randperm(rows, generator=generator)
        validation_indices = permutation[:validation_rows]
        train_indices = permutation[validation_rows:]
        splits.append((path, rows, train_indices, validation_indices))
        shard_records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows_used": rows,
                "train_rows": int(train_indices.numel()),
                "validation_rows": int(validation_indices.numel()),
                "round_index": metadata.get("round_index"),
                "curriculum_level": metadata.get("curriculum_level"),
                "teacher_checkpoint_sha256": metadata.get(
                    "teacher_checkpoint_sha256"
                ),
            }
        )
        print(
            f"[DATA] {path}: rows={rows}, train={train_indices.numel()}, "
            f"validation={validation_indices.numel()}"
        )
        del observations

    model = build_model(args.device)
    optimizer = torch.optim.Adam(
        model.student.parameters(),
        lr=args.learning_rate,
    )

    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_csv.open("w", encoding="utf-8", newline="")
    writer = csv.writer(metrics_file)
    writer.writerow(
        [
            "epoch",
            "learning_rate",
            "train_mse",
            *(f"train_{name}_mse" for name in ACTION_NAMES),
            "validation_mse",
            *(f"validation_{name}_mse" for name in ACTION_NAMES),
            "is_best",
        ]
    )

    best_validation_mse = float("inf")

    for epoch_index in range(args.epochs):
        epoch = epoch_index + 1
        learning_rate = cosine_lr(
            epoch_index,
            args.epochs,
            args.learning_rate,
            args.min_learning_rate,
        )
        set_lr(optimizer, learning_rate)

        model.student.train()
        train_error_sum = torch.zeros(4, dtype=torch.float64)
        train_rows = 0
        shard_order = list(range(len(splits)))
        random.Random(args.seed + epoch).shuffle(shard_order)

        for split_index in shard_order:
            path, rows, train_indices, _ = splits[split_index]
            observations, actions, _ = load_shard(path)
            observations = observations[:rows]
            actions = actions[:rows]
            generator = torch.Generator().manual_seed(
                args.seed + 100_003 * epoch + split_index
            )
            order = torch.randperm(train_indices.numel(), generator=generator)
            indices_for_epoch = train_indices[order]

            for start in range(0, indices_for_epoch.numel(), args.batch_size):
                indices = indices_for_epoch[start : start + args.batch_size]
                obs_batch = observations.index_select(0, indices).to(
                    args.device, non_blocking=True
                )
                target_batch = actions.index_select(0, indices).to(
                    args.device, non_blocking=True
                )
                prediction = model.act_inference(obs_batch)
                loss = F.mse_loss(prediction, target_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.student.parameters(), args.gradient_clip_norm
                )
                optimizer.step()
                train_error_sum += (
                    (prediction - target_batch)
                    .pow(2)
                    .sum(dim=0)
                    .detach()
                    .cpu()
                    .double()
                )
                train_rows += indices.numel()
            del observations, actions

        model.student.eval()
        validation_error_sum = torch.zeros(4, dtype=torch.float64)
        validation_rows = 0
        with torch.inference_mode():
            for path, rows, _, validation_indices in splits:
                observations, actions, _ = load_shard(path)
                observations = observations[:rows]
                actions = actions[:rows]
                for start in range(
                    0,
                    validation_indices.numel(),
                    args.validation_batch_size,
                ):
                    indices = validation_indices[
                        start : start + args.validation_batch_size
                    ]
                    obs_batch = observations.index_select(0, indices).to(
                        args.device, non_blocking=True
                    )
                    target_batch = actions.index_select(0, indices).to(
                        args.device, non_blocking=True
                    )
                    prediction = model.act_inference(obs_batch)
                    validation_error_sum += (
                        (prediction - target_batch)
                        .pow(2)
                        .sum(dim=0)
                        .cpu()
                        .double()
                    )
                    validation_rows += indices.numel()
                del observations, actions

        train_action_mse = (train_error_sum / train_rows).tolist()
        validation_action_mse = (
            validation_error_sum / validation_rows
        ).tolist()
        train_mse = sum(train_action_mse) / 4.0
        validation_mse = sum(validation_action_mse) / 4.0
        is_best = validation_mse < best_validation_mse
        if is_best:
            best_validation_mse = validation_mse

        checkpoint = {
            "student_state_dict": model.student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_mse": best_validation_mse,
            "history_length": 32,
            "token_dim": 8,
            "temporal_channels": 128,
            "student_hidden_dims": [256, 256],
            "num_student_obs": 256,
            "num_teacher_obs": 20,
            "num_actions": 4,
            "schema_version": "corners_dagger_v1",
            "training_seed": args.seed,
            "validation_fraction": args.validation_fraction,
            "git_commit": git_commit(),
            "data_dir": str(args.data_dir.resolve()),
            "manifest": str(args.manifest.resolve()) if args.manifest else None,
            "manifest_sha256": (
                sha256_file(args.manifest) if args.manifest else None
            ),
            "shards": shard_records,
        }

        args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, last_checkpoint)
        if is_best:
            torch.save(checkpoint, args.output_checkpoint)

        writer.writerow(
            [
                epoch,
                learning_rate,
                train_mse,
                *train_action_mse,
                validation_mse,
                *validation_action_mse,
                int(is_best),
            ]
        )
        metrics_file.flush()
        print(
            f"Epoch {epoch:03d}: train={train_mse:.8f}, "
            f"validation={validation_mse:.8f}, "
            f"validation_actions={validation_action_mse}, best={is_best}"
        )

    metrics_file.close()
    summary_path = args.metrics_csv.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "best_validation_mse": best_validation_mse,
                "best_checkpoint": str(args.output_checkpoint.resolve()),
                "best_checkpoint_sha256": sha256_file(args.output_checkpoint),
                "last_checkpoint": str(last_checkpoint.resolve()),
                "last_checkpoint_sha256": sha256_file(last_checkpoint),
                "metrics_csv": str(args.metrics_csv.resolve()),
                "metrics_csv_sha256": sha256_file(args.metrics_csv),
                "git_commit": git_commit(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[PASS] Fresh DAgger training complete")
    print(f"best_validation_mse={best_validation_mse:.10f}")
    print(f"best_checkpoint={args.output_checkpoint}")


if __name__ == "__main__":
    main()
