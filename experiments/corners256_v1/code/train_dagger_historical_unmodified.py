import argparse
import glob
import os
import random
import sys

from isaaclab.app import AppLauncher


# -------------------------------------------------------------------------
# Launch Isaac Sim before importing anything through the envs package.
# -------------------------------------------------------------------------
launcher_parser = argparse.ArgumentParser(
    add_help=False,
)

AppLauncher.add_app_launcher_args(
    launcher_parser
)

launcher_args, remaining_args = (
    launcher_parser.parse_known_args()
)

launcher_args.headless = True
launcher_args.enable_cameras = False

app_launcher = AppLauncher(
    launcher_args
)

simulation_app = app_launcher.app

# Preserve only the arguments belonging to train_dagger.py.
sys.argv = [sys.argv[0]] + remaining_args


# Imports below this point may transitively import Isaac Lab / carb.
import torch
import torch.nn.functional as F

from envs.aruco_hover.agents.vision_student_teacher import (
    VisionStudentTeacher,
)


def get_learning_rate(epoch: int) -> float:
    if epoch >= 50:
        return 1.0e-5

    progress = epoch / 50.0

    return (
        1.0e-3
        + progress * (1.0e-5 - 1.0e-3)
    )


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()

    shard_paths = sorted(
        glob.glob(
            os.path.join(
                args.data_dir,
                "round_*_shard_*.pt",
            )
        )
    )

    if not shard_paths:
        raise RuntimeError(
            f"No shards found in {args.data_dir}"
        )

    model = VisionStudentTeacher(
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
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.student.parameters(),
        lr=1.0e-3,
    )

    start_epoch = 0

    if args.resume_checkpoint is not None:
        checkpoint = torch.load(
            args.resume_checkpoint,
            map_location=args.device,
        )

        model.student.load_state_dict(
            checkpoint["student_state_dict"],
            strict=True,
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        start_epoch = int(
            checkpoint.get("epoch", 0)
        )

    model.train()

    for local_epoch in range(args.epochs):
        epoch = start_epoch + local_epoch

        learning_rate = get_learning_rate(epoch)

        set_learning_rate(
            optimizer,
            learning_rate,
        )

        random.shuffle(shard_paths)

        accumulated_loss = 0.0
        number_of_batches = 0

        for shard_path in shard_paths:
            shard = torch.load(
                shard_path,
                map_location="cpu",
            )

            observations = (
                shard["observations"]
                .to(torch.float32)
            )

            teacher_actions = (
                shard["teacher_actions"]
                .to(torch.float32)
            )

            permutation = torch.randperm(
                observations.shape[0]
            )

            for start in range(
                0,
                observations.shape[0],
                args.batch_size,
            ):
                indices = permutation[
                    start : start + args.batch_size
                ]

                if indices.numel() < args.batch_size:
                    continue

                obs_batch = observations[
                    indices
                ].to(
                    args.device,
                    non_blocking=True,
                )

                target_batch = teacher_actions[
                    indices
                ].to(
                    args.device,
                    non_blocking=True,
                )

                predicted_actions = model.act_inference(
                    obs_batch
                )

                loss = F.mse_loss(
                    predicted_actions,
                    target_batch,
                )

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.student.parameters(),
                    1.0,
                )

                optimizer.step()

                accumulated_loss += loss.item()
                number_of_batches += 1

        mean_loss = (
            accumulated_loss
            / max(number_of_batches, 1)
        )

        print(
            f"Epoch {epoch + 1}: "
            f"loss={mean_loss:.6f}, "
            f"lr={learning_rate:.8f}, "
            f"shards={len(shard_paths)}"
        )

        output_directory = os.path.dirname(
            args.output_checkpoint
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        torch.save(
            {
                "student_state_dict":
                    model.student.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch": epoch + 1,

                "history_length": 32,
                "token_dim": 8,
                "temporal_channels": 128,
            },
            args.output_checkpoint,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()