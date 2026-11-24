import argparse
import importlib.util
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/home/sumukh/AerialCatching")

DATA_PATH = (
    ROOT / "experiments/state_bottleneck/data/"
    "pooled_student_epoch70_all_levels.pt"
)

AUX_SCRIPT = (
    ROOT / "experiments/corners_tcn_student_v1/"
    "train_aux13_student.py"
)

AUX_CKPT = (
    ROOT / "checkpoints/corners_tcn_student_v1/"
    "stage4_aux13_ft/stage4_aux13_ft_best.pt"
)

OUT_DIR = ROOT / "experiments/state_bottleneck/results"

# Velocity and projected gravity in the original teacher observation.
TEACHER_INDICES = [3, 4, 5, 9, 10, 11]
NAMES = ["vx", "vy", "vz", "gx", "gy", "gz"]


class Critical6Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 6),
        )

    def forward(self, latent):
        return self.network(latent)


def load_aux_module():
    spec = importlib.util.spec_from_file_location(
        "historical_aux13_training",
        AUX_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def encode_all(policy, observations, device):
    policy.eval()
    outputs = []

    for batch in observations.split(2048):
        outputs.append(
            policy.encode_temporal_state(
                batch.to(device)
            ).cpu()
        )

    return torch.cat(outputs)


@torch.no_grad()
def predict_all(
    policy, head, inputs, device, frozen
):
    policy.eval()
    head.eval()

    outputs = []

    for batch in inputs.split(2048):
        batch = batch.to(device)

        latent = (
            batch
            if frozen
            else policy.encode_temporal_state(batch)
        )

        outputs.append(head(latent).cpu())

    return torch.cat(outputs)


def report_metrics(label, prediction, target):
    errors = prediction - target

    rmse = errors.square().mean(0).sqrt()

    r2 = 1 - (
        errors.square().sum(0)
        /
        (target - target.mean(0))
        .square().sum(0).clamp_min(1e-8)
    )

    print(
        f"[{label}] samples={len(target)} "
        f"mean_R2={r2.mean().item():.5f}",
        flush=True,
    )

    for name, error, score in zip(
        NAMES, rmse.tolist(), r2.tolist()
    ):
        print(
            f"  {name:>3s} "
            f"RMSE={error:.5f} "
            f"R2={score:.5f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["frozen", "finetune"],
        required=True,
    )
    parser.add_argument(
        "--epochs", type=int, default=35
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024
    )
    args = parser.parse_args()

    torch.manual_seed(7500)
    random.seed(7500)
    torch.set_num_threads(4)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    module = load_aux_module()

    data = torch.load(
        DATA_PATH,
        map_location="cpu",
        weights_only=False,
    )

    observations = data["observations"].float()

    targets = data["teacher_observations"][
        :, TEACHER_INDICES
    ].float()

    episode_steps = data["episode_steps"].long()

    train_mask, val_mask, test_mask = (
        module.make_episode_split(
            data["environment_ids"].long(),
            data["episode_ids"].long(),
        )
    )

    print(
        "[DATA]",
        "total=", len(observations),
        "train=", int(train_mask.sum()),
        "val=", int(val_mask.sum()),
        "test=", int(test_mask.sum()),
        flush=True,
    )

    if observations.shape[1] != 256:
        raise RuntimeError("Unexpected corner-history shape")

    if targets.shape[1] != 6:
        raise RuntimeError("Unexpected state-target shape")

    # Training-only normalization, saved with the checkpoint.
    target_mean = targets[train_mask].mean(0)

    target_std = (
        targets[train_mask]
        .std(0)
        .clamp_min(0.05)
    )

    normalized = (
        targets - target_mean
    ) / target_std

    # Early history is important: deployment starts with
    # repeated frames and no mature temporal history.
    sample_weights = torch.where(
        episode_steps < 31,
        torch.tensor(2.0),
        torch.tensor(1.0),
    )

    aux_ckpt = torch.load(
        AUX_CKPT,
        map_location="cpu",
        weights_only=False,
    )

    policy = module.build_policy(device)

    policy.student.load_state_dict(
        aux_ckpt["student_state_dict"],
        strict=True,
    )

    policy.eval()

    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    frozen = args.mode == "frozen"

    if frozen:
        print("[INFO] Precomputing frozen TCN features")
        inputs = encode_all(
            policy, observations, device
        )

        if inputs.shape[1] != 128:
            raise RuntimeError("Unexpected latent dimension")

    else:
        inputs = observations

        # Fine-tune only this specialist's TCN.
        # The original AUX13 estimator remains untouched.
        for parameter in policy.tcn.parameters():
            parameter.requires_grad_(True)

    head = Critical6Head().to(device)

    groups = [
        {
            "params": head.parameters(),
            "lr": 6e-4,
        }
    ]

    if not frozen:
        groups.append(
            {
                "params": policy.tcn.parameters(),
                "lr": 5e-5,
            }
        )

    optimizer = torch.optim.AdamW(
        groups,
        weight_decay=1e-5,
    )

    loader = DataLoader(
        TensorDataset(
            inputs[train_mask],
            normalized[train_mask],
            sample_weights[train_mask],
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUT_DIR / f"critical6_{args.mode}.pt"
    )

    if output_path.exists():
        raise FileExistsError(
            f"Preserving existing checkpoint: {output_path}"
        )

    best_val = float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):

        head.train()

        if not frozen:
            policy.tcn.train()

        total_loss = 0.0
        total_samples = 0

        for xb, yb, weights in loader:

            xb = xb.to(device)
            yb = yb.to(device)
            weights = weights.to(device)

            latent = (
                xb
                if frozen
                else policy.encode_temporal_state(xb)
            )

            prediction = head(latent)

            sample_loss = (
                prediction - yb
            ).square().mean(dim=1)

            loss = (
                (sample_loss * weights).sum()
                / weights.sum()
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            trainable = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
            ]

            nn.utils.clip_grad_norm_(
                trainable, 1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(xb)
            total_samples += len(xb)

        prediction = predict_all(
            policy,
            head,
            inputs[val_mask],
            device,
            frozen,
        )

        val_error = (
            prediction - normalized[val_mask]
        ).square().mean(dim=1)

        val_weights = sample_weights[val_mask]

        val_loss = (
            (val_error * val_weights).sum()
            / val_weights.sum()
        ).item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            stale = 0

            torch.save(
                {
                    "mode": args.mode,
                    "epoch": epoch,
                    "validation_loss": best_val,
                    "teacher_indices": TEACHER_INDICES,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "head_state_dict": {
                        k: v.detach().cpu()
                        for k, v in head.state_dict().items()
                    },
                    "tcn_state_dict": {
                        k: v.detach().cpu()
                        for k, v in policy.tcn.state_dict().items()
                    },
                    "source_aux13_checkpoint": str(AUX_CKPT),
                },
                output_path,
            )
        else:
            stale += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[TRAIN] mode={args.mode} "
                f"epoch={epoch} "
                f"train={total_loss/total_samples:.6f} "
                f"val={val_loss:.6f} "
                f"best={best_val:.6f}",
                flush=True,
            )

        if stale >= 10:
            print("[EARLY STOP]", flush=True)
            break

    # Reload the best checkpoint for held-out evaluation.
    best_ckpt = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    head.load_state_dict(
        best_ckpt["head_state_dict"],
        strict=True,
    )

    policy.tcn.load_state_dict(
        best_ckpt["tcn_state_dict"],
        strict=True,
    )

    prediction = predict_all(
        policy,
        head,
        inputs[test_mask],
        device,
        frozen,
    )

    prediction = (
        prediction * target_std + target_mean
    )

    test_targets = targets[test_mask]
    test_steps = episode_steps[test_mask]

    print("\n========== HELD-OUT RESULTS ==========")

    report_metrics(
        "ALL",
        prediction,
        test_targets,
    )

    early = test_steps < 31
    mature = ~early

    if early.any():
        report_metrics(
            "EARLY",
            prediction[early],
            test_targets[early],
        )

    if mature.any():
        report_metrics(
            "MATURE",
            prediction[mature],
            test_targets[mature],
        )

    print("[SAVED]", output_path, flush=True)


if __name__ == "__main__":
    main()
