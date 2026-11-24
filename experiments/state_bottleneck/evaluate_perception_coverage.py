import importlib.util
import math
from pathlib import Path

import torch


ROOT = Path("/home/sumukh/AerialCatching")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.set_num_threads(4)

SOURCES = {
    "student_existing": (
        ROOT / "experiments/state_bottleneck/data/"
        "pooled_student_epoch70_all_levels.pt"
    ),
    "teacher_level0": (
        ROOT / "experiments/state_bottleneck/data/"
        "perception_teacher_l0/round_100_shard_0000.pt"
    ),
    "mixed_level0": (
        ROOT / "experiments/state_bottleneck/data/"
        "perception_mixed_l0/round_101_shard_0000.pt"
    ),
}

AUX_SCRIPT = (
    ROOT / "experiments/corners_tcn_student_v1/"
    "train_aux13_student.py"
)

AUX_CKPT = (
    ROOT / "checkpoints/corners_tcn_student_v1/"
    "stage4_aux13_ft/stage4_aux13_ft_best.pt"
)

spec = importlib.util.spec_from_file_location(
    "aux13_perception", AUX_SCRIPT
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

checkpoint = torch.load(
    AUX_CKPT, map_location="cpu", weights_only=False
)

indices = list(checkpoint["auxiliary_target_indices"])

expected = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 19]

assert indices == expected, (
    f"Unexpected state ordering: {indices}"
)

policy = module.build_policy(DEVICE)
policy.student.load_state_dict(
    checkpoint["student_state_dict"], strict=True
)
policy.eval()

head = module.AuxiliaryStateHead().to(DEVICE)
head.load_state_dict(
    checkpoint["auxiliary_state_head_dict"], strict=True
)
head.eval()

mean = checkpoint["auxiliary_target_mean"].to(DEVICE)
std = checkpoint["auxiliary_target_std"].to(DEVICE)

GROUPS = {
    "position": slice(0, 3),
    "velocity": slice(3, 6),
    "gravity": slice(6, 9),
    "direction": slice(9, 12),
    "distance": slice(12, 13),
}


@torch.inference_mode()
def predict(observations):
    outputs = []

    for batch in observations.split(2048):
        x = batch.to(DEVICE)

        h = policy.encode_temporal_state(x)
        y = head(h) * std + mean

        outputs.append(y.cpu())

    return torch.cat(outputs)


def percentile(values, q):
    return torch.quantile(
        values.float(), q
    ).item()


def describe(name, target, prediction):
    if len(target) == 0:
        print(f"[{name}] empty")
        return

    print(f"\n[{name}] rows={len(target)}")

    for group, sl in GROUPS.items():
        error = prediction[:, sl] - target[:, sl]

        rmse = error.square().mean(0).sqrt()
        target_std = target[:, sl].std(0)

        abs_error = error.abs()

        print(
            f"  {group:>9s}: "
            f"RMSE={[round(x, 4) for x in rmse.tolist()]} "
            f"P90_abs={[round(x, 4) for x in torch.quantile(abs_error, 0.9, dim=0).tolist()]} "
            f"target_std={[round(x, 4) for x in target_std.tolist()]}"
        )


for source_name, path in SOURCES.items():
    print("\n" + "=" * 90)
    print("[SOURCE]", source_name)
    print("[PATH]", path)

    if not path.exists():
        print("[MISSING]")
        continue

    data = torch.load(
        path, map_location="cpu", weights_only=False
    )

    observations = data["observations"].float()
    teacher = data["teacher_observations"].float()
    steps = data["episode_steps"].long()

    assert observations.ndim == 2
    assert observations.shape[1] == 256
    assert teacher.shape == (len(observations), 20)

    target = teacher[:, indices]

    if source_name == "student_existing":
        level_mask = (
            data["curriculum_levels"].float().abs() < 1e-6
        )
    else:
        level_mask = torch.ones(
            len(observations), dtype=torch.bool
        )

    mature = steps >= 31
    early = ~mature

    if "controller_is_teacher" in data:
        is_teacher = data["controller_is_teacher"].bool()
    else:
        is_teacher = torch.zeros(
            len(observations), dtype=torch.bool
        )

    if "environment_ids" in data and "episode_ids" in data:
        pairs = torch.stack(
            [
                data["environment_ids"].long(),
                data["episode_ids"].long(),
            ],
            dim=1,
        )

        print(
            "[EPISODE GROUPS]",
            len(torch.unique(pairs, dim=0)),
        )

    print("[ROWS]", len(observations))
    print("[LEVEL0 ROWS]", int(level_mask.sum()))
    print("[MATURE LEVEL0]", int((level_mask & mature).sum()))
    print("[EARLY LEVEL0]", int((level_mask & early).sum()))
    print("[TEACHER FRACTION]", is_teacher.float().mean().item())

    # Describe the difficulty/coverage of the collected states.
    selected = target[level_mask]

    speed = selected[:, 3:6].norm(dim=1)

    gravity = selected[:, 6:9]

    tilt = torch.atan2(
        gravity[:, :2].norm(dim=1),
        (-gravity[:, 2]).clamp_min(1e-8),
    ) * (180.0 / math.pi)

    print(
        "[SPEED] median=",
        round(percentile(speed, 0.5), 4),
        "p90=",
        round(percentile(speed, 0.9), 4),
        "p99=",
        round(percentile(speed, 0.99), 4),
    )

    print(
        "[TILT DEGREES] median=",
        round(percentile(tilt, 0.5), 3),
        "p90=",
        round(percentile(tilt, 0.9), 3),
        "p99=",
        round(percentile(tilt, 0.99), 3),
    )

    prediction = predict(observations)

    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()

    describe(
        "LEVEL0 ALL",
        target[level_mask],
        prediction[level_mask],
    )

    describe(
        "LEVEL0 MATURE",
        target[level_mask & mature],
        prediction[level_mask & mature],
    )

    describe(
        "LEVEL0 EARLY",
        target[level_mask & early],
        prediction[level_mask & early],
    )

    if source_name == "mixed_level0":
        describe(
            "MATURE TEACHER-CONTROLLED",
            target[mature & is_teacher],
            prediction[mature & is_teacher],
        )

        describe(
            "MATURE STUDENT-CONTROLLED",
            target[mature & ~is_teacher],
            prediction[mature & ~is_teacher],
        )

    print("[DONE]", source_name, flush=True)
