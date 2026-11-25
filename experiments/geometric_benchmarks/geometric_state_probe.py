"""Offline geometric observability test for ArUco corner histories."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "student": (
        ROOT / "datasets/state_estimation/"
        "pooled_student_epoch70_all_levels.pt"
    ),
    "teacher": (
        ROOT / "datasets/state_estimation/"
        "perception_teacher_l0/round_100_shard_0000.pt"
    ),
    "mixed": (
        ROOT / "datasets/state_estimation/"
        "perception_mixed_l0/round_101_shard_0000.pt"
    ),
}

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--samples", type=int, default=1500)
parser.add_argument("--noise-px", type=float, default=0.0)
parser.add_argument("--matched_student_test", action="store_true")

args = parser.parse_args()

# Exact marker corner ordering in the simulator:
# World offsets:
# [0,-h,-h], [0,+h,-h], [0,+h,+h], [0,-h,+h].
#
# Object X = world Y
# Object Y = -world Z
# Object Z = -world X
h = 0.05

OBJECT_POINTS = np.array([
    [-h, +h, 0.0],
    [+h, +h, 0.0],
    [+h, -h, 0.0],
    [-h, -h, 0.0],
], dtype=np.float64)

W, H = args.width, args.height
F = W * 24.0 / 20.955

K = np.array([
    [F, 0.0, W / 2.0],
    [0.0, F, H / 2.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# Body -> conventional OpenCV camera coordinates:
# camera X = -body Y
# camera Y = -body Z
# camera Z =  body X
R_CAM_BODY = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])

# Marker object -> world orientation.
R_WORLD_OBJ = np.array([
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

DT = 1.0 / 50.0

np.set_printoptions(precision=5, suppress=True)

NOISE_RNG = np.random.default_rng(314159)


def valid(corners):
    return bool(
        np.isfinite(corners).all()
        and (corners >= 0.0).all()
        and (corners <= 1.0).all()
    )


def fit_pose(corners, noise_px=0.0):
    """Return a geometric pose estimate without privileged state."""

    if not valid(corners):
        return None

    pixels = corners.astype(np.float64).copy()
    pixels[:, 0] *= W
    pixels[:, 1] *= H

    if noise_px > 0:
        pixels += NOISE_RNG.normal(
            loc=0.0,
            scale=noise_px,
            size=pixels.shape,
        )

    success, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS,
        pixels,
        K,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    R_CAM_OBJ = cv2.Rodrigues(rvec)[0]
    t_cam = tvec.reshape(3)

    if not np.isfinite(R_CAM_OBJ).all():
        return None

    if not np.isfinite(t_cam).all() or t_cam[2] <= 0:
        return None

    reprojected, _ = cv2.projectPoints(
        OBJECT_POINTS, rvec, tvec, K, None
    )

    reprojection_rmse = np.sqrt(
        np.mean(
            (reprojected.reshape(4, 2) - pixels) ** 2
        )
    )

    # World -> body orientation.
    R_BODY_WORLD = (
        R_CAM_BODY.T
        @ R_CAM_OBJ
        @ R_WORLD_OBJ.T
    )

    marker_b = R_CAM_BODY.T @ t_cam

    distance = np.linalg.norm(marker_b)

    if distance < 1e-8:
        return None

    direction_b = marker_b / distance

    gravity_b = (
        R_BODY_WORLD
        @ np.array([0.0, 0.0, -1.0])
    )

    # Physical marker is 1.47 m ahead of the hover target.
    hover_offset_w = np.array([1.47, 0.0, 0.0])

    hover_position_b = (
        marker_b - R_BODY_WORLD @ hover_offset_w
    )

    # Diagnostic: some teacher-observation code constructs
    # the direction using a nominal 1.50 m offset.
    nominal_marker_b = (
        marker_b
        + R_BODY_WORLD @ np.array([0.03, 0.0, 0.0])
    )
    nominal_direction_b = (
        nominal_marker_b / np.linalg.norm(nominal_marker_b)
    )

    # Marker-relative displacement expressed in world.
    # Its change across time gives vehicle translation
    # if the marker is stationary.
    marker_relative_w = (
        R_WORLD_OBJ @ R_CAM_OBJ.T @ t_cam
    )

    return {
        "position": hover_position_b,
        "hover_distance": float(np.linalg.norm(hover_position_b)),
        "nominal_direction": nominal_direction_b,
        "gravity": gravity_b,
        "direction": direction_b,
        "distance": distance,
        "rotation_bw": R_BODY_WORLD,
        "marker_relative_w": marker_relative_w,
        "reprojection": reprojection_rmse,
    }


def report(name, errors):
    if not errors:
        print(f"  {name}: no valid estimates")
        return

    e = np.asarray(errors, dtype=np.float64)

    rmse = np.sqrt(np.mean(e ** 2, axis=0))
    norm = np.linalg.norm(e.reshape(len(e), -1), axis=1)

    print(
        f"  {name:>13s}: "
        f"n={len(e):4d} "
        f"RMSE={np.round(rmse, 5)} "
        f"P90_error_norm={np.quantile(norm, 0.9):.5f}"
    )


def self_test():
    """Check the coordinate transformations on synthetic ideal data."""

    true_rotation = cv2.Rodrigues(
        np.array([0.06, -0.12, 0.15])
    )[0]

    true_marker_b = np.array([1.4, 0.1, 0.15])

    Rco = (
        R_CAM_BODY
        @ true_rotation
        @ R_WORLD_OBJ
    )

    tco = R_CAM_BODY @ true_marker_b

    projected, _ = cv2.projectPoints(
        OBJECT_POINTS,
        cv2.Rodrigues(Rco)[0],
        tco,
        K,
        None,
    )

    normalized = projected.reshape(4, 2).copy()
    normalized[:, 0] /= W
    normalized[:, 1] /= H

    estimate = fit_pose(normalized)

    if estimate is None:
        raise RuntimeError("Synthetic PnP reconstruction failed")

    true_gravity = (
        true_rotation @ np.array([0.0, 0.0, -1.0])
    )

    gravity_error = np.linalg.norm(
        estimate["gravity"] - true_gravity
    )

    position_error = abs(
        estimate["distance"]
        - np.linalg.norm(true_marker_b)
    )

    true_hover_position_b = (
        true_marker_b
        - true_rotation @ np.array([1.47, 0.0, 0.0])
    )

    hover_position_error = np.linalg.norm(
        estimate["position"] - true_hover_position_b
    )

    if (
        gravity_error > 1e-4
        or position_error > 1e-4
        or hover_position_error > 1e-4
    ):
        raise RuntimeError(
            f"Geometry self-test failed: "
            f"gravity={gravity_error}, "
            f"distance={position_error}"
        )

    print("[PASS] Synthetic geometric reconstruction")


def main():
    self_test()

    rng = np.random.default_rng(2026)

    print(f"[CAMERA] W={W} H={H} focal_px={F:.4f}")
    print(f"[MARKER] size=0.10m")
    print(f"[TEMPORAL] dt={DT:.4f}s")

    for name, path in SOURCES.items():
        print("\n" + "=" * 75)
        print("[SOURCE]", name)

        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        observations = (
            data["observations"].float().numpy()
            .reshape(-1, 32, 4, 2)
        )

        teacher = (
            data["teacher_observations"].float().numpy()
        )

        steps = data["episode_steps"].long().numpy()

        if name == "student":
            levels = (
                data["curriculum_levels"].float().numpy()
            )
            level0 = np.abs(levels) < 1e-6
        else:
            level0 = np.ones(len(steps), dtype=bool)

        current = observations[:, -1]

        visible = (
            np.isfinite(current).all(axis=(1, 2))
            & (current >= 0).all(axis=(1, 2))
            & (current <= 1).all(axis=(1, 2))
        )

        mature = steps >= 31

        print("[LEVEL0 ROWS]", int(level0.sum()))
        print(
            "[VISIBLE CURRENT]",
            int((visible & level0).sum()),
            "/",
            int(level0.sum()),
        )

        eligible_mask = level0 & mature & visible

        if args.matched_student_test:
            if name != "student":
                continue

            groups, inverse = torch.unique(
                torch.stack([
                    data["environment_ids"].long(),
                    data["episode_ids"].long(),
                ], dim=1),
                dim=0,
                return_inverse=True,
            )

            generator = torch.Generator().manual_seed(20261118)
            permutation = torch.randperm(
                len(groups), generator=generator
            )

            n_train = int(0.80 * len(groups))
            n_val = int(0.10 * len(groups))

            test_groups = torch.zeros(
                len(groups), dtype=torch.bool
            )
            test_groups[
                permutation[n_train + n_val:]
            ] = True

            previous = observations[:, -2]
            previous_visible = (
                np.isfinite(previous).all(axis=(1, 2))
                & (previous >= 0).all(axis=(1, 2))
                & (previous <= 1).all(axis=(1, 2))
            )

            eligible_mask &= (
                test_groups[inverse].numpy()
                & previous_visible
            )

        eligible = np.flatnonzero(eligible_mask)

        if args.matched_student_test:
            print("[MATCHED TEST ROWS]", len(eligible))
            if len(eligible) != 1252:
                raise RuntimeError(
                    f"Expected 1252 matched rows, got {len(eligible)}"
                )

        if len(eligible) == 0:
            print("[NO VISIBLE MATURE ROWS]")
            continue

        selected = rng.choice(
            eligible,
            size=min(args.samples, len(eligible)),
            replace=False,
        )

        errors = {
            "position": [],
            "gravity": [],
            "nominal_direction": [],
            "direction": [],
            "distance": [],
            "velocity_k1": [],
            "velocity_k5": [],
            "velocity_k10": [],
        }

        reproject = []

        for idx in selected:
            history = observations[idx]

            pose = fit_pose(history[-1], args.noise_px)

            if pose is None:
                continue

            reproject.append(pose["reprojection"])

            errors["position"].append(
                pose["position"] - teacher[idx, 0:3]
            )

            errors["nominal_direction"].append(
                pose["nominal_direction"] - teacher[idx, 12:15]
            )

            errors["gravity"].append(
                pose["gravity"] - teacher[idx, 9:12]
            )

            errors["direction"].append(
                pose["direction"] - teacher[idx, 12:15]
            )

            errors["distance"].append(
                np.array([
                    pose["hover_distance"] - teacher[idx, 19]
                ])
            )

            # Estimate body-frame velocity from changes in
            # marker-relative world position.
            for k in (1, 5, 10):
                previous = fit_pose(history[-1-k], args.noise_px)

                if previous is None:
                    continue

                velocity_world = (
                    previous["marker_relative_w"]
                    - pose["marker_relative_w"]
                ) / (k * DT)

                velocity_body = (
                    pose["rotation_bw"]
                    @ velocity_world
                )

                errors[f"velocity_k{k}"].append(
                    velocity_body - teacher[idx, 3:6]
                )

        print("[POSE SOLUTIONS]", len(reproject))

        if reproject:
            print(
                "[REPROJECTION PX]",
                "median=",
                round(np.median(reproject), 5),
                "p90=",
                round(np.quantile(reproject, 0.9), 5),
            )

        for key, value in errors.items():
            report(key, value)

        print("[DONE]", name, flush=True)


if __name__ == "__main__":
    main()
