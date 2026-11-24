from pathlib import Path
import importlib.util
import json

import cv2
import numpy as np
import torch


ROOT = Path("experiments/state_bottleneck")

DATA = ROOT / "data/pooled_student_epoch70_all_levels.pt"

PROTOCOL = ROOT / (
    "state_reconstruction_20261119T195844Z/protocol.json"
)

TRANSFORMER_SOURCE = ROOT / "supervised_multiscale_transformer.py"

TRANSFORMER_CKPT = ROOT / (
    "multiscale_plain_huber_20261119T205226Z/"
    "multiscale_transformer_H32.pt"
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_module(
    "observability_probe",
    ROOT / "supervised_observability_probe.py",
)

transformer_module = load_module(
    "supervised_transformer",
    TRANSFORMER_SOURCE,
)


# --------------------------------------------------
# Camera and marker geometry
# --------------------------------------------------

f = 24.0 / 20.955

K = np.array([
    [f, 0.0, 0.5],
    [0.0, f, 0.5],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# Body coordinates -> OpenCV camera coordinates:
# Xcv = -Ybody
# Ycv = -Zbody
# Zcv =  Xbody

C = np.array([
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
    [1.0,  0.0,  0.0],
], dtype=np.float64)

half = 0.05

OBJECT_POINTS = np.array([
    [0.0, -half, -half],
    [0.0,  half, -half],
    [0.0,  half,  half],
    [0.0, -half,  half],
], dtype=np.float64)

HOVER_OFFSET = np.array(
    [-1.5, 0.0, 0.0], dtype=np.float64
)

GRAVITY_WORLD = np.array(
    [0.0, 0.0, -1.0], dtype=np.float64
)


def estimate_pose(corners):
    """Estimate body-frame state using only four corners."""

    corners = np.asarray(corners, dtype=np.float64)

    result = cv2.solvePnPGeneric(
        OBJECT_POINTS,
        corners,
        K,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not result[0]:
        return None

    best = None

    for rvec, tvec in zip(result[1], result[2]):
        R_cv_world = cv2.Rodrigues(rvec)[0]
        t_cv = tvec.reshape(3)

        camera_points = (
            R_cv_world @ OBJECT_POINTS.T
        ).T + t_cv

        if (
            not np.isfinite(camera_points).all()
            or np.any(camera_points[:, 2] <= 0.05)
        ):
            continue

        projected, _ = cv2.projectPoints(
            OBJECT_POINTS,
            rvec,
            tvec,
            K,
            None,
        )

        error = float(np.mean(
            np.linalg.norm(
                projected.reshape(4, 2) - corners,
                axis=1,
            )
        ))

        if best is None or error < best[0]:
            best = (error, R_cv_world, t_cv)

    if best is None:
        return None

    reprojection_error, R_cv_world, t_cv = best

    # World -> body rotation.
    R_body_world = C.T @ R_cv_world

    # Drone position relative to marker center,
    # expressed in marker-aligned world coordinates.
    position_world = -R_cv_world.T @ t_cv

    position_error_body = R_body_world @ (
        HOVER_OFFSET - position_world
    )

    gravity_body = R_body_world @ GRAVITY_WORLD

    marker_direction_body = (
        C.T @ t_cv / max(np.linalg.norm(t_cv), 1e-8)
    )

    hover_distance = np.linalg.norm(
        position_error_body
    )

    prediction = np.concatenate([
        position_error_body,
        gravity_body,
        marker_direction_body,
        [hover_distance],
    ])

    return reprojection_error, prediction


# --------------------------------------------------
# Load exact previous test split
# --------------------------------------------------

raw = torch.load(
    DATA,
    map_location="cpu",
    weights_only=False,
)

def get(key):
    return raw[key].detach().cpu().numpy()


observations = get("observations").reshape(-1, 32, 8)
teacher = get("teacher_observations")

valid = (
    (get("episode_steps") >= 31)
    & np.isfinite(observations).all(axis=(1, 2))
    & np.isfinite(teacher).all(axis=1)
)

observations = observations[valid]
teacher = teacher[valid]

protocol = json.loads(PROTOCOL.read_text())

split, _, _ = probe.build_masks(
    get("curriculum_levels")[valid],
    get("environment_ids")[valid],
    get("episode_ids")[valid],
    np.random.default_rng(protocol["seed"]),
)

test_mask = split == 2

observations = observations[test_mask]
teacher = teacher[test_mask]

assert len(observations) == 10764


# --------------------------------------------------
# Reconstruct current pose
# --------------------------------------------------

latest = observations[:, -1].reshape(-1, 4, 2)

all_four_visible = np.all(
    latest > -0.99,
    axis=(1, 2),
)

pnp_predictions = np.full(
    (len(observations), 10),
    np.nan,
    dtype=np.float64,
)

reprojection_errors = []

for i in np.flatnonzero(all_four_visible):
    result = estimate_pose(latest[i])

    if result is not None:
        error, prediction = result

        reprojection_errors.append(error)
        pnp_predictions[i] = prediction


# --------------------------------------------------
# Load frozen supervised Transformer
# --------------------------------------------------

checkpoint = torch.load(
    TRANSFORMER_CKPT,
    map_location="cpu",
    weights_only=False,
)

model = transformer_module.TemporalEstimator()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

learned = probe.predict(
    model,
    np.ascontiguousarray(
        observations,
        dtype=np.float32,
    ),
    np.asarray(checkpoint["target_mean"]),
    np.asarray(checkpoint["target_std"]),
    device,
)

# Compare only the same physical quantities.
truth = np.concatenate([
    teacher[:, 0:3],
    teacher[:, 9:15],
    teacher[:, 19:20],
], axis=1)

learned = np.concatenate([
    learned[:, 0:3],
    learned[:, 9:15],
    learned[:, 15:16],
], axis=1)


def compare(subset, label):
    subset = subset & np.isfinite(
        pnp_predictions
    ).all(axis=1)

    print()
    print("=" * 80)
    print(label)
    print("N =", int(subset.sum()))
    print("=" * 80)

    groups = [
        ("position_m", slice(0, 3)),
        ("projected_gravity", slice(3, 6)),
        ("marker_direction", slice(6, 9)),
        ("hover_distance_m", slice(9, 10)),
    ]

    for name, indices in groups:
        target = truth[subset, indices]

        for method, estimate in [
            ("PnP", pnp_predictions[subset, indices]),
            ("TransformerHuber", learned[subset, indices]),
        ]:
            difference = estimate - target

            rmse = np.sqrt(
                np.mean(difference ** 2)
            )

            p95 = np.quantile(
                np.linalg.norm(difference, axis=1),
                0.95,
            )

            print(
                f"{name:<22} "
                f"{method:<18} "
                f"RMSE={rmse:.6f} "
                f"P95norm={p95:.6f}"
            )


# --------------------------------------------------
# Synthetic coordinate-convention check
# --------------------------------------------------

rvec, _ = cv2.Rodrigues(C)

synthetic_corners, _ = cv2.projectPoints(
    OBJECT_POINTS,
    rvec,
    np.array([0.0, 0.0, 1.5]),
    K,
    None,
)

synthetic_result = estimate_pose(
    synthetic_corners.reshape(4, 2)
)

assert synthetic_result is not None

assert np.linalg.norm(
    synthetic_result[1][:3]
) < 1e-6, "PnP coordinate-convention check failed"

print("[PASS] Synthetic coordinate-convention check")

print()
print("========== PNP COVERAGE ==========")

available = np.isfinite(
    pnp_predictions
).all(axis=1)

print("Test samples:", len(observations))
print("Four-corner observations:", int(all_four_visible.sum()))
print("Successful PnP estimates:", int(available.sum()))
print("Coverage:", f"{100*available.mean():.2f}%")

if reprojection_errors:
    print(
        "Mean normalized reprojection error:",
        float(np.mean(reprojection_errors)),
    )

omega = np.linalg.norm(
    teacher[:, 6:9],
    axis=1,
)

compare(
    all_four_visible,
    "ALL FOUR CORNERS VISIBLE",
)

compare(
    all_four_visible & (omega < 3),
    "FOUR CORNERS VISIBLE, ANGULAR SPEED < 3 RAD/S",
)

compare(
    all_four_visible & (omega >= 3),
    "FOUR CORNERS VISIBLE, ANGULAR SPEED >= 3 RAD/S",
)

print()
print("[DONE] Matched geometric pose comparison completed.")
print("[NOTE] Missing/partial-corner samples are not assigned PnP estimates.")
print("[NOTE] No velocity estimates have been computed yet.")
