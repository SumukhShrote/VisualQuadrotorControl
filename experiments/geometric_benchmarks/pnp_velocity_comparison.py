import contextlib
import io
import runpy

import cv2
import numpy as np

source = (
    "experiments/geometric_benchmarks/"
    "pnp_pose_geometry_corrected.py"
)

# Reuse the exact held-out samples and calibrated geometry.
with contextlib.redirect_stdout(io.StringIO()):
    s = runpy.run_path(source)

obs = s["observations"].reshape(-1, 32, 4, 2)
truth = s["teacher"]
K = s["K"]
C = s["C"]
points = s["OBJECT_POINTS"]

n = len(obs)
dt = 0.02

# Frozen supervised Transformer predictions.
learned = s["probe"].predict(
    s["model"],
    np.ascontiguousarray(
        s["observations"], dtype=np.float32
    ),
    np.asarray(s["checkpoint"]["target_mean"]),
    np.asarray(s["checkpoint"]["target_std"]),
    s["device"],
)

assert learned.shape == (n, 16)

print("Test samples:", n)
print("Policy interval:", dt, "seconds")


def recover(corners):
    corners = np.asarray(corners, dtype=np.float64)

    if not np.isfinite(corners).all():
        return None

    if np.any(corners < -0.99):
        return None

    result = cv2.solvePnPGeneric(
        points,
        corners,
        K,
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not result[0]:
        return None

    best = None

    for rvec, tvec in zip(result[1], result[2]):
        Rcw = cv2.Rodrigues(rvec)[0]
        t = tvec.reshape(3)

        cam = (Rcw @ points.T).T + t

        if not np.isfinite(cam).all():
            continue

        if np.any(cam[:, 2] <= 0.05):
            continue

        projected, _ = cv2.projectPoints(
            points, rvec, tvec, K, None
        )

        error = np.mean(np.linalg.norm(
            projected.reshape(4, 2) - corners,
            axis=1,
        ))

        if best is None or error < best[0]:
            best = (error, Rcw, t)

    if best is None:
        return None

    _, Rcw, t = best

    # World -> body.
    Rbw = C.T @ Rcw

    # Drone position relative to marker,
    # expressed in marker-aligned coordinates.
    pw = -Rcw.T @ t

    return Rbw, pw


def metrics(pred, target):
    diff = pred - target

    rmse = np.sqrt(np.mean(diff ** 2))

    p95 = np.quantile(
        np.linalg.norm(diff, axis=1),
        0.95,
    )

    return rmse, p95


def print_comparison(title, mask, prediction):
    count = int(mask.sum())

    if count == 0:
        return

    print("\n" + title)
    print("N =", count)

    for name, sl in [
        ("linear_velocity", slice(0, 3)),
        ("angular_velocity", slice(3, 6)),
    ]:
        target = truth[mask, 3:9][:, sl]

        geom = prediction[mask, sl]

        neural = learned[mask, 3:9][:, sl]

        a = metrics(geom, target)
        b = metrics(neural, target)

        print(
            f"{name:<20} "
            f"PnP RMSE={a[0]:.4f} "
            f"PnP P95={a[1]:.4f} | "
            f"Transformer RMSE={b[0]:.4f} "
            f"Transformer P95={b[1]:.4f}"
        )


# Current PnP pose.
current = [
    recover(obs[i, -1])
    for i in range(n)
]

current_valid = np.array([
    pose is not None for pose in current
])

print("\nCurrent PnP coverage:", current_valid.mean())

omega_norm = np.linalg.norm(truth[:, 6:9], axis=1)

for lag in [1, 2, 4, 8]:
    print("\n" + "=" * 75)
    print(f"PNP FINITE DIFFERENCE — LAG {lag}")
    print("=" * 75)

    previous = [
        recover(obs[i, -1-lag])
        for i in range(n)
    ]

    prediction = np.full(
        (n, 6),
        np.nan,
        dtype=np.float64,
    )

    for i in range(n):
        if current[i] is None or previous[i] is None:
            continue

        Rnow, pnow = current[i]
        Rold, pold = previous[i]

        elapsed = lag * dt

        linear_world = (pnow - pold) / elapsed
        linear_body = Rnow @ linear_world

        relative_rotation = Rnow @ Rold.T

        rotation_vector = cv2.Rodrigues(
            relative_rotation
        )[0].reshape(3)

        angular_body = -rotation_vector / elapsed

        prediction[i, :3] = linear_body
        prediction[i, 3:] = angular_body

    valid = np.isfinite(prediction).all(axis=1)

    print(
        f"Coverage: {valid.sum()}/{n} "
        f"({100*valid.mean():.2f}%)"
    )

    print_comparison(
        "ALL MATCHED VALID SAMPLES",
        valid,
        prediction,
    )

    print_comparison(
        "ORDINARY ROTATION: |omega| < 3 rad/s",
        valid & (omega_norm < 3),
        prediction,
    )

    print_comparison(
        "AGGRESSIVE ROTATION: |omega| >= 3 rad/s",
        valid & (omega_norm >= 3),
        prediction,
    )

print("\n[PASS] Causal geometric velocity comparison complete.")
