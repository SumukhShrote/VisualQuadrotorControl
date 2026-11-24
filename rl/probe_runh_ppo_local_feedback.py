from pathlib import Path as _RepositoryPath
from pathlib import Path
import csv
import json

import torch
import torch.nn as nn


CHECKPOINT = Path(
    str(_RepositoryPath(__file__).resolve().parents[1] / 'logs/rsl_rl/CTBR_Hover_RunH_Exact_Teacher/2026-08-04_18-51-26_runh_exact_env_teacher_nominal_seed42_v1/model_749.pt')
)

OUTPUT_DIR = Path(
    str(_RepositoryPath(__file__).resolve().parents[1] / 'logs/rsl_rl/CTBR_Hover_RunH_Exact_Teacher/2026-08-04_18-51-26_runh_exact_env_teacher_nominal_seed42_v1/local_feedback_probe_v1')
)

# Environment contract.
#
# At level attitude and zero yaw:
#
#   marker_point - robot_position
#       = [hover_distance, 0, 0] + position_error
#
# where:
#   position_error = hover_point - robot_position
#
HOVER_DISTANCE_M = 1.5

VARIABLES = [
    "e_x",
    "e_y",
    "e_z",
    "v_x",
    "v_y",
    "v_z",
]

ACTION_NAMES = [
    "thrust",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
]

DELTAS = [
    0.02,
    0.05,
    0.10,
]


def build_actor():
    actor = nn.Sequential(
        nn.Linear(20, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 4),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    full_state_dict = checkpoint["model_state_dict"]

    actor_state_dict = {
        key.removeprefix("actor."): value
        for key, value in full_state_dict.items()
        if key.startswith("actor.")
    }

    result = actor.load_state_dict(
        actor_state_dict,
        strict=True,
    )

    print("[PASS] Actor strict load:", result)

    actor.eval()

    return actor


def physical_state_to_observation(x):
    """
    x:
        [e_x, e_y, e_z, v_x, v_y, v_z]

    Probe assumptions:
        - level attitude
        - zero yaw
        - zero angular velocity
        - previous action = zero
        - body/world axes aligned

    PPO observation:
        [0:3]   pos_err_b
        [3:6]   lin_vel_b
        [6:9]   ang_vel_b
        [9:12]  gravity_b
        [12:15] marker_dir_b
        [15:19] previous_action
        [19]    distance_to_hover
    """

    assert x.shape == (6,)

    position_error = x[0:3]
    linear_velocity = x[3:6]

    angular_velocity = torch.zeros(
        3,
        dtype=x.dtype,
        device=x.device,
    )

    gravity_b = torch.tensor(
        [0.0, 0.0, -1.0],
        dtype=x.dtype,
        device=x.device,
    )

    marker_vector = torch.stack(
        (
            HOVER_DISTANCE_M + position_error[0],
            position_error[1],
            position_error[2],
        )
    )

    marker_direction = (
        marker_vector
        / torch.linalg.norm(marker_vector)
    )

    previous_action = torch.zeros(
        4,
        dtype=x.dtype,
        device=x.device,
    )

    distance = torch.linalg.norm(
        position_error
    ).reshape(1)

    obs = torch.cat(
        (
            position_error,
            linear_velocity,
            angular_velocity,
            gravity_b,
            marker_direction,
            previous_action,
            distance,
        )
    )

    assert obs.shape == (20,)

    return obs


@torch.no_grad()
def policy(actor, x):
    obs = physical_state_to_observation(x)

    raw_action = actor(
        obs.unsqueeze(0)
    ).squeeze(0)

    clipped_action = torch.clamp(
        raw_action,
        -1.0,
        1.0,
    )

    return obs, raw_action, clipped_action


def format_vector(x):
    return "[" + ", ".join(
        f"{float(v):+.7f}"
        for v in x
    ) + "]"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    actor = build_actor()

    x0 = torch.zeros(
        6,
        dtype=torch.float32,
    )

    obs0, raw0, clipped0 = policy(
        actor,
        x0,
    )

    print()
    print("=" * 110)
    print("RUN-H PRIVILEGED PPO LOCAL FEEDBACK PROBE")
    print("=" * 110)

    print("\nCheckpoint:")
    print(CHECKPOINT)

    print("\nNominal physical state:")
    print(
        dict(
            position_error_m=[0.0, 0.0, 0.0],
            linear_velocity_mps=[0.0, 0.0, 0.0],
            angular_velocity_radps=[0.0, 0.0, 0.0],
            hover_distance_m=HOVER_DISTANCE_M,
        )
    )

    print("\nNominal 20-D observation:")
    print(format_vector(obs0))

    print("\nNominal raw action:")
    print(format_vector(raw0))

    print("\nNominal clipped action:")
    print(format_vector(clipped0))

    rows = []

    gain_matrices_raw = {}
    gain_matrices_clipped = {}

    for delta in DELTAS:
        K_raw = torch.zeros(
            4,
            6,
            dtype=torch.float32,
        )

        K_clipped = torch.zeros(
            4,
            6,
            dtype=torch.float32,
        )

        print()
        print("=" * 110)
        print(f"DELTA = {delta:.3f}")
        print("=" * 110)

        for variable_index, variable_name in enumerate(
            VARIABLES
        ):
            x_minus = x0.clone()
            x_plus = x0.clone()

            x_minus[variable_index] = -delta
            x_plus[variable_index] = +delta

            (
                obs_minus,
                raw_minus,
                clipped_minus,
            ) = policy(
                actor,
                x_minus,
            )

            (
                obs_plus,
                raw_plus,
                clipped_plus,
            ) = policy(
                actor,
                x_plus,
            )

            gain_raw = (
                raw_plus - raw_minus
            ) / (2.0 * delta)

            gain_clipped = (
                clipped_plus - clipped_minus
            ) / (2.0 * delta)

            K_raw[:, variable_index] = gain_raw
            K_clipped[:, variable_index] = (
                gain_clipped
            )

            print(
                f"\n{variable_name}:"
                f"  -delta={format_vector(raw_minus)}"
                f"\n{'':6s}"
                f"  +delta={format_vector(raw_plus)}"
                f"\n{'':6s}"
                f"  gain  ={format_vector(gain_raw)}"
            )

            row = {
                "delta": delta,
                "variable": variable_name,
            }

            for j, action_name in enumerate(
                ACTION_NAMES
            ):
                row[
                    f"raw_minus_{action_name}"
                ] = float(raw_minus[j])

                row[
                    f"raw_plus_{action_name}"
                ] = float(raw_plus[j])

                row[
                    f"raw_gain_{action_name}"
                ] = float(gain_raw[j])

                row[
                    f"clipped_minus_{action_name}"
                ] = float(clipped_minus[j])

                row[
                    f"clipped_plus_{action_name}"
                ] = float(clipped_plus[j])

                row[
                    f"clipped_gain_{action_name}"
                ] = float(gain_clipped[j])

            rows.append(row)

        gain_matrices_raw[str(delta)] = (
            K_raw.tolist()
        )

        gain_matrices_clipped[str(delta)] = (
            K_clipped.tolist()
        )

        print()
        print(
            "RAW LOCAL GAIN MATRIX"
            "  rows=[T,p,q,r]"
            "  cols=[ex,ey,ez,vx,vy,vz]"
        )

        for action_index, action_name in enumerate(
            ACTION_NAMES
        ):
            print(
                f"{action_name:10s}: "
                + " ".join(
                    f"{float(v):+10.5f}"
                    for v in K_raw[action_index]
                )
            )

        print()
        print(
            "CLIPPED/APPLIED LOCAL GAIN MATRIX"
            "  rows=[T,p,q,r]"
            "  cols=[ex,ey,ez,vx,vy,vz]"
        )

        for action_index, action_name in enumerate(
            ACTION_NAMES
        ):
            print(
                f"{action_name:10s}: "
                + " ".join(
                    f"{float(v):+10.5f}"
                    for v in K_clipped[action_index]
                )
            )

    csv_path = (
        OUTPUT_DIR
        / "ppo_local_feedback_finite_difference.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(CHECKPOINT),
        "architecture": "20-64-64-4 tanh",
        "empirical_normalization": False,
        "deterministic_inference": True,
        "hover_distance_m": HOVER_DISTANCE_M,
        "nominal_observation": [
            float(v)
            for v in obs0
        ],
        "nominal_raw_action": [
            float(v)
            for v in raw0
        ],
        "nominal_clipped_action": [
            float(v)
            for v in clipped0
        ],
        "variables": VARIABLES,
        "actions": ACTION_NAMES,
        "gain_matrix_convention": (
            "rows=[thrust,roll_rate,pitch_rate,yaw_rate], "
            "cols=[e_x,e_y,e_z,v_x,v_y,v_z], "
            "e=hover_position-robot_position"
        ),
        "gain_matrices_raw": gain_matrices_raw,
        "gain_matrices_clipped": (
            gain_matrices_clipped
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "ppo_local_feedback_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("=" * 110)
    print("[PASS]", csv_path)
    print("[PASS]", json_path)
    print("=" * 110)


if __name__ == "__main__":
    main()
