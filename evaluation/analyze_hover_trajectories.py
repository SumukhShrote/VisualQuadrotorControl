#!/usr/bin/env python3
"""Physical hover analysis using recorded first-episode trajectories.

State contract:
  position_w:        [0:3] meters
  linear_velocity_w: [7:10] meters/second
  distance_to_goal:  [17] meters

Metrics:
  position_RMS = sqrt(mean(distance_to_goal**2))
  speed_RMS    = sqrt(mean(||linear_velocity_w||**2))

First-episode-only:
  Use trajectory indices [0, episode_length).
  Exclude later episodes following automatic resets.

Aggregate RMS metrics:
  Calculate RMS independently for each initial episode,
  then average across episodes (equal episode weighting).

Settled-hover diagnostic:
  Among successful episodes, inspect final 100 samples (~2 s).
  Strict hold = at least 95% of those samples simultaneously
  within 0.10 m of the goal and below 0.20 m/s.

This hold definition is a diagnostic threshold, not a previously
established task success criterion.
"""

from pathlib import Path
import csv
import math
import statistics

import torch


ROOT = Path(__file__).resolve().parents[1]

RUN = ROOT / (
    "results/historical_experiments/state_bottleneck/overnight_runs/"
    "motion_20261119T045119Z"
)

RECORDS = ROOT / "results/historical_evaluations/initial_episode_records"

POLICIES = ("frozen", "control")
REGIMES = ("standard", "motion")

SUMMARY = RUN / "hover_accuracy_summary.csv"
DETAILS = RUN / "hover_accuracy_episodes.csv"


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else float("nan")


def fmt(value, digits=3):
    return (
        f"{value:.{digits}f}"
        if math.isfinite(value)
        else "N/A"
    )


def load_run(regime, level_index, policy):
    prefix = f"eval_{regime}_L{level_index}_{policy}"

    if policy == "frozen":
        trajectory_path = ROOT / "results/historical_evaluations/raw_tensors" / f"{prefix}eval_full_states.pt"
        if not trajectory_path.is_file():
            trajectory_path = ROOT / "checkpoints" / f"{prefix}eval_full_states.pt"
    else:
        trajectory_path = (
            RUN / f"{prefix}eval_full_states.pt"
        )

    record_path = (
        RECORDS / f"{prefix}_initial_episodes.pt"
    )

    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)

    if not record_path.is_file():
        raise FileNotFoundError(record_path)

    states = torch.load(
        trajectory_path,
        map_location="cpu",
        weights_only=False,
    )

    record = torch.load(
        record_path,
        map_location="cpu",
        weights_only=False,
    )

    if tuple(states.shape) != (64, 500, 18):
        raise RuntimeError(
            f"{prefix}: unexpected state shape {tuple(states.shape)}"
        )

    if not torch.isfinite(states).all():
        raise RuntimeError(f"{prefix}: nonfinite trajectory")

    if not bool(record["recorded"].all()):
        raise RuntimeError(f"{prefix}: missing first-episode records")

    if not bool(
        (record["survived"] ^ record["died"]).all()
    ):
        raise RuntimeError(
            f"{prefix}: inconsistent survival/death records"
        )

    if (
        str(record["save_prefix"]) != prefix
        or int(record["num_envs"]) != 64
        or int(record["max_steps"]) != 500
    ):
        raise RuntimeError(
            f"{prefix}: metadata does not match trajectory"
        )

    # The first logged state should be close to the recorded
    # initial state, whether sampled just before/after the first step.
    initial_xyz = record["initial_full_state"][:, :3]
    first_xyz = states[:, 0, :3]

    initial_displacement = torch.linalg.vector_norm(
        first_xyz - initial_xyz,
        dim=1,
    )

    maximum_displacement = float(
        initial_displacement.max()
    )

    print(
        f"[ALIGNMENT] {prefix}: "
        f"max initial-to-first-frame displacement="
        f"{maximum_displacement:.5f} m"
    )

    if maximum_displacement > 0.15:
        raise RuntimeError(
            f"{prefix}: trajectory time alignment requires inspection"
        )

    rows = []

    for env_id in range(64):
        length = int(record["episode_length"][env_id])

        if not 1 <= length <= 500:
            raise RuntimeError(
                f"{prefix}, env={env_id}: invalid length {length}"
            )

        survived = bool(record["survived"][env_id])

        # Restrict to the first episode.
        # The terminal sample is excluded where indexing is ambiguous.
        trajectory = states[env_id, :length]

        distance = trajectory[:, 17].double()

        speed = torch.linalg.vector_norm(
            trajectory[:, 7:10].double(),
            dim=1,
        )

        if bool((distance < 0).any()):
            raise RuntimeError(
                f"{prefix}, env={env_id}: negative distance"
            )

        pos_rms = float(
            torch.sqrt(torch.mean(distance.square()))
        )

        speed_rms = float(
            torch.sqrt(torch.mean(speed.square()))
        )

        within_10cm = float(
            (distance <= 0.10).double().mean()
        )

        within_20cm = float(
            (distance <= 0.20).double().mean()
        )

        last2_pos = float("nan")
        last2_speed = float("nan")
        last2_within10 = float("nan")
        last2_within20 = float("nan")
        strict_hold = False

        if survived and length >= 100:
            d2 = distance[-100:]
            v2 = speed[-100:]

            last2_pos = float(
                torch.sqrt(torch.mean(d2.square()))
            )

            last2_speed = float(
                torch.sqrt(torch.mean(v2.square()))
            )

            last2_within10 = float(
                (d2 <= 0.10).double().mean()
            )

            last2_within20 = float(
                (d2 <= 0.20).double().mean()
            )

            joint_fraction = float(
                (
                    (d2 <= 0.10) &
                    (v2 <= 0.20)
                ).double().mean()
            )

            strict_hold = joint_fraction >= 0.95

        rows.append({
            "regime": regime,
            "level": level_index / 4.0,
            "policy": policy,
            "env_id": env_id,
            "seed": int(record["seed"]),
            "survived": int(survived),
            "episode_length": length,
            "initial_episode_return": float(
                record["episode_return"][env_id]
            ),
            "position_rms_m": pos_rms,
            "speed_rms_mps": speed_rms,
            "max_distance_m": float(distance.max()),
            "fraction_within_10cm": within_10cm,
            "fraction_within_20cm": within_20cm,
            "last2_position_rms_m": last2_pos,
            "last2_speed_rms_mps": last2_speed,
            "last2_fraction_within_10cm": last2_within10,
            "last2_fraction_within_20cm": last2_within20,
            "strict_last2_hold": int(strict_hold),
        })

    return record, rows


def summarize(rows):
    successful = [r for r in rows if r["survived"]]

    return {
        "regime": rows[0]["regime"],
        "level": rows[0]["level"],
        "policy": rows[0]["policy"],
        "n": len(rows),
        "survival_percent": 100 * len(successful) / len(rows),
        "mean_initial_return": mean(
            r["initial_episode_return"] for r in rows
        ),
        "mean_episode_length": mean(
            r["episode_length"] for r in rows
        ),
        "mean_position_rms_m": mean(
            r["position_rms_m"] for r in rows
        ),
        "mean_speed_rms_mps": mean(
            r["speed_rms_mps"] for r in rows
        ),
        "mean_fraction_within_10cm": mean(
            r["fraction_within_10cm"] for r in rows
        ),
        "mean_fraction_within_20cm": mean(
            r["fraction_within_20cm"] for r in rows
        ),
        "successful_last2_position_rms_m": mean(
            r["last2_position_rms_m"] for r in successful
        ),
        "successful_last2_speed_rms_mps": mean(
            r["last2_speed_rms_mps"] for r in successful
        ),
        "successful_last2_within_10cm": mean(
            r["last2_fraction_within_10cm"] for r in successful
        ),
        "successful_last2_within_20cm": mean(
            r["last2_fraction_within_20cm"] for r in successful
        ),
        "strict_hold_count": sum(
            r["strict_last2_hold"] for r in rows
        ),
    }


def save_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    summaries = []
    details = []

    for regime in REGIMES:
        for level_index in range(5):
            records = {}

            for policy in POLICIES:
                record, rows = load_run(
                    regime, level_index, policy
                )

                records[policy] = record
                details.extend(rows)
                summaries.append(summarize(rows))

            # Verify paired initial conditions.
            frozen = records["frozen"]
            control = records["control"]

            state_difference = float(
                (
                    frozen["initial_full_state"] -
                    control["initial_full_state"]
                ).abs().max()
            )

            obs_difference = float(
                (
                    frozen["initial_policy_observation"] -
                    control["initial_policy_observation"]
                ).abs().max()
            )

            print(
                f"[PAIRED] regime={regime} "
                f"level={level_index / 4:.2f}: "
                f"state_max_diff={state_difference:.8g}, "
                f"corners_max_diff={obs_difference:.8g}"
            )

            if state_difference > 1e-5 or obs_difference > 1e-5:
                raise RuntimeError(
                    "Initial conditions differ across policies. "
                    "Do not present this as a paired comparison."
                )

    save_csv(SUMMARY, summaries)
    save_csv(DETAILS, details)

    for regime in REGIMES:
        print("\n" + "=" * 105)
        print(
            "STANDARD RESETS"
            if regime == "standard"
            else "MOTION-PERTURBED RESETS"
        )
        print("=" * 105)

        print(
            f"{'Level':>5} {'Policy':>8} {'Survive':>8} "
            f"{'Return':>8} {'PosRMS':>9} {'SpeedRMS':>9} "
            f"{'Within10':>9} {'Within20':>9}"
        )

        for row in summaries:
            if row["regime"] != regime:
                continue

            print(
                f"{row['level']:>5.2f} "
                f"{row['policy']:>8} "
                f"{row['survival_percent']:>7.1f}% "
                f"{row['mean_initial_return']:>8.2f} "
                f"{row['mean_position_rms_m']:>9.3f} "
                f"{row['mean_speed_rms_mps']:>9.3f} "
                f"{100 * row['mean_fraction_within_10cm']:>8.1f}% "
                f"{100 * row['mean_fraction_within_20cm']:>8.1f}%"
            )

        print()
        print("HOVER PRECISION AMONG SUCCESSFUL EPISODES")
        print(
            f"{'Level':>5} {'Policy':>8} "
            f"{'Last2Pos':>9} {'Last2Speed':>11} "
            f"{'Last2<10cm':>11} {'StrictHold':>11}"
        )

        for row in summaries:
            if row["regime"] != regime:
                continue

            print(
                f"{row['level']:>5.2f} "
                f"{row['policy']:>8} "
                f"{fmt(row['successful_last2_position_rms_m']):>9} "
                f"{fmt(row['successful_last2_speed_rms_mps']):>11} "
                f"{fmt(100 * row['successful_last2_within_10cm'], 1):>10}% "
                f"{row['strict_hold_count']:>7}/{row['n']}"
            )

    print("\n[PASS] First-episode-only analysis completed.")
    print("[SUMMARY]", SUMMARY)
    print("[DETAILS]", DETAILS)


if __name__ == "__main__":
    main()
