from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
import random
import statistics
import subprocess

ROOT = Path("/home/sumukh/AerialCatching")
BASE = ROOT / "experiments/state_bottleneck"
OUT = Path(__file__).resolve().parent

RUN = BASE / "precision_ab_control_20261123T041432Z"

EVALUATOR = (
    BASE
    / "action_history_study_20261120T200813Z"
    / "evaluate_action_history_precision_reward.py"
)

PYTHON = "/home/sumukh/env_isaaclab/bin/python"

POLICIES = {
    "starting_checkpoint": RUN / "policy_initial.ts",
    "control_after_10": RUN / "model_19.ts",
    "control_after_20": RUN / "model_29.ts",
}

SEEDS = [7301, 8301, 9301]
LEVELS = [0, 1]

for path in [EVALUATOR, *POLICIES.values()]:
    if not path.is_file():
        raise FileNotFoundError(path)

print("[PASS] All evaluation inputs exist", flush=True)

jobs = [
    (name, path, seed, level)
    for name, path in POLICIES.items()
    for seed in SEEDS
    for level in LEVELS
]


def evaluate(gpu, assigned):
    output = {}

    for name, policy, seed, level in assigned:
        label = f"{name}_seed{seed}_L{level}"

        destination = OUT / label
        logfile = OUT / f"{label}.log"

        command = [
            PYTHON, "-u", str(EVALUATOR),
            "--task", "Isaac-QuadrotorHoverStudentDistill-v0",
            "--policy", str(policy),
            "--variant", label,
            "--level", str(level),
            "--num-envs", "64",
            "--steps", "500",
            "--seed", str(seed),
            "--device", f"cuda:{gpu}",
            "--out", str(destination),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(ROOT)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        print(
            f"[GPU {gpu}] {label}",
            flush=True,
        )

        with logfile.open("w") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )

        summary_file = destination / "summary.json"
        episodes_file = destination / "episodes.csv"

        if (
            process.returncode != 0
            or not summary_file.is_file()
            or not episodes_file.is_file()
        ):
            print(
                logfile.read_text(errors="replace")[-5000:],
                flush=True,
            )
            raise RuntimeError(f"Evaluation failed: {label}")

        summary = json.loads(summary_file.read_text())

        with episodes_file.open(newline="") as f:
            episodes = list(csv.DictReader(f))

        if len(episodes) != 64:
            raise RuntimeError(
                f"Expected 64 episodes, found {len(episodes)}"
            )

        output[(name, seed, level)] = {
            "summary": summary,
            "episodes": episodes,
        }

    return output


# Run independent evaluation processes on both GPUs.
with ThreadPoolExecutor(max_workers=2) as pool:
    a = pool.submit(evaluate, 0, jobs[::2])
    b = pool.submit(evaluate, 1, jobs[1::2])

    results = {}
    results.update(a.result())
    results.update(b.result())


def rows_for(policy, level=None):
    rows = {}

    for seed in SEEDS:
        for lv in LEVELS:
            if level is not None and lv != level:
                continue

            result = results[(policy, seed, lv)]

            for row in result["episodes"]:
                key = (seed, lv, int(row["env_id"]))

                if key in rows:
                    raise RuntimeError(f"Duplicate episode key: {key}")

                rows[key] = row

    return rows


def mean_metric(rows, metric):
    return statistics.fmean(
        float(row[metric])
        for row in rows.values()
    )


def total_survival(policy, level=None):
    total = 0

    for seed in SEEDS:
        for lv in LEVELS:
            if level is not None and lv != level:
                continue

            total += int(
                results[(policy, seed, lv)]
                ["summary"]["full_horizon_survival_count"]
            )

    return total


def bootstrap_ci(differences, repetitions=2000):
    rng = random.Random(20261123)
    n = len(differences)

    estimates = []

    for _ in range(repetitions):
        sample = [
            differences[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(statistics.fmean(sample))

    estimates.sort()

    return (
        estimates[int(0.025 * repetitions)],
        estimates[int(0.975 * repetitions)],
    )


METRICS = [
    "return_sum",
    "final_two_second_tight_hold",
    "tight_hover_fraction",
    "position_rms_m",
    "speed_rms_mps",
]

print("\n" + "=" * 90)
print("INDEPENDENT CHECKPOINT VALIDATION")
print("=" * 90)

for level in [0, 1, None]:
    level_name = (
        f"LEVEL {level}" if level is not None else "COMBINED"
    )

    print(f"\n========== {level_name} ==========")

    for name in POLICIES:
        rows = rows_for(name, level)

        survival = total_survival(name, level)
        holds = sum(
            int(float(r["final_two_second_tight_hold"]))
            for r in rows.values()
        )

        print(
            f"{name:23s}"
            f" survival={survival:3d}/{len(rows)}"
            f" holds={holds:3d}/{len(rows)}"
            f" return={mean_metric(rows, 'return_sum'):8.3f}"
            f" tight_fraction={mean_metric(rows, 'tight_hover_fraction'):.4f}"
            f" pos={mean_metric(rows, 'position_rms_m'):.5f}"
            f" speed={mean_metric(rows, 'speed_rms_mps'):.5f}"
        )

    baseline = rows_for("starting_checkpoint", level)

    for candidate in ("control_after_10", "control_after_20"):
        updated = rows_for(candidate, level)

        if set(baseline) != set(updated):
            raise RuntimeError(
                "Matched evaluation episode IDs differ"
            )

        print(f"\nPaired changes: {candidate} minus starting checkpoint")

        for metric in METRICS:
            differences = [
                float(updated[k][metric])
                - float(baseline[k][metric])
                for k in sorted(baseline)
            ]

            avg = statistics.fmean(differences)
            low, high = bootstrap_ci(differences)

            print(
                f"  {metric:32s}"
                f" delta={avg:+.6f}"
                f" approximate_95%_CI=[{low:+.6f}, {high:+.6f}]"
            )


summary_out = {
    f"{name}_seed{seed}_L{level}":
        results[(name, seed, level)]["summary"]
    for name in POLICIES
    for seed in SEEDS
    for level in LEVELS
}

(OUT / "all_summaries.json").write_text(
    json.dumps(summary_out, indent=2)
)

print("\n" + "=" * 90)
print("[PASS] Independent validation complete")
print("[RESULTS]", OUT)
print("=" * 90)
