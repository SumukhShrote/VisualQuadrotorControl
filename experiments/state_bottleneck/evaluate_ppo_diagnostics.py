from pathlib import Path
import subprocess
import torch

ROOT = Path("/home/sumukh/AerialCatching")

PYTHON = "/home/sumukh/env_isaaclab/bin/python"

BASE = ROOT / (
    "experiments/state_bottleneck/overnight_runs/"
    "motion_20261119T045119Z/"
    "corners_tcn_control_continued.pt"
)

PPO = ROOT / (
    "logs/rsl_rl/CTBR_Hover_Corners_Asymmetric_PPO/"
    "2026-11-19_12-55-54"
)

OUT = ROOT / (
    "experiments/state_bottleneck/"
    "ppo_large_scale/diagnostic_evaluations"
)

RECORDS = ROOT / "eval_logs/initial_episode_records"

OUT.mkdir(parents=True, exist_ok=True)

TASK = "Isaac-QuadrotorHoverAsymmetricPPO-v0"

SEED = 9314
LEVEL = 1.0
ENVS = 64

cases = [
    ("dagger", None),
    ("ppo_warmup_9", 9),
    ("ppo_25", 25),
    ("ppo_75", 75),
    ("ppo_125", 125),
]

results = {}

for label, step in cases:
    prefix = f"diag_identical_L1_{label}"

    command = [
        PYTHON,
        "-u",
        "-m",
        "rl.eval_rslrl",
        "--task", TASK,
        "--num_envs", str(ENVS),
        "--seed", str(SEED),
        "--curriculum_level", str(LEVEL),
        "--use_gru", "false",
        "--save_prefix", prefix,
        "--headless",
    ]

    if step is None:
        command += [
            "--student_checkpoint", str(BASE),
        ]
        trajectory_dir = BASE.parent
    else:
        checkpoint = PPO / f"model_{step}.pt"

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        command += [
            "--experiment_name",
            "CTBR_Hover_Corners_Asymmetric_PPO",
            "--load_run",
            "2026-11-19_12-55-54",
            "--load_checkpoint",
            f"model_{step}.pt",
        ]

        trajectory_dir = PPO

    record_path = (
        RECORDS / f"{prefix}_initial_episodes.pt"
    )

    trajectory_path = (
        trajectory_dir / f"{prefix}eval_full_states.pt"
    )

    log_path = OUT / f"{label}.log"

    print(f"\n[EVALUATING] {label}", flush=True)

    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        print(
            f"[FAILED] {label}, exit={result.returncode}"
        )

        print(
            "\n".join(
                log_path.read_text(
                    errors="replace"
                ).splitlines()[-35:]
            )
        )

        raise RuntimeError(
            f"Evaluation failed: {label}"
        )

    record = torch.load(
        record_path,
        map_location="cpu",
        weights_only=False,
    )

    states = torch.load(
        trajectory_path,
        map_location="cpu",
        weights_only=False,
    )

    assert tuple(states.shape) == (64, 500, 18)
    assert bool(record["recorded"].all())

    lengths = record["episode_length"].long()
    survived = record["survived"].bool()

    position_rms = []
    speed_rms = []
    within_20 = []

    for env_id in range(64):
        length = int(lengths[env_id])

        trajectory = states[env_id, :length]

        distance = trajectory[:, 17]

        speed = torch.linalg.vector_norm(
            trajectory[:, 7:10],
            dim=-1,
        )

        position_rms.append(
            float(
                distance.square().mean().sqrt()
            )
        )

        speed_rms.append(
            float(
                speed.square().mean().sqrt()
            )
        )

        within_20.append(
            float(
                (distance <= 0.20).float().mean()
            )
        )

    if bool(survived.any()):
        successful_final_distance = float(
            states[survived, 498, 17].mean()
        )
    else:
        successful_final_distance = float("nan")

    results[label] = {
        "initial": record["initial_full_state"],
        "obs": record["initial_policy_observation"],
        "states": states,
        "survived": survived,
        "return": float(
            record["episode_return"].mean()
        ),
        "length": float(lengths.float().mean()),
        "position_rms": sum(position_rms) / len(position_rms),
        "speed_rms": sum(speed_rms) / len(speed_rms),
        "within_20": sum(within_20) / len(within_20),
        "final_distance": successful_final_distance,
    }

    print(
        f"[RESULT] {label}: "
        f"survival={int(survived.sum())}/64 "
        f"length={results[label]['length']:.1f} "
        f"position_rms={results[label]['position_rms']:.3f} "
        f"within20={100*results[label]['within_20']:.1f}%",
        flush=True,
    )

print("\n" + "=" * 100)
print("DETERMINISTIC EVALUATION — IDENTICAL TASK, SEED, LEVEL")
print("=" * 100)

print(
    f"{'Policy':<18} {'Survival':>10} {'Return':>9} "
    f"{'Length':>9} {'PosRMS':>9} "
    f"{'SpeedRMS':>10} {'Within20':>9} "
    f"{'FinalDist':>10}"
)

for label, _ in cases:
    r = results[label]

    print(
        f"{label:<18} "
        f"{int(r['survived'].sum()):>7}/64 "
        f"{r['return']:>9.2f} "
        f"{r['length']:>9.1f} "
        f"{r['position_rms']:>9.3f} "
        f"{r['speed_rms']:>10.3f} "
        f"{100*r['within_20']:>8.1f}% "
        f"{r['final_distance']:>10.3f}"
    )

print("\n========== POSITIVE CONTROL ==========")

base = results["dagger"]
warmup = results["ppo_warmup_9"]

state_diff = float(
    (base["initial"] - warmup["initial"])
    .abs()
    .max()
)

obs_diff = float(
    (base["obs"] - warmup["obs"])
    .abs()
    .max()
)

trajectory_diff = float(
    (base["states"] - warmup["states"])
    .abs()
    .max()
)

survival_agreement = float(
    (
        base["survived"] ==
        warmup["survived"]
    ).float().mean()
)

print("Initial full-state max difference:", state_diff)
print("Initial corners max difference:", obs_diff)
print("Full trajectory max difference:", trajectory_diff)
print("Survival agreement:", survival_agreement)

if state_diff != 0 or obs_diff != 0:
    print(
        "[WARNING] Initial conditions are not identical. "
        "Results must not be treated as paired."
    )

if trajectory_diff > 1e-4:
    print(
        "[WARNING] Frozen DAgger and warm-up PPO "
        "do not reproduce identical trajectories. "
        "Investigate before attributing differences "
        "to PPO optimization."
    )
else:
    print(
        "[PASS] Frozen DAgger actor and PPO warm-up "
        "checkpoint reproduce matching trajectories."
    )

print("\n[LOG DIRECTORY]", OUT)
