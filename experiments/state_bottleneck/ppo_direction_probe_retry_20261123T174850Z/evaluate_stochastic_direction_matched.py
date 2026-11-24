from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
import statistics
import subprocess
import torch

ROOT = Path("/home/sumukh/AerialCatching")
BASE = ROOT / "experiments/state_bottleneck"
RUN = Path(__file__).resolve().parent
OUT = RUN / "stochastic_direction_matched_reward"
OUT.mkdir(exist_ok=True)

PYTHON = "/home/sumukh/env_isaaclab/bin/python"

EVALUATOR = (
    BASE
    / "action_history_study_20261120T200813Z"
    / "evaluate_action_history_precision_reward_matched_stable.py"
)

CHECKPOINT = (
    BASE
    / "long_horizon_gae_20261123T032419Z"
    / "model_9.pt"
)

REFERENCE = (
    BASE
    / "precision_ab_control_20261123T041432Z"
    / "policy_initial.ts"
)

SCALES = [-4, -1, 0, 1, 4]
SEEDS = [7513, 8513, 9513]

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True,
)

std = ckpt["model_state_dict"]["std"].detach().float()

if std.shape != (4,) or not torch.isfinite(std).all():
    raise RuntimeError("Invalid saved action standard deviation")

if (std <= 0).any():
    raise RuntimeError("Nonpositive action standard deviation")

print("[ACTION STD]", std.tolist(), flush=True)


class StochasticCTBR(torch.nn.Module):
    def __init__(self, policy, action_std):
        super().__init__()
        self.policy = policy
        self.register_buffer("action_std", action_std)

    def forward(
        self,
        corner_history: torch.Tensor,
        actions8: torch.Tensor,
    ) -> torch.Tensor:
        deterministic_action = self.policy(corner_history, actions8)

        raw_mean = torch.atanh(
            deterministic_action.clamp(-0.999999, 0.999999)
        )

        noise = torch.randn_like(raw_mean) * self.action_std

        return torch.tanh(raw_mean + noise)


def source_policy(scale):
    if scale == 0:
        return REFERENCE

    if scale < 0:
        return (
            RUN / "direction_policies"
            / f"direction_minus_{abs(scale)}.ts"
        )

    return (
        RUN / "direction_policies"
        / f"direction_plus_{scale}.ts"
    )


policies = {}

for scale in SCALES:
    source = source_policy(scale)

    if not source.is_file():
        raise FileNotFoundError(source)

    base = torch.jit.load(str(source), map_location="cpu")

    wrapper = StochasticCTBR(base, std)

    scripted = torch.jit.script(wrapper)

    destination = OUT / f"sampled_{scale:+d}.ts"

    torch.jit.save(scripted, str(destination))

    # Verify that the serialized module can execute
    # and produces finite, bounded CTBR actions.
    test_policy = torch.jit.load(
        str(destination),
        map_location="cpu",
    )

    # The deployed K=8 policy accepts corner history and
    # action history as separate arguments.
    test_inputs = [
        (torch.zeros(8, 32, 8), torch.zeros(8, 8, 4)),
        (torch.zeros(8, 256), torch.zeros(8, 32)),
        (torch.zeros(8, 32, 8), torch.zeros(8, 32)),
        (torch.zeros(8, 256), torch.zeros(8, 8, 4)),
    ]

    actions = None
    errors = []

    with torch.no_grad():
        for corners, history in test_inputs:
            try:
                candidate = test_policy(corners, history)

                if candidate.shape == (8, 4):
                    actions = candidate
                    break

            except Exception as exc:
                errors.append(str(exc)[-250:])

    if actions is None:
        raise RuntimeError(
            "Could not execute the two-input policy. "
            f"Errors: {errors}"
        )

    if actions.shape != (8, 4):
        raise RuntimeError(
            f"Unexpected action shape at scale {scale}: "
            f"{tuple(actions.shape)}"
        )

    if not torch.isfinite(actions).all():
        raise RuntimeError("Nonfinite stochastic action")

    if (actions.abs() > 1.0).any():
        raise RuntimeError("Stochastic action out of bounds")

    policies[scale] = destination

    print(
        f"[PASS] Stochastic policy exported: scale={scale:+d}",
        flush=True,
    )


def evaluate_seed(seed, gpu):
    output = {}

    for scale in SCALES:
        label = f"sampled_{scale:+d}_seed{seed}_L1"

        destination = OUT / label
        log = OUT / f"{label}.log"

        command = [
            PYTHON, "-u", str(EVALUATOR),
            "--task",
            "Isaac-QuadrotorHoverStudentDistill-v0",
            "--policy", str(policies[scale]),
            "--variant", label,
            "--level", "1",
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

        with log.open("w") as handle:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        summary_path = destination / "summary.json"
        episodes_path = destination / "episodes.csv"

        if (
            result.returncode != 0
            or not summary_path.is_file()
            or not episodes_path.is_file()
        ):
            print(log.read_text(errors="replace")[-5000:])
            raise RuntimeError(f"Evaluation failed: {label}")

        summary = json.loads(summary_path.read_text())

        with episodes_path.open(newline="") as f:
            episodes = list(csv.DictReader(f))

        if len(episodes) != 64:
            raise RuntimeError(
                f"Expected 64 episodes: {label}"
            )

        output[(scale, seed)] = {
            "summary": summary,
            "episodes": episodes,
        }

    return output


with ThreadPoolExecutor(max_workers=2) as pool:
    a = pool.submit(evaluate_seed, 7513, 0)
    b = pool.submit(evaluate_seed, 8513, 1)

    results = {}
    results.update(a.result())
    results.update(b.result())

# Keep policies for each seed on the same GPU.
results.update(evaluate_seed(9513, 0))


def aggregate(scale):
    episodes = []
    survival = 0

    for seed in SEEDS:
        record = results[(scale, seed)]

        episodes.extend(record["episodes"])

        survival += int(
            record["summary"]["full_horizon_survival_count"]
        )

    return {
        "survival": survival,
        "holds": sum(
            int(float(r["final_two_second_tight_hold"]))
            for r in episodes
        ),
        "return": statistics.fmean(
            float(r["return_sum"])
            for r in episodes
        ),
        "tight_fraction": statistics.fmean(
            float(r["tight_hover_fraction"])
            for r in episodes
        ),
        "position_rms": statistics.fmean(
            float(r["position_rms_m"])
            for r in episodes
        ),
        "speed_rms": statistics.fmean(
            float(r["speed_rms_mps"])
            for r in episodes
        ),
    }


print()
print("=" * 95)
print("STOCHASTIC PPO UPDATE-DIRECTION EVALUATION — LEVEL 1")
print("=" * 95)

metrics = {
    scale: aggregate(scale)
    for scale in SCALES
}

baseline = metrics[0]

for scale in SCALES:
    m = metrics[scale]

    print(
        f"scale={scale:+3d} "
        f"survival={m['survival']:3d}/192 "
        f"holds={m['holds']:3d}/192 "
        f"return={m['return']:8.4f} "
        f"tight={m['tight_fraction']:.5f} "
        f"pos={m['position_rms']:.6f} "
        f"speed={m['speed_rms']:.6f}"
    )

    if scale != 0:
        print(
            f"       Δreturn={m['return']-baseline['return']:+.5f} "
            f"Δholds={m['holds']-baseline['holds']:+d} "
            f"Δtight={m['tight_fraction']-baseline['tight_fraction']:+.5f}"
        )

print()
print("=" * 95)
print("PER-SEED STOCHASTIC RETURNS")
print("=" * 95)

for seed in SEEDS:
    print(f"\nSeed {seed}")

    for scale in SCALES:
        rows = results[(scale, seed)]["episodes"]

        mean_return = statistics.fmean(
            float(r["return_sum"]) for r in rows
        )

        holds = sum(
            int(float(r["final_two_second_tight_hold"]))
            for r in rows
        )

        print(
            f"  scale={scale:+3d} "
            f"return={mean_return:.4f} "
            f"holds={holds}/64"
        )

(OUT / "stochastic_results.json").write_text(
    json.dumps(
        {
            "scales": SCALES,
            "seeds": SEEDS,
            "action_std": std.tolist(),
            "metrics": {
                str(k): v for k, v in metrics.items()
            },
        },
        indent=2,
    )
)

print()
print("[PASS] Stochastic direction evaluation completed")
print("[RESULTS]", OUT / "stochastic_results.json")
