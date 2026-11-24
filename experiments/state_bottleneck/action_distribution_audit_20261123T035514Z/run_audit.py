from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
import subprocess
import sys

import torch


ROOT = Path("/home/sumukh/AerialCatching")
BASE = ROOT / "experiments/state_bottleneck"

RUN = BASE / "long_horizon_gae_20261123T032419Z"

EVALUATOR = (
    BASE
    / "action_history_study_20261120T200813Z"
    / "evaluate_action_history_precision_reward.py"
)

OUT = Path(__file__).resolve().parent

PYTHON = "/home/sumukh/env_isaaclab/bin/python"

dagger_ts = RUN / "policy_initial.ts"
best_ts = RUN / "model_9.ts"
best_pt = RUN / "model_9.pt"

for path in (EVALUATOR, dagger_ts, best_ts, best_pt):
    if not path.is_file():
        raise FileNotFoundError(path)

# Load the actual trained action standard deviation.
checkpoint = torch.load(
    best_pt,
    map_location="cpu",
    weights_only=True,
)

state = checkpoint["model_state_dict"]

if "std" not in state:
    raise RuntimeError(
        "Could not find trained PPO action std in checkpoint"
    )

ppo_std = state["std"].detach().float().flatten().clamp(
    min=1e-4,
    max=0.3,
)

assert ppo_std.numel() == 4

dagger_std = torch.full_like(ppo_std, 0.03)

print("[PPO ACTION STD]", ppo_std.tolist(), flush=True)
print("[DAgger ACTION STD]", dagger_std.tolist(), flush=True)

# Import wrapper from a real file so TorchScript can compile it.
spec = importlib.util.spec_from_file_location(
    "stochastic_policy_module",
    OUT / "stochastic_policy_module.py",
)

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

StochasticCTBR = module.StochasticCTBR

policies = {
    "dagger_deterministic": dagger_ts,
    "ppo_deterministic": best_ts,
}

for label, source, std in (
    ("dagger_stochastic", dagger_ts, dagger_std),
    ("ppo_stochastic", best_ts, ppo_std),
):
    base = torch.jit.load(str(source), map_location="cpu").eval()

    wrapper = StochasticCTBR(base, std).eval()
    compiled = torch.jit.script(wrapper)

    destination = OUT / f"{label}.ts"
    compiled.save(str(destination))

    policies[label] = destination

    print("[PASS] Created", destination, flush=True)


jobs = []

for label, policy in policies.items():
    for level in (0, 1):
        jobs.append((label, policy, level))


def evaluate(gpu, assigned_jobs):
    results = {}

    for label, policy, level in assigned_jobs:
        name = f"{label}_L{level}"
        destination = OUT / name
        logfile = OUT / f"{name}.log"

        command = [
            PYTHON,
            "-u",
            str(EVALUATOR),
            "--task",
            "Isaac-QuadrotorHoverStudentDistill-v0",
            "--policy",
            str(policy),
            "--variant",
            name,
            "--level",
            str(level),
            "--num-envs",
            "64",
            "--steps",
            "500",
            "--seed",
            "6300",
            "--device",
            f"cuda:{gpu}",
            "--out",
            str(destination),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(ROOT)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        print(
            f"[GPU {gpu}] Evaluating {name}",
            flush=True,
        )

        with logfile.open("w") as f:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        summary_file = destination / "summary.json"

        if process.returncode != 0 or not summary_file.is_file():
            print(logfile.read_text(errors="replace")[-4000:])
            raise RuntimeError(f"Evaluation failed: {name}")

        results[name] = json.loads(summary_file.read_text())

        print(
            f"[GPU {gpu}] Completed {name}",
            flush=True,
        )

    return results


# One independent evaluation process per GPU.
with ThreadPoolExecutor(max_workers=2) as pool:
    first = pool.submit(evaluate, 0, jobs[::2])
    second = pool.submit(evaluate, 1, jobs[1::2])

    results = {}
    results.update(first.result())
    results.update(second.result())


print("\n" + "=" * 90)
print("STOCHASTIC VS DETERMINISTIC POLICY COMPARISON")
print("=" * 90)

for label in policies:
    for level in (0, 1):
        name = f"{label}_L{level}"
        r = results[name]

        print(
            f"{name:29s}"
            f" survival={r['full_horizon_survival_count']:2d}/64"
            f" holds={r['final_two_second_tight_hold_count']:2d}/64"
            f" return={r['mean_return']:9.4f}"
            f" pos={r['mean_position_rms_m']:.5f}"
            f" speed={r['mean_speed_rms_mps']:.5f}"
        )

(OUT / "comparison.json").write_text(
    json.dumps(results, indent=2)
)

print("\n[PASS] Action-distribution audit complete")
print("[OUTPUT]", OUT)
