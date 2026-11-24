from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys

import torch


ROOT = Path("/home/sumukh/AerialCatching")
BASE = ROOT / "experiments/state_bottleneck"

RUN = Path(__file__).resolve().parent

PYTHON = "/home/sumukh/env_isaaclab/bin/python"

START_CKPT = (
    BASE
    / "long_horizon_gae_20261123T032419Z"
    / "model_9.pt"
)

AFTER_CKPT = RUN / "direction_after.pt"

REFERENCE_TS = (
    BASE
    / "precision_ab_control_20261123T041432Z"
    / "policy_initial.ts"
)

EVALUATOR = (
    BASE
    / "action_history_study_20261120T200813Z"
    / "evaluate_action_history_precision_reward.py"
)

POLICY_DIR = RUN / "direction_policies"
POLICY_DIR.mkdir(exist_ok=True)

EVAL_DIR = RUN / "direction_evaluation"
EVAL_DIR.mkdir(exist_ok=True)

SCALES = [-4, -1, 0, 1, 4]
SEEDS = [7513, 8513, 9513]
LEVELS = [0, 1]

for path in (
    START_CKPT,
    AFTER_CKPT,
    REFERENCE_TS,
    EVALUATOR,
):
    if not path.is_file():
        raise FileNotFoundError(path)


# ==========================================================
# 1. Verify exactly one PPO update was accepted.
# ==========================================================

training_log = (RUN / "training.log").read_text(
    errors="replace"
)

accepted = training_log.count(
    "[TRUST GUARD] accepted=True"
)

rejected = training_log.count(
    "[TRUST GUARD] accepted=False"
)

if accepted != 1 or rejected != 0:
    raise RuntimeError(
        "Expected exactly one accepted PPO update; "
        f"accepted={accepted}, rejected={rejected}"
    )

print("[PASS] Exactly one PPO update accepted", flush=True)


# ==========================================================
# 2. Load checkpoint states.
# ==========================================================

before = torch.load(
    START_CKPT,
    map_location="cpu",
    weights_only=True,
)["model_state_dict"]

after = torch.load(
    AFTER_CKPT,
    map_location="cpu",
    weights_only=True,
)["model_state_dict"]

if set(before) != set(after):
    raise RuntimeError(
        "Starting and updated model keys differ"
    )

actor_keys = [
    key
    for key in before
    if key.startswith("actor.")
]

if not actor_keys:
    raise RuntimeError("No actor parameters found")

deltas = {}

for key in actor_keys:
    a = before[key].detach().cpu()
    b = after[key].detach().cpu()

    if a.shape != b.shape:
        raise RuntimeError(
            f"Parameter shape mismatch: {key}"
        )

    deltas[key] = b.float() - a.float()


# ==========================================================
# 3. Match TorchScript parameters to checkpoint parameters.
#
# No guessed key ordering is allowed.
# ==========================================================

def signature(tensor):
    t = tensor.detach().cpu().contiguous()

    data = t.view(torch.uint8).numpy().tobytes()

    return (
        str(t.dtype),
        tuple(t.shape),
        hashlib.sha256(data).hexdigest(),
    )


reference = torch.jit.load(
    str(REFERENCE_TS),
    map_location="cpu",
)

checkpoint_by_signature = defaultdict(list)

for key in actor_keys:
    checkpoint_by_signature[
        signature(before[key])
    ].append(key)


def suffix_score(script_name, checkpoint_name):
    a = script_name.split(".")
    b = checkpoint_name.split(".")

    score = 0

    while (
        score < min(len(a), len(b))
        and a[-1 - score] == b[-1 - score]
    ):
        score += 1

    return score


mapping = {}
unmatched = []
ambiguous = []

for script_name, parameter in reference.named_parameters():

    candidates = checkpoint_by_signature.get(
        signature(parameter),
        [],
    )

    if not candidates:
        unmatched.append(
            (script_name, tuple(parameter.shape))
        )
        continue

    if len(candidates) == 1:
        mapping[script_name] = candidates[0]
        continue

    ranked = sorted(
        (
            suffix_score(script_name, key),
            key,
        )
        for key in candidates
    )

    highest = ranked[-1][0]

    best = [
        key
        for score, key in ranked
        if score == highest
    ]

    if len(best) != 1 or highest == 0:
        ambiguous.append(
            (script_name, candidates)
        )
        continue

    mapping[script_name] = best[0]


print()
print("=" * 75)
print("TORCHSCRIPT PARAMETER MAPPING")
print("=" * 75)

print(
    "Script parameters:",
    len(list(reference.named_parameters())),
)

print("Matched parameters:", len(mapping))
print("Unmatched:", len(unmatched))
print("Ambiguous:", len(ambiguous))

if unmatched:
    print("\nUnmatched parameters:")
    for entry in unmatched[:30]:
        print(" ", entry)

if ambiguous:
    print("\nAmbiguous parameters:")
    for entry in ambiguous[:30]:
        print(" ", entry)

if unmatched or ambiguous:
    raise RuntimeError(
        "Cannot safely export direction policies. "
        "TorchScript/checkpoint parameter mapping "
        "must be exact."
    )

if len(set(mapping.values())) != len(mapping):
    raise RuntimeError(
        "Multiple TorchScript parameters mapped to "
        "the same checkpoint parameter"
    )

if not mapping:
    raise RuntimeError("Empty TorchScript mapping")

print("[PASS] Exact parameter correspondence established")


# ==========================================================
# 4. Measure actual policy-parameter changes.
# ==========================================================

print()
print("=" * 75)
print("ACTOR UPDATE MAGNITUDE")
print("=" * 75)

for group in (
    "actor.estimator.",
    "actor.head.",
):

    keys = [
        key
        for key in actor_keys
        if key.startswith(group)
    ]

    ref_sq = sum(
        before[key].double().square().sum().item()
        for key in keys
    )

    delta_sq = sum(
        deltas[key].double().square().sum().item()
        for key in keys
    )

    relative = (
        delta_sq ** 0.5
        / max(ref_sq ** 0.5, 1e-12)
    )

    print(
        group,
        "relative_change=",
        f"{relative:.8e}",
    )

changed_script_params = sum(
    torch.count_nonzero(deltas[key]).item() > 0
    for key in mapping.values()
)

if changed_script_params == 0:
    raise RuntimeError(
        "PPO did not change any exported actor parameters"
    )

print(
    "[PASS] Changed deployed parameter tensors:",
    changed_script_params,
)


# ==========================================================
# 5. Export policies along and against actual update.
# ==========================================================

policy_paths = {}

for scale in SCALES:

    if scale == 0:
        policy_paths[scale] = REFERENCE_TS
        continue

    model = torch.jit.load(
        str(REFERENCE_TS),
        map_location="cpu",
    )

    with torch.no_grad():

        for name, parameter in model.named_parameters():

            checkpoint_key = mapping[name]

            reference_value = parameter.detach().clone()

            delta = deltas[checkpoint_key].to(
                dtype=parameter.dtype
            )

            parameter.copy_(
                reference_value + scale * delta
            )

    filename = (
        f"direction_plus_{scale}.ts"
        if scale > 0
        else f"direction_minus_{abs(scale)}.ts"
    )

    destination = POLICY_DIR / filename

    torch.jit.save(
        model,
        str(destination),
    )

    # Reopen the serialized policy and verify the actual
    # parameter perturbations survived TorchScript export.
    verified = torch.jit.load(
        str(destination),
        map_location="cpu",
    )

    ref_parameters = dict(
        reference.named_parameters()
    )

    verified_parameters = dict(
        verified.named_parameters()
    )

    if set(ref_parameters) != set(verified_parameters):
        raise RuntimeError(
            f"Serialized parameter keys changed: {scale}"
        )

    for name, parameter in verified_parameters.items():

        ref = ref_parameters[name].detach()
        delta = deltas[mapping[name]].to(
            dtype=ref.dtype
        )

        expected = ref + scale * delta

        if not torch.equal(parameter.detach(), expected):
            raise RuntimeError(
                f"Exported weight mismatch: "
                f"scale={scale}, parameter={name}"
            )

    policy_paths[scale] = destination

    print(
        f"[PASS] Exported scale={scale}: {destination}",
        flush=True,
    )


# ==========================================================
# 6. Evaluate using the existing precision-reward evaluator.
# ==========================================================

def evaluate_level(level, gpu):

    results = {}

    for seed in SEEDS:

        for scale in SCALES:

            label = (
                f"direction_{scale:+d}"
                f"_seed{seed}_L{level}"
            )

            destination = EVAL_DIR / label

            log = EVAL_DIR / f"{label}.log"

            command = [
                PYTHON,
                "-u",
                str(EVALUATOR),
                "--task",
                "Isaac-QuadrotorHoverStudentDistill-v0",
                "--policy",
                str(policy_paths[scale]),
                "--variant",
                label,
                "--level",
                str(level),
                "--num-envs",
                "64",
                "--steps",
                "500",
                "--seed",
                str(seed),
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
                f"[GPU {gpu}] {label}",
                flush=True,
            )

            with log.open("w") as handle:

                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )

            summary_path = (
                destination / "summary.json"
            )

            episodes_path = (
                destination / "episodes.csv"
            )

            if (
                process.returncode != 0
                or not summary_path.is_file()
                or not episodes_path.is_file()
            ):
                print(
                    log.read_text(
                        errors="replace"
                    )[-5000:],
                    flush=True,
                )

                raise RuntimeError(
                    f"Evaluation failed: {label}"
                )

            with episodes_path.open(newline="") as f:
                episodes = list(csv.DictReader(f))

            if len(episodes) != 64:
                raise RuntimeError(
                    f"Expected 64 episodes: {label}"
                )

            results[(scale, seed, level)] = episodes

    return results


print()
print("=" * 75)
print("RUNNING CLOSED-LOOP DIRECTION EVALUATIONS")
print("=" * 75)

# Keep each level on one GPU, so matched evaluations
# for the same level use the same hardware.
with ThreadPoolExecutor(max_workers=2) as pool:

    future_l0 = pool.submit(evaluate_level, 0, 0)
    future_l1 = pool.submit(evaluate_level, 1, 1)

    results = {}

    results.update(future_l0.result())
    results.update(future_l1.result())


# ==========================================================
# 7. Aggregate results.
# ==========================================================

def get_rows(scale, level=None):

    rows = []

    for seed in SEEDS:
        for lv in LEVELS:

            if level is not None and lv != level:
                continue

            rows.extend(
                results[(scale, seed, lv)]
            )

    return rows


def aggregate(rows):

    n = len(rows)

    return {
        "episodes": n,

        "holds": sum(
            int(float(r["final_two_second_tight_hold"]))
            for r in rows
        ),

        "survival": sum(
            float(r["episode_length"]) >= 499
            for r in rows
        ),

        "return": statistics.fmean(
            float(r["return_sum"])
            for r in rows
        ),

        "tight_fraction": statistics.fmean(
            float(r["tight_hover_fraction"])
            for r in rows
        ),

        "position_rms": statistics.fmean(
            float(r["position_rms_m"])
            for r in rows
        ),

        "speed_rms": statistics.fmean(
            float(r["speed_rms_mps"])
            for r in rows
        ),
    }


all_metrics = {}

print()
print("=" * 100)
print("PPO ACTOR UPDATE-DIRECTION RESULTS")
print("=" * 100)

for level in (0, 1, None):

    label = (
        f"LEVEL {level}"
        if level is not None
        else "COMBINED"
    )

    print("\n" + "-" * 100)
    print(label)
    print("-" * 100)

    baseline = aggregate(get_rows(0, level))

    for scale in SCALES:

        metrics = aggregate(
            get_rows(scale, level)
        )

        key = (
            f"L{level}_scale{scale:+d}"
            if level is not None
            else f"combined_scale{scale:+d}"
        )

        all_metrics[key] = metrics

        print(
            f"scale={scale:+3d} | "
            f"survival={int(metrics['survival']):3d}/"
            f"{metrics['episodes']} | "
            f"holds={metrics['holds']:3d}/"
            f"{metrics['episodes']} | "
            f"return={metrics['return']:8.3f} | "
            f"tight={metrics['tight_fraction']:.4f} | "
            f"pos={metrics['position_rms']:.5f} | "
            f"speed={metrics['speed_rms']:.5f}"
        )

        if scale != 0:

            print(
                f"          "
                f"delta_holds="
                f"{metrics['holds'] - baseline['holds']:+d} "
                f"delta_return="
                f"{metrics['return'] - baseline['return']:+.4f} "
                f"delta_pos="
                f"{metrics['position_rms'] - baseline['position_rms']:+.6f} "
                f"delta_speed="
                f"{metrics['speed_rms'] - baseline['speed_rms']:+.6f}"
            )


# ==========================================================
# 8. Report results separately by seed.
# ==========================================================

print()
print("=" * 100)
print("PER-SEED COMPARISON")
print("=" * 100)

for level in LEVELS:

    for seed in SEEDS:

        print(
            f"\nSeed={seed} | Level={level}"
        )

        for scale in SCALES:

            metrics = aggregate(
                results[(scale, seed, level)]
            )

            print(
                f"  scale={scale:+3d}: "
                f"holds={metrics['holds']:2d}/64 "
                f"return={metrics['return']:.3f} "
                f"pos={metrics['position_rms']:.5f} "
                f"speed={metrics['speed_rms']:.5f}"
            )


( RUN / "direction_results.json" ).write_text(
    json.dumps(
        {
            "training_seed": 8700,
            "evaluation_seeds": SEEDS,
            "scales": SCALES,
            "metrics": all_metrics,
        },
        indent=2,
    )
)

print()
print("=" * 100)
print("[PASS] PPO direction experiment completed")
print("[RESULTS]", RUN / "direction_results.json")
print("=" * 100)
