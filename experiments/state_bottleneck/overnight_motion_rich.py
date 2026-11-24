#!/usr/bin/env python3
"""Matched-budget overnight test: standard-reset vs motion-rich DAgger."""

import argparse
import ast
import csv
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import traceback

ROOT = Path("/home/sumukh/AerialCatching")

ENV_FILE = ROOT / "envs/aruco_hover/quadrotor_aruco_hover_env.py"

TEACHER = ROOT / (
    "logs/rsl_rl/CTBR_Hover_Teacher/"
    "2026-06-25_12-41-52/model_749.pt"
)

START = ROOT / "checkpoints/corners_stage2_final_dagger.pt"

COLLECT = ROOT / "rl/collect_dagger_corners256_v1.py"
TRAIN = ROOT / "rl/train_dagger_corners256_v1.py"
EVAL = ROOT / "rl/eval_rslrl.py"

TASK = "Isaac-QuadrotorHoverStudentDistill-v0"

COLLECT_LEVELS = [0.0, 0.5, 1.0]
EVAL_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def make_motion_patch(original):
    tree = ast.parse(original)
    choices = []

    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue

        if fn.name != "_reset_idx":
            continue

        for call in ast.walk(fn):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "write_root_velocity_to_sim"
            ):
                choices.append((fn, call))

    if len(choices) != 1:
        raise RuntimeError(
            f"Expected one root-velocity reset call; found {len(choices)}"
        )

    fn, call = choices[0]

    call_text = ast.get_source_segment(original, call) or ""

    if "default_root" not in call_text:
        raise RuntimeError(
            "Cannot verify reset velocity source. "
            "Refusing to modify environment."
        )

    lines = original.splitlines(keepends=True)
    index = call.lineno - 1

    line = lines[index]
    indent = re.match(r"^\s*", line).group(0)

    if not line.lstrip().startswith(
        "self._robot.write_root_velocity_to_sim("
    ):
        raise RuntimeError("Unexpected velocity-write call structure")

    injection = """
# Controlled motion-rich resets; disabled by default.
import os as _mr_os
if _mr_os.environ.get("CORNERS_MOTION_RICH_RESET") == "1":
    _mr_n = default_root.shape[0]

    _mr_mask = (
        torch.rand((_mr_n, 1), device=default_root.device) < 0.75
    ).to(default_root.dtype)

    _mr_linear = torch.randn(
        (_mr_n, 3),
        device=default_root.device,
        dtype=default_root.dtype,
    ) * 0.28

    _mr_angular = torch.randn(
        (_mr_n, 3),
        device=default_root.device,
        dtype=default_root.dtype,
    ) * 0.18

    default_root[:, 7:10] += (
        _mr_mask * _mr_linear.clamp(-0.50, 0.50)
    )

    default_root[:, 10:13] += (
        _mr_mask * _mr_angular.clamp(-0.35, 0.35)
    )
"""

    inserted = "".join(
        indent + line + "\n"
        for line in injection.strip().splitlines()
    )

    modified = (
        "".join(lines[:index])
        + inserted
        + "".join(lines[index:])
    )

    ast.parse(modified)

    if modified.count("CORNERS_MOTION_RICH_RESET") != 1:
        raise RuntimeError("Reset patch failed validation")

    print(
        f"[PATCH] {fn.name}, velocity write at line {call.lineno}",
        flush=True,
    )

    return modified


def headless_args(path):
    text = path.read_text()

    if (
        "AppLauncher.add_app_launcher_args" in text
        or re.search(
            r'add_argument\(\s*["\']--headless',
            text,
        )
    ):
        return ["--headless"]

    return []


def run_stage(name, command, run_dir, motion=False):
    log = run_dir / f"{name}.log"

    env = os.environ.copy()

    env["PYTHONPATH"] = (
        str(ROOT)
        + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )

    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    env["CORNERS_MOTION_RICH_RESET"] = (
        "1" if motion else "0"
    )

    print("\n" + "=" * 78, flush=True)
    print(f"[STAGE] {name}", flush=True)
    print(
        f"[RESET] {'MOTION-RICH' if motion else 'STANDARD'}",
        flush=True,
    )
    print("[COMMAND]", shlex.join(command), flush=True)
    print("[LOG]", log, flush=True)

    with log.open("w") as f:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        lines = log.read_text(errors="replace").splitlines()

        print(
            f"[FAILED] {name}: return code {result.returncode}",
            flush=True,
        )
        print("\n".join(lines[-45:]), flush=True)

        raise RuntimeError(f"Stage failed: {name}")

    print(f"[PASS] {name}", flush=True)

    return log


def preflight():
    for path in [
        ENV_FILE,
        TEACHER,
        START,
        COLLECT,
        TRAIN,
        EVAL,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    required = [
        (
            COLLECT,
            [
                "--teacher_checkpoint",
                "--student_checkpoint",
                "--output_dir",
                "--round_index",
                "--curriculum_level",
                "--teacher_probability",
                "--controller_hold_steps",
                "--num_samples",
                "--shard_size",
                "--num_envs",
                "--seed",
                "--task",
            ],
        ),
        (
            TRAIN,
            [
                "--data_dir",
                "--output_checkpoint",
                "--resume_checkpoint",
                "--epochs",
                "--batch_size",
                "--device",
                "--seed",
            ],
        ),
        (
            EVAL,
            [
                "--student_checkpoint",
                "--num_envs",
                "--seed",
                "--curriculum_level",
                "--task",
            ],
        ),
    ]

    for path, flags in required:
        source = path.read_text()

        missing = [
            flag for flag in flags
            if flag not in source
        ]

        if missing:
            raise RuntimeError(
                f"{path.name} missing expected arguments: {missing}"
            )

    registration = (
        ROOT / "envs/aruco_hover/__init__.py"
    ).read_text()

    if TASK not in registration:
        raise RuntimeError(
            f"Task registration not found: {TASK}"
        )

    print(
        "[PREFLIGHT PASS] Checkpoints, scripts, "
        "and expected CLI arguments found.",
        flush=True,
    )


def verify_data(directory):
    shards = sorted(directory.rglob("*.pt"))

    if not shards:
        raise RuntimeError(
            f"No training shards produced: {directory}"
        )

    sizes = [p.stat().st_size for p in shards]

    if any(size < 4096 for size in sizes):
        raise RuntimeError(
            f"Suspiciously small training shard: {directory}"
        )

    gb = sum(sizes) / 1e9

    print(
        f"[DATA] {directory.name}: "
        f"{len(shards)} shards, {gb:.2f} GB",
        flush=True,
    )


def parse_evaluation(log):
    text = log.read_text(errors="replace")

    if "STUDENT-ONLY EVALUATION SUMMARY" not in text:
        raise RuntimeError(
            f"Evaluation did not produce summary: {log}"
        )

    def get(pattern):
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    return {
        "initial_episodes": get(
            r"Initial episodes:\s*(\d+)"
        ),
        "initial_timeouts": get(
            r"Initial timeouts:\s*(\d+)"
        ),
        "initial_deaths": get(
            r"Initial deaths:\s*(\d+)"
        ),
        "mean_return": get(
            r"Mean completed return:\s*([\d.\-]+)"
        ),
        "mean_length": get(
            r"Mean completed length:\s*([\d.\-]+)"
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.is_relative_to(ROOT):
        raise RuntimeError(
            "Run directory must be inside AerialCatching"
        )

    status = run_dir / "STATUS.txt"

    preflight()

    original = ENV_FILE.read_text()

    if "CORNERS_MOTION_RICH_RESET" in original:
        raise RuntimeError(
            "Motion-rich patch already present. "
            "Refusing double modification."
        )

    modified = make_motion_patch(original)

    # Keep a recoverable copy even if the machine crashes.
    (run_dir / "environment_before_patch.py").write_text(
        original
    )

    ENV_FILE.write_text(modified)

    try:
        print(
            "[ENV] Temporary guarded motion-reset patch installed.",
            flush=True,
        )

        status.write_text("RUNNING\n")

        checkpoints = {
            "frozen": START,
        }

        # --------------------------------------------------
        # 1. COLLECT + TRAIN TWO MATCHED CONTINUATIONS
        # --------------------------------------------------
        for arm in ["control", "motion"]:
            data_dir = run_dir / "data" / arm
            data_dir.mkdir(parents=True, exist_ok=True)

            for index, level in enumerate(COLLECT_LEVELS):

                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    "rl.collect_dagger_corners256_v1",

                    "--task", TASK,

                    "--teacher_checkpoint", str(TEACHER),
                    "--student_checkpoint", str(START),

                    "--output_dir", str(data_dir),
                    "--round_index", str(100 + index),

                    "--curriculum_level", str(level),
                    "--teacher_probability", "0.75",
                    "--controller_hold_steps", "25",

                    "--num_samples", "60000",
                    "--shard_size", "60000",
                    "--num_envs", "128",

                    "--seed", str(5210 + index),
                ]

                command += headless_args(COLLECT)

                run_stage(
                    f"collect_{arm}_L{index}",
                    command,
                    run_dir,
                    motion=(arm == "motion"),
                )

            verify_data(data_dir)

            output_checkpoint = (
                run_dir / f"corners_tcn_{arm}_continued.pt"
            )

            command = [
                sys.executable,
                "-u",
                "-m",
                "rl.train_dagger_corners256_v1",

                "--data_dir", str(data_dir),

                "--resume_checkpoint", str(START),
                "--output_checkpoint", str(output_checkpoint),

                "--epochs", "12",
                "--batch_size", "512",
                "--device", "cuda:0",
                "--seed", "2048",
            ]

            run_stage(
                f"train_{arm}",
                command,
                run_dir,
            )

            if not output_checkpoint.is_file():
                raise RuntimeError(
                    "Training finished without producing checkpoint: "
                    f"{output_checkpoint}"
                )

            checkpoints[arm] = output_checkpoint

        # --------------------------------------------------
        # 2. MATCHED CLOSED-LOOP EVALUATION
        # --------------------------------------------------
        rows = []

        for regime in ["standard", "motion"]:
            for index, level in enumerate(EVAL_LEVELS):

                for arm in ["motion", "control", "frozen"]:

                    name = (
                        f"eval_{regime}_L{index}_{arm}"
                    )

                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "rl.eval_rslrl",

                        "--task", TASK,

                        "--student_checkpoint",
                        str(checkpoints[arm]),

                        "--num_envs", "64",
                        "--seed", str(9310 + index),
                        "--curriculum_level", str(level),

                        "--use_gru", "false",
                        "--save_prefix", name,
                    ]

                    command += headless_args(EVAL)

                    log = run_stage(
                        name,
                        command,
                        run_dir,
                        motion=(regime == "motion"),
                    )

                    row = {
                        "regime": regime,
                        "level": level,
                        "policy": arm,
                    }

                    row.update(parse_evaluation(log))
                    rows.append(row)

                    # Incremental results: survive interruptions.
                    with (run_dir / "summary.csv").open(
                        "w",
                        newline="",
                    ) as f:

                        writer = csv.DictWriter(
                            f,
                            fieldnames=list(rows[0]),
                        )

                        writer.writeheader()
                        writer.writerows(rows)

                    n = int(row["initial_episodes"])
                    survived = int(row["initial_timeouts"])

                    survival = (
                        100.0 * survived / max(1, n)
                    )

                    print(
                        f"[RESULT] {regime} "
                        f"level={level:.2f} "
                        f"policy={arm} "
                        f"survival={survival:.1f}% "
                        f"return={row['mean_return']} "
                        f"length={row['mean_length']}",
                        flush=True,
                    )

        status.write_text("COMPLETED\n")

        print(
            "\n[DONE] Overnight experiment completed.",
            flush=True,
        )
        print(
            "[RESULTS]", run_dir / "summary.csv",
            flush=True,
        )

    except BaseException as error:
        status.write_text(
            f"FAILED: {type(error).__name__}: {error}\n"
        )
        raise

    finally:
        # Restore exactly the source that existed before this job.
        if ENV_FILE.read_text() == modified:
            ENV_FILE.write_text(original)

            print(
                "[CLEANUP] Original environment source restored.",
                flush=True,
            )
        else:
            print(
                "[WARNING] Environment changed externally; "
                "automatic restoration skipped.",
                flush=True,
            )


def handle_termination(signum, frame):
    raise KeyboardInterrupt(
        "Termination requested; restoring environment."
    )


if __name__ == "__main__":
    signal.signal(
        signal.SIGTERM,
        handle_termination,
    )

    main()
