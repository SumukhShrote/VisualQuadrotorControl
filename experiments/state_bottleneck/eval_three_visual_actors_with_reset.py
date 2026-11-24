#!/usr/bin/env python3
"""Closed-loop evaluation for corners-only TorchScript policies."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path("experiments/state_bottleneck")
SUITE = ROOT / "action_transformer_suite_20261120T005220Z"


def launch_suite():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Isaac-QuadrotorHoverStudent-v0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args(sys.argv[2:])

    output = ROOT / (
        "closed_loop_transformers_"
        + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    output.mkdir(parents=True, exist_ok=False)

    def worker(gpu, jobs):
        results = []

        for variant, level in jobs:
            name = f"{variant}_level{level}"
            checkpoint = SUITE / variant / "policy.ts"

            assert checkpoint.is_file(), checkpoint

            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--policy", str(checkpoint),
                "--variant", variant,
                "--level", str(level),
                "--num-envs", str(args.num_envs),
                "--seed", str(args.seed),
                "--steps", str(args.steps),
                "--out", str(output / name),
                "--device", f"cuda:{gpu}",
            ]

            if args.task:
                command += ["--task", args.task]

            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)

            env["PYTHONPATH"] = (
                os.getcwd()
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )

            log = output / f"{name}.log"

            print(
                f"[START] GPU {gpu}: {name}",
                flush=True,
            )

            with log.open("w") as handle:
                result = subprocess.run(
                    command,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )

            print(
                f"[DONE] GPU {gpu}: {name}, "
                f"returncode={result.returncode}",
                flush=True,
            )

            if result.returncode:
                print(
                    "\n".join(
                        log.read_text(
                            errors="replace"
                        ).splitlines()[-25:]
                    ),
                    flush=True,
                )

            # Isaac Sim may return code 0 even if no result
            # file was produced. Never count that as success.
            summary = output / name / "summary.json"

            if result.returncode == 0 and not summary.is_file():
                print(
                    f"[ERROR] {name}: missing summary.json",
                    flush=True,
                )
                result.returncode = 97

            results.append(
                (name, result.returncode)
            )

        return results

    with ThreadPoolExecutor(max_workers=2) as pool:
        gpu0 = pool.submit(
            worker,
            0,
            [
                ("direct", 0),
                ("joint", 0),
                ("direct", 1),
                ("joint", 1),
            ],
        )

        gpu1 = pool.submit(
            worker,
            1,
            [
                ("pretrained", 0),
                ("pretrained", 1),
            ],
        )

        statuses = gpu0.result() + gpu1.result()

    print("\n========== CLOSED-LOOP RESULTS ==========")

    for name, code in statuses:
        summary_path = (
            output / name / "summary.json"
        )

        if code or not summary_path.exists():
            print(
                f"{name}: FAILED (code={code})"
            )
            continue

        result = json.loads(
            summary_path.read_text()
        )

        print(
            f"{name:<22}"
            f"survival={result['survival_rate']:.1%}  "
            f"length={result['mean_length']:.1f}  "
            f"pos_RMS={result['mean_position_rms_m']:.3f}m  "
            f"speed_RMS={result['mean_speed_rms_mps']:.3f}m/s  "
            f"tight_hold="
            f"{result['final_two_second_tight_hold_count']}"
            f"/{result['num_envs']}"
        )

    print("\n[RESULTS]", output)

    if any(code for _, code in statuses):
        raise RuntimeError(
            "Some evaluations failed. Logs are preserved."
        )


def run_episode():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--task", default="Isaac-QuadrotorHoverStudent-v0")
    parser.add_argument("--level", type=float, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4300)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--out", required=True)

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import envs
    import isaaclab_tasks

    from isaaclab_tasks.utils import parse_env_cfg

    # Discover the registered corners-only student task.
    candidates = []
    hover_tasks = []

    for name, spec in gym.envs.registry.items():
        lower = name.lower()

        if (
            "hover" in lower
            and (
                "quadrotor" in lower
                or "aruco" in lower
            )
        ):
            hover_tasks.append(name)

            if (
                "student" in lower
                or "student" in str(spec.kwargs).lower()
            ):
                candidates.append(name)

    if args.task:
        task = args.task
    elif len(candidates) == 1:
        task = candidates[0]
    else:
        print("[AVAILABLE HOVER TASKS]", hover_tasks)
        print("[STUDENT CANDIDATES]", candidates)

        raise RuntimeError(
            "Could not identify a unique student task. "
            "Specify its registered ID using --task."
        )

    print("[TASK]", task, flush=True)

    cfg = parse_env_cfg(
        task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=True,
    )

    assert hasattr(
        cfg, "forced_curriculum_level"
    ), "Selected task lacks fixed-curriculum support."

    cfg.forced_curriculum_level = args.level

    if hasattr(cfg, "eval_mode"):
        cfg.eval_mode = True

    if hasattr(cfg, "enable_onboard_camera"):
        cfg.enable_onboard_camera = False

    if hasattr(cfg, "seed"):
        cfg.seed = args.seed

    env = None

    try:
        print("[TRACE 1] Before gym.make", flush=True)
        env = gym.make(task, cfg=cfg)
        print("[TRACE 2] Environment created", flush=True)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # The Isaac Lab environment was initialized by gym.make.
        # Follow the existing AerialCatching evaluator contract:
        # obtain its current observations without calling reset again.
        # Gymnasium's OrderEnforcing wrapper requires reset()
        # before the first step(). The configuration is already
        # seeded using cfg.seed.
        print("[TRACE 3] Before env.reset()", flush=True)

        try:
            observations, extras = env.reset()

            if not (
                isinstance(observations, dict)
                and "policy" in observations
            ):
                if (
                    isinstance(extras, dict)
                    and isinstance(
                        extras.get("observations"), dict
                    )
                ):
                    observations = extras["observations"]
                else:
                    raise RuntimeError(
                        "Unexpected reset observation structure: "
                        f"{type(observations)}"
                    )

        except BaseException as exc:
            import traceback

            print(
                "[RESET EXCEPTION]",
                type(exc).__name__,
                repr(exc),
                flush=True,
            )

            traceback.print_exc()

            raise RuntimeError(
                "Environment reset failed"
            ) from exc

        print(
            "[TRACE 4] Initial observations obtained:",
            {
                k: tuple(v.shape)
                for k, v in observations.items()
                if hasattr(v, "shape")
            },
            flush=True,
        )

        assert isinstance(observations, dict)
        assert "policy" in observations
        assert "teacher" in observations

        assert observations["policy"].shape == (
            args.num_envs,
            256,
        ), (
            "Expected 256-D corner observations, got "
            f"{observations['policy'].shape}"
        )

        assert observations["teacher"].shape[1] == 20

        device = observations["policy"].device

        print("[TRACE 5] Before TorchScript load", flush=True)
        policy = torch.jit.load(
            args.policy,
            map_location=device,
        ).eval()
        print("[TRACE 6] TorchScript loaded", flush=True)

        with torch.no_grad():
            test = policy(
                torch.zeros(
                    args.num_envs,
                    256,
                    device=device,
                )
            )

        print(
            "[TRACE 7] Policy test inference complete",
            tuple(test.shape),
            flush=True,
        )
        assert test.shape == (
            args.num_envs,
            4,
        )

        assert torch.isfinite(test).all()

        n = args.num_envs
        horizon = args.steps

        position = np.full(
            (n, horizon),
            np.nan,
            dtype=np.float32,
        )

        speed = np.full_like(
            position,
            np.nan,
        )

        angular_speed = np.full_like(
            position,
            np.nan,
        )

        lengths = np.full(
            n,
            horizon,
            dtype=int,
        )

        terminated = np.zeros(
            n, dtype=bool
        )

        timeout = np.zeros(
            n, dtype=bool
        )

        finished = np.zeros(
            n, dtype=bool
        )

        returns = np.zeros(
            n,
            dtype=np.float64,
        )

        print("[TRACE 8] Entering rollout", flush=True)
        for step in range(horizon):
            if step < 3:
                print(
                    f"[TRACE 9] Step {step}: before policy",
                    flush=True,
                )
            print(
                f"[TRACE A] Step {step}: before state read",
                flush=True,
            )
            state = observations["teacher"]
            print("[TRACE B] State retrieved", flush=True)

            live = ~finished

            pos_now = torch.linalg.vector_norm(
                state[:, 0:3],
                dim=-1,
            ).cpu().numpy()

            speed_now = torch.linalg.vector_norm(
                state[:, 3:6],
                dim=-1,
            ).cpu().numpy()

            omega_now = torch.linalg.vector_norm(
                state[:, 6:9],
                dim=-1,
            ).cpu().numpy()

            position[live, step] = pos_now[live]
            speed[live, step] = speed_now[live]
            angular_speed[live, step] = omega_now[live]

            print("[TRACE C] Before policy inference", flush=True)

            with torch.no_grad():
                actions = policy(
                    observations["policy"]
                ).clamp(-1.0, 1.0)

                print(
                    "[TRACE D] Policy inference complete",
                    "action=", actions[0].cpu().tolist(),
                    "finite=", bool(torch.isfinite(actions).all()),
                    flush=True,
                )

                print("[TRACE E] Before env.step", flush=True)

                (
                    observations,
                    rewards,
                    term,
                    trunc,
                    _,
                ) = env.step(actions)

                print("[TRACE F] env.step returned", flush=True)

            if step < 3:
                print(
                    f"[TRACE 10] Step {step}: env.step complete",
                    flush=True,
                )

            term = (
                term.reshape(-1)
                .cpu()
                .numpy()
                .astype(bool)
            )

            trunc = (
                trunc.reshape(-1)
                .cpu()
                .numpy()
                .astype(bool)
            )

            reward = (
                rewards.reshape(-1)
                .cpu()
                .numpy()
            )

            returns[live] += reward[live]

            just_finished = live & (term | trunc)

            terminated[just_finished] = term[
                just_finished
            ]

            timeout[just_finished] = (
                trunc[just_finished]
                & ~term[just_finished]
            )

            lengths[just_finished] = step + 1
            finished[just_finished] = True

            if (step + 1) % 100 == 0:
                print(
                    f"[STEP] {step+1}/{horizon} "
                    f"completed={finished.sum()}/{n}",
                    flush=True,
                )

            if finished.all():
                break

        rows = []

        for i in range(n):
            length = int(lengths[i])

            p = position[i, :length]
            v = speed[i, :length]
            w = angular_speed[i, :length]

            tight = (
                (p <= 0.10)
                & (v <= 0.20)
            )

            near = (
                (p <= 0.20)
                & (v <= 0.20)
            )

            hold = bool(
                timeout[i]
                and length >= 100
                and tight[-100:].all()
            )

            if length >= 25:
                streaks = np.convolve(
                    near.astype(np.int32),
                    np.ones(25, dtype=np.int32),
                    mode="valid",
                )

                converged = bool(
                    streaks.max() >= 25
                )
            else:
                converged = False

            rows.append({
                "env_id": i,
                "length": length,
                "timeout": int(timeout[i]),
                "terminated": int(terminated[i]),
                "censored": int(not finished[i]),
                "return_sum": float(returns[i]),
                "position_rms_m": float(
                    np.sqrt(np.mean(p ** 2))
                ),
                "speed_rms_mps": float(
                    np.sqrt(np.mean(v ** 2))
                ),
                "angular_speed_rms_radps": float(
                    np.sqrt(np.mean(w ** 2))
                ),
                "within_20cm_fraction": float(
                    np.mean(p <= 0.20)
                ),
                "tight_hover_fraction": float(
                    np.mean(tight)
                ),
                "converged": int(converged),
                "final_two_second_tight_hold": int(hold),
            })

        def average(key):
            return float(
                np.mean([
                    row[key] for row in rows
                ])
            )

        summary = {
            "policy": args.variant,
            "task": task,
            "curriculum_level": args.level,
            "seed": args.seed,
            "num_envs": n,
            "max_steps": horizon,
            "completed": int(finished.sum()),
            "timeouts": int(timeout.sum()),
            "terminations": int(terminated.sum()),
            "censored": int((~finished).sum()),
            "survival_rate": average("timeout"),
            "mean_length": average("length"),
            "mean_return": average("return_sum"),
            "mean_position_rms_m": average(
                "position_rms_m"
            ),
            "mean_speed_rms_mps": average(
                "speed_rms_mps"
            ),
            "mean_within_20cm_fraction": average(
                "within_20cm_fraction"
            ),
            "mean_tight_hover_fraction": average(
                "tight_hover_fraction"
            ),
            "convergence_rate": average("converged"),
            "final_two_second_tight_hold_count": int(
                sum(
                    row["final_two_second_tight_hold"]
                    for row in rows
                )
            ),
        }

        print("[TRACE 11] Rollout complete; saving", flush=True)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)

        with (out / "episodes.csv").open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
            )

            writer.writeheader()
            writer.writerows(rows)

        (out / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        print(
            "[RESULT]",
            json.dumps(summary),
            flush=True,
        )

    except BaseException as exc:
        import traceback

        print(
            "[CAPTURED EXCEPTION]",
            type(exc).__name__,
            repr(exc),
            flush=True,
        )

        traceback.print_exc()

        raise RuntimeError(
            "Closed-loop evaluation stopped unexpectedly"
        ) from exc

    finally:
        print("[TRACE 12] Closing simulation", flush=True)
        if env is not None:
            env.close()

        simulation_app.close()


if __name__ == "__main__":
    if (
        len(sys.argv) > 1
        and sys.argv[1] == "suite"
    ):
        launch_suite()
    else:
        run_episode()
