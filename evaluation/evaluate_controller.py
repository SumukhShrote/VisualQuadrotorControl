
"""Independent closed-loop evaluation of reference controllers."""

import argparse
import json


def main():

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()

    parser.add_argument("--out", required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=14001)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--level", type=float, default=1.)

    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["original", "expanded"],
        default=["original", "expanded"],
    )

    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["baseline", "student", "teacher", "candidate"],
        default=["baseline", "student", "teacher"],
    )

    parser.add_argument(
        "--require-reference-success",
        action="store_true",
    )

    parser.add_argument("--candidate-checkpoint")
    parser.add_argument("--candidate-base")
    parser.add_argument("--candidate-policy")

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    candidate_paths = (
        args.candidate_checkpoint,
        args.candidate_base,
        args.candidate_policy,
    )

    if any(candidate_paths) and not all(candidate_paths):
        parser.error("Supply all three candidate paths")

    if "candidate" in args.policies and not all(candidate_paths):
        parser.error("Candidate paths are required")
    args.headless = True
    args.enable_cameras = False

    launcher = AppLauncher(args)

    env = None

    try:

        import torch
        import gymnasium as gym
        import envs

        from isaaclab_tasks.utils import parse_env_cfg

        from envs.aruco_hover.quadrotor_aruco_hover_env import (
            ArucoHoverTeacherEnv,
        )

        from rl import pose_attention_runtime as runtime

        from utils.checkpoint_utils import (
            verify_reference_artifacts,
            load_reference_controller,
            load_deployment_policy,
        )

        runtime.configure(args)

        count = verify_reference_artifacts()

        print("[PASS] Verified reference artifacts:", count)

        checkpoint = torch.load(
            runtime.REF / "pose_attention/checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )

        assert checkpoint["round"] == 15

        student = load_reference_controller(
            device=args.device,
            train_head=False,
        )

        exported = load_deployment_policy(
            device=args.device,
        )

        candidate = candidate_export = None

        if "candidate" in args.policies:
            candidate = load_reference_controller(
                checkpoint=args.candidate_checkpoint,
                base_checkpoint=args.candidate_base,
                device=args.device,
                train_head=False,
            )

            candidate_export = load_deployment_policy(
                checkpoint=args.candidate_policy,
                device=args.device,
            )

        teacher = runtime.teacher_load()

        baseline = torch.jit.load(
            str(runtime.REFERENCE),
            map_location=args.device,
        ).eval()

        runtime.patch_reset(ArucoHoverTeacherEnv)

        runtime.RESET["scale"] = 1.0

        cfg = parse_env_cfg(
            runtime.TASK,
            device=args.device,
            num_envs=args.num_envs,
            use_fabric=True,
        )

        cfg.forced_curriculum_level = args.level
        cfg.enable_onboard_camera = False
        cfg.visual_dr_enabled = False
        cfg.debug_vis = False

        if hasattr(cfg, "eval_mode"):
            cfg.eval_mode = True

        cfg.episode_length_s = args.duration

        env = gym.make(runtime.TASK, cfg=cfg)

        actual_duration = (
            env.unwrapped.max_episode_length
            * env.unwrapped.step_dt
        )

        if abs(actual_duration - args.duration) > .05:
            raise RuntimeError("Incorrect episode duration")

        obs, h = runtime.reset(
            env, args.level, False, 36000
        )

        with torch.no_grad():

            u = student(obs["policy"], h)
            reference_u = exported(obs["policy"], h)

            torch.testing.assert_close(
                u,
                reference_u,
                atol=1e-5,
                rtol=1e-4,
            )

        print(
            "[PASS] Canonical eager policy matches export:",
            float((u - reference_u).abs().max()),
        )

        if candidate is not None:
            with torch.no_grad():
                eager = candidate(obs["policy"], h)
                scripted = candidate_export(obs["policy"], h)

                torch.testing.assert_close(
                    eager, scripted, atol=1e-5, rtol=1e-4
                )

            print(
                "[PASS] Candidate eager/export error:",
                float((eager - scripted).abs().max())
            )

        seeds = args.seeds

        if seeds is None:
            seeds = (
                [36001, 36002]
                if args.duration == 10.
                else [37001]
            )

        policies = {
            "baseline": baseline,
            "student": student,
            "teacher": None,
            "candidate": candidate,
        }

        results = []

        for seed in seeds:

            for domain in args.domains:

                expanded = domain == "expanded"

                for kind in args.policies:

                    print(
                        "[VALIDATING]",
                        kind,
                        "domain=", domain,
                        "seed=", seed,
                        flush=True,
                    )

                    with torch.no_grad():

                        result = runtime.validate_flight(
                            env,
                            policies[kind],
                            teacher,
                            kind,
                            args.level,
                            expanded,
                            seed,
                        )

                    results.append(result)

        report = {
            "implementation": "canonical_eager_controller",
            "candidate_paths": list(candidate_paths)
                if candidate is not None else None,
            "results": results,
        }

        (runtime.OUT / "evaluation_results.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        if args.require_reference_success:

            student_results = [
                row for row in results
                if row["policy"] == "student"
            ]

            if not student_results:
                raise RuntimeError("No student evaluation results")

            for row in student_results:

                assert row["survival_count"] == args.num_envs
                assert row["hold_count"] == args.num_envs

                if args.duration >= 60.:
                    assert row["continuous_hold_after_10s"] == 1.

            print("[PASS] Reference performance reproduction gate")

        (runtime.OUT / "COMPLETE").write_text(
            "validation complete\n"
        )

    finally:

        if env is not None:
            env.close()

        launcher.app.close()


if __name__ == "__main__":
    main()
