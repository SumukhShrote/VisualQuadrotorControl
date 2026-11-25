
"""Reference frozen-representation DAgger training entry point."""

import argparse
from pathlib import Path


def main():

    from isaaclab.app import AppLauncher

    P = argparse.ArgumentParser()

    P.add_argument("--out", required=True)

    P.add_argument("--num-envs", type=int, default=512)

    P.add_argument("--pose-updates", type=int, default=12000)

    P.add_argument("--rounds", type=int, default=24)
    P.add_argument("--train-representation", action="store_true")
    P.add_argument("--representation-lr", type=float, default=1e-5)
    P.add_argument("--head-lr", type=float, default=1e-4)
    P.add_argument("--batch", type=int, default=2048)
    P.add_argument("--pose-checkpoint", default=None)

    P.add_argument("--rollout-steps", type=int, default=192)

    P.add_argument("--updates-per-round", type=int, default=180)

    P.add_argument("--seed", type=int, default=14001)
    P.add_argument("--base-checkpoint", default=None)
    P.add_argument("--base-policy", default=None)

    AppLauncher.add_app_launcher_args(P)

    args = P.parse_args()

    if bool(args.base_checkpoint) != bool(args.base_policy):
        P.error("Supply both base checkpoint and base policy")
    args.headless = True
    args.enable_cameras = False

    launcher = AppLauncher(args)

    try:
        from rl import pose_attention_runtime as runtime

        runtime.configure(args)

        if args.base_checkpoint:
            runtime.BASE = Path(args.base_checkpoint).resolve(strict=True)
            runtime.REFERENCE = Path(args.base_policy).resolve(strict=True)
        runtime.train_main()

    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
