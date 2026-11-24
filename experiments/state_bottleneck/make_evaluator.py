from pathlib import Path

source = Path("rl/eval_rslrl.py")
output = Path("experiments/state_bottleneck/eval_closed_loop.py")

if output.exists():
    raise FileExistsError(
        f"{output} already exists. Preserving existing file."
    )

text = source.read_text()

# -------------------------------------------------------
# 1. Add evaluation mode
# -------------------------------------------------------

anchor = 'parser.add_argument("--student_checkpoint", type=str, default=None)'

addition = '''
parser.add_argument(
    "--state_bottleneck_mode",
    choices=["oracle", "visual"],
    required=True,
)
'''

if text.count(anchor) != 1:
    raise RuntimeError("Could not locate argument insertion point")

text = text.replace(anchor, anchor + "\n" + addition, 1)

# -------------------------------------------------------
# 2. Load frozen controllers before evaluation
# -------------------------------------------------------

loop_anchor = (
    "    with torch.no_grad():\n"
    "        while simulation_app.is_running():"
)

setup = '''
    # ===================================================
    # STATE BOTTLENECK EXPERIMENT
    # ===================================================

    if args_cli.student_checkpoint is None:
        raise RuntimeError(
            "Pass the historical Stage-3 student checkpoint "
            "to initialize the corners-only evaluator."
        )

    root = Path("/home/sumukh/AerialCatching")

    oracle_path = (
        root / "experiments/state_bottleneck/results/"
        "oracle13_teacher_bc.pt"
    )

    oracle_checkpoint = torch.load(
        oracle_path,
        map_location="cpu",
        weights_only=False,
    )

    oracle_controller = torch.nn.Sequential(
        torch.nn.Linear(13, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 4),
        torch.nn.Tanh(),
    ).to(device)

    # Training checkpoint uses the OracleMLP.net prefix.
    oracle_state_dict = {
        key.removeprefix("net."): value
        for key, value in oracle_checkpoint["model"].items()
    }

    oracle_controller.load_state_dict(
        oracle_state_dict,
        strict=True,
    )

    oracle_controller.eval()

    oracle_mean = oracle_checkpoint[
        "feature_mean"
    ].to(device)

    oracle_std = oracle_checkpoint[
        "feature_std"
    ].to(device)

    indices = [
        0, 1, 2, 3, 4, 5,
        9, 10, 11, 12, 13, 14, 19,
    ]

    index_tensor = torch.tensor(
        indices,
        dtype=torch.long,
        device=device,
    )

    visual_policy = None
    auxiliary_head = None

    if args_cli.state_bottleneck_mode == "visual":

        aux_path = (
            root / "checkpoints/corners_tcn_student_v1/"
            "stage4_aux13_ft/stage4_aux13_ft_best.pt"
        )

        aux_checkpoint = torch.load(
            aux_path,
            map_location="cpu",
            weights_only=False,
        )

        if list(aux_checkpoint["auxiliary_target_indices"]) != indices:
            raise RuntimeError("AUX13 feature order mismatch")

        visual_policy = ppo_runner.alg.policy

        visual_policy.student.load_state_dict(
            aux_checkpoint["student_state_dict"],
            strict=True,
        )

        visual_policy.eval()

        if not hasattr(
            visual_policy,
            "encode_temporal_state",
        ):
            raise RuntimeError(
                "Policy lacks encode_temporal_state()"
            )

        auxiliary_head = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 13),
        ).to(device)

        auxiliary_state_dict = {
            key.removeprefix("network."): value
            for key, value in aux_checkpoint[
                "auxiliary_state_head_dict"
            ].items()
        }

        auxiliary_head.load_state_dict(
            auxiliary_state_dict,
            strict=True,
        )

        auxiliary_head.eval()

        aux_mean = aux_checkpoint[
            "auxiliary_target_mean"
        ].to(device)

        aux_std = aux_checkpoint[
            "auxiliary_target_std"
        ].to(device)

    if "teacher" not in obs_dict:
        raise RuntimeError(
            "Environment did not expose the 20-D "
            "teacher observation."
        )

    if obs_dict["teacher"].shape[-1] != 20:
        raise RuntimeError(
            "Unexpected privileged observation dimension."
        )

    print(
        "[STATE BOTTLENECK] mode=",
        args_cli.state_bottleneck_mode,
        flush=True,
    )

    print(
        "[STATE BOTTLENECK] oracle checkpoint=",
        oracle_path,
        flush=True,
    )

    # ===================================================
'''

if text.count(loop_anchor) != 1:
    raise RuntimeError("Could not locate evaluation loop")

text = text.replace(
    loop_anchor,
    setup + "\n" + loop_anchor,
    1,
)

# -------------------------------------------------------
# 3. Replace only the action source
# -------------------------------------------------------

old_action = '''                raw_actions = agent(
                    obs_dict["policy"]
                )'''

new_action = '''                true_state13 = obs_dict[
                    "teacher"
                ].index_select(1, index_tensor)

                if args_cli.state_bottleneck_mode == "oracle":
                    controller_state = true_state13

                else:
                    corner_history = obs_dict["policy"]

                    latent = visual_policy.encode_temporal_state(
                        corner_history
                    )

                    estimated_normalized = auxiliary_head(
                        latent
                    )

                    controller_state = (
                        estimated_normalized * aux_std
                        + aux_mean
                    )

                if not torch.isfinite(controller_state).all():
                    raise RuntimeError(
                        "Non-finite physical-state estimate"
                    )

                controller_input = (
                    (controller_state - oracle_mean)
                    / oracle_std
                ).clamp(-15.0, 15.0)

                raw_actions = oracle_controller(
                    controller_input
                )

                if steps < 3:
                    print(
                        "[BOTTLENECK]",
                        "step=", steps,
                        "mode=", args_cli.state_bottleneck_mode,
                        "true13=", true_state13[0].detach().cpu().tolist(),
                        "input13=", controller_state[0].detach().cpu().tolist(),
                        "action=", raw_actions[0].detach().cpu().tolist(),
                        flush=True,
                    )'''

if text.count(old_action) != 1:
    raise RuntimeError(
        "Could not locate the original action call"
    )

text = text.replace(old_action, new_action, 1)

output.write_text(text)

print("[CREATED]", output)
print("[PASS] Original evaluator unchanged")
