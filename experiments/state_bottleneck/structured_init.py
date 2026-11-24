import argparse
import copy
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Load the policy without importing the Isaac Sim environment.
import sys

_policy_file = (
    Path(__file__).resolve().parents[2]
    / "rl/corners_asymmetric_ppo/policy.py"
)

_spec = importlib.util.spec_from_file_location(
    "offline_corners_policy", _policy_file
)

if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load {_policy_file}")

_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

CornersAsymmetricActorCritic = (
    _module.CornersAsymmetricActorCritic
)



ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / (
    "experiments/state_bottleneck/data/"
    "pooled_student_epoch70_all_levels.pt"
)

PPO400 = ROOT / (
    "logs/rsl_rl/CTBR_Hover_Corners_Asymmetric_Scratch_PPO/"
    "2026-08-06_08-30-34_"
    "corners_tcn_asymmetric_scratch_runh_level0_seed42_mb64_v1/"
    "model_400.pt"
)

AUX_SCRIPT = ROOT / (
    "experiments/corners_tcn_student_v1/train_aux13_student.py"
)

OUT = ROOT / "experiments/state_bottleneck/results"

STATE_INDICES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 19]


class StructuredCornersActorCritic(CornersAsymmetricActorCritic):
    """
    Corners-only actor, privileged critic.

    Actor:
        corners -> TCN -> h
        h -> explicitly supervised state
        h -> learned latent
        (state, latent) -> residual action
        h -> original base actor

    PPO distribution and critic are inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.state_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ELU(),
            nn.Linear(256, 13),
        )

        self.latent_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
        )

        self.residual_head = nn.Sequential(
            nn.Linear(13 + 64, 128),
            nn.ELU(),
            nn.Linear(128, 4),
        )

        # Exact behavioral preservation at initialization.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.register_buffer(
            "state_target_mean", torch.zeros(13)
        )
        self.register_buffer(
            "state_target_std", torch.ones(13)
        )

    def encode_structured(self, observations):
        h = self.encode_temporal_state(observations)
        estimated_normalized_state = self.state_head(h)
        z = self.latent_head(h)
        return h, estimated_normalized_state, z

    def _actor_mean(self, observations):
        h, estimated_state, z = self.encode_structured(
            observations
        )

        base_logits = self.actor(h)

        residual = self.residual_head(
            torch.cat([estimated_state, z], dim=-1)
        )

        return torch.tanh(base_logits + residual)


def build_policy(device, structured=True):
    cls = (
        StructuredCornersActorCritic
        if structured
        else CornersAsymmetricActorCritic
    )

    return cls(
        num_actor_obs=256,
        num_critic_obs=20,
        num_actions=4,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=0.5,
        noise_std_type="scalar",
        history_length=32,
        token_dim=8,
        temporal_channels=128,
    ).to(device)


def load_split_function():
    spec = importlib.util.spec_from_file_location(
        "historical_aux13", AUX_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_episode_split


def load_dataset():
    data = torch.load(
        DATA, map_location="cpu", weights_only=False
    )

    observations = data["observations"].float()
    actions = data["teacher_actions"].float()
    states = data["teacher_observations"][
        :, STATE_INDICES
    ].float()

    split_fn = load_split_function()

    train, val, test = split_fn(
        data["environment_ids"].long(),
        data["episode_ids"].long(),
    )

    assert observations.shape == (100000, 256)
    assert actions.shape == (100000, 4)
    assert states.shape == (100000, 13)

    assert not (train & val).any()
    assert not (train & test).any()
    assert not (val & test).any()

    return observations, actions, states, train, val, test


def initialize_from_ppo(policy, observations, device):
    ckpt = torch.load(
        PPO400, map_location="cpu", weights_only=False
    )

    original = build_policy(device, structured=False)

    original.load_state_dict(
        ckpt["model_state_dict"], strict=True
    )
    original.eval()

    # Bypass the RSL-RL load_state_dict override, which returns bool.
    # Use PyTorch's implementation to inspect missing/unexpected keys.
    result = nn.Module.load_state_dict(
        policy,
        ckpt["model_state_dict"],
        strict=False,
    )

    if result.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys: {result.unexpected_keys}"
        )

    allowed_prefixes = (
        "state_head.",
        "latent_head.",
        "residual_head.",
        "state_target_",
    )

    if any(
        not key.startswith(allowed_prefixes)
        for key in result.missing_keys
    ):
        raise RuntimeError(
            f"Unexpected missing keys: {result.missing_keys}"
        )

    for key, value in original.state_dict().items():
        transferred = policy.state_dict()[key]
        if not torch.equal(value, transferred):
            raise RuntimeError(f"Weight transfer failed: {key}")

    policy.eval()

    with torch.no_grad():
        x = observations[:512].to(device)

        original_actions = original.act_inference(x)
        structured_actions = policy.act_inference(x)

        maximum_difference = (
            original_actions - structured_actions
        ).abs().max().item()

    print(
        "[PARITY] maximum_action_difference=",
        maximum_difference,
        flush=True,
    )

    if maximum_difference > 1e-7:
        raise RuntimeError(
            "Structured policy changed PPO400 behavior"
        )

    print(
        "[PASS] Exact PPO checkpoint behavior preserved",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["ppo400", "distill"], required=True
    )
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(7500)
    torch.set_num_threads(4)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    output = OUT / f"structured_init_{args.mode}.pt"

    if output.exists():
        raise FileExistsError(
            f"Existing initialization preserved: {output}"
        )

    observations, actions, states, train, val, test = (
        load_dataset()
    )

    state_mean = states[train].mean(0)
    state_std = states[train].std(0).clamp_min(0.05)

    normalized_states = (
        states - state_mean
    ) / state_std

    action_scale = (
        actions[train].std(0).clamp_min(0.05).to(device)
    )

    policy = build_policy(device)

    with torch.no_grad():
        policy.state_target_mean.copy_(
            state_mean.to(device)
        )
        policy.state_target_std.copy_(
            state_std.to(device)
        )

    if args.mode == "ppo400":
        initialize_from_ppo(
            policy, observations, device
        )

        # The PPO actor and encoder are frozen during
        # this initialization. Only fit the new state head.
        for parameter in policy.parameters():
            parameter.requires_grad_(False)

        for parameter in policy.state_head.parameters():
            parameter.requires_grad_(True)

        policy.eval()

        print("[INFO] Precomputing PPO400 features", flush=True)

        features = []

        with torch.no_grad():
            for batch in observations.split(2048):
                features.append(
                    policy.encode_temporal_state(
                        batch.to(device)
                    ).cpu()
                )

        inputs = torch.cat(features)

        params = policy.state_head.parameters()
        learning_rate = 5e-4
        batch_size = 2048

    else:
        inputs = observations

        # Critic is not trained by offline imitation.
        for parameter in policy.critic.parameters():
            parameter.requires_grad_(False)

        if hasattr(policy, "std"):
            policy.std.requires_grad_(False)

        params = [
            p for p in policy.parameters()
            if p.requires_grad
        ]

        learning_rate = 3e-4
        batch_size = 512

    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        weight_decay=1e-5,
    )

    def make_loader(mask, shuffle):
        return DataLoader(
            TensorDataset(
                inputs[mask],
                actions[mask],
                normalized_states[mask],
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )

    train_loader = make_loader(train, True)
    val_loader = make_loader(val, False)

    best_loss = float("inf")
    stale = 0

    for epoch in range(1, args.epochs + 1):
        if args.mode == "distill":
            policy.train()
        else:
            policy.state_head.train()

        train_sum = 0.0
        train_count = 0

        for xb, ab, sb in train_loader:
            xb = xb.to(device)
            ab = ab.to(device)
            sb = sb.to(device)

            if args.mode == "ppo400":
                estimated_state = policy.state_head(xb)
                loss = (estimated_state - sb).square().mean()

            else:
                h, estimated_state, z = (
                    policy.encode_structured(xb)
                )

                predicted_action = policy.act_inference(xb)

                action_loss = (
                    ((predicted_action - ab) / action_scale)
                    .square().mean()
                )

                state_loss = (
                    (estimated_state - sb).square().mean()
                )

                loss = action_loss + 0.2 * state_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups
                 for p in group["params"]],
                1.0,
            )

            optimizer.step()

            train_sum += loss.item() * len(xb)
            train_count += len(xb)

        policy.eval()
        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, ab, sb in val_loader:
                xb = xb.to(device)
                ab = ab.to(device)
                sb = sb.to(device)

                if args.mode == "ppo400":
                    prediction = policy.state_head(xb)
                    loss = (prediction - sb).square().mean()

                else:
                    _, prediction, _ = (
                        policy.encode_structured(xb)
                    )

                    predicted_action = policy.act_inference(xb)

                    action_loss = (
                        ((predicted_action - ab) / action_scale)
                        .square().mean()
                    )

                    state_loss = (
                        (prediction - sb).square().mean()
                    )

                    loss = action_loss + 0.2 * state_loss

                val_sum += loss.item() * len(xb)
                val_count += len(xb)

        val_loss = val_sum / val_count

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            stale = 0

            torch.save(
                {
                    "model_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in policy.state_dict().items()
                    },
                    "initialization": args.mode,
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "state_indices": STATE_INDICES,
                    "source_ppo_checkpoint": (
                        str(PPO400)
                        if args.mode == "ppo400"
                        else None
                    ),
                },
                output,
            )

        else:
            stale += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[TRAIN] mode={args.mode} "
                f"epoch={epoch} "
                f"train={train_sum/train_count:.6f} "
                f"val={val_loss:.6f} "
                f"best={best_loss:.6f}",
                flush=True,
            )

        if stale >= 10:
            break

    ckpt = torch.load(
        output, map_location="cpu", weights_only=False
    )

    policy.load_state_dict(
        ckpt["model_state_dict"], strict=True
    )
    policy.eval()

    with torch.no_grad():
        predictions = []
        state_predictions = []

        for batch in observations[test].split(2048):
            xb = batch.to(device)

            predictions.append(
                policy.act_inference(xb).cpu()
            )

            _, estimated, _ = policy.encode_structured(xb)

            state_predictions.append(
                (
                    estimated * policy.state_target_std
                    + policy.state_target_mean
                ).cpu()
            )

        predicted_actions = torch.cat(predictions)
        predicted_states = torch.cat(state_predictions)

    action_targets = actions[test]
    state_targets = states[test]

    action_r2 = 1 - (
        (predicted_actions - action_targets)
        .square().sum(0)
        /
        (action_targets - action_targets.mean(0))
        .square().sum(0).clamp_min(1e-8)
    )

    state_rmse = (
        predicted_states - state_targets
    ).square().mean(0).sqrt()

    print("\n========== INITIALIZATION RESULTS ==========")
    print("Mode:", args.mode)
    print("Best epoch:", ckpt["epoch"])
    print("Mean held-out action R2:", action_r2.mean().item())
    print("Velocity RMSE:", state_rmse[3:6].tolist())
    print("Gravity RMSE:", state_rmse[6:9].tolist())
    print("[SAVED]", output, flush=True)
    print("[DONE]", args.mode, flush=True)


if __name__ == "__main__":
    main()
