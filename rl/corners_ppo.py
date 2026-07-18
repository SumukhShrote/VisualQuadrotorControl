from __future__ import annotations

import copy

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO


class CornersAnchoredPPO(PPO):
    """
    PPO with a frozen copy of the original DAgger actor.

    The anchor penalizes movement of the deterministic action mean away
    from the proven DAgger action on current on-policy observations.
    """

    def __init__(
        self,
        *args,
        dagger_anchor_coef: float = 100.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dagger_anchor_coef = float(dagger_anchor_coef)
        self._dagger_reference_student = None

    def set_dagger_reference(
        self,
        student_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not hasattr(self.policy, "student"):
            raise AttributeError(
                "The corners policy has no student module."
            )

        reference = copy.deepcopy(
            self.policy.student
        ).to(self.device)

        if student_state_dict is not None:
            reference.load_state_dict(
                student_state_dict,
                strict=True,
            )

        reference.eval()

        for parameter in reference.parameters():
            parameter.requires_grad_(False)

        self._dagger_reference_student = reference

        print(
            "[CornersAnchoredPPO] Frozen DAgger reference created:",
            f"anchor_coef={self.dagger_anchor_coef}",
        )

    def _reference_actions(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if self._dagger_reference_student is None:
            raise RuntimeError(
                "The frozen DAgger reference was not initialized."
            )

        sequence = observations.reshape(
            observations.shape[0],
            int(self.policy.history_length),
            int(self.policy.token_dim),
        ).transpose(1, 2)

        with torch.no_grad():
            temporal_features = (
                self._dagger_reference_student[0](sequence)
            )

            latent = temporal_features[:, :, -1]

            raw_actions = (
                self._dagger_reference_student[1](latent)
            )

            return torch.tanh(raw_actions)

    def update(self):
        if self.policy.is_recurrent:
            raise RuntimeError(
                "CornersAnchoredPPO requires a non-recurrent policy."
            )

        if self.rnd is not None:
            raise RuntimeError(
                "RND is unsupported by CornersAnchoredPPO."
            )

        if self.symmetry is not None:
            raise RuntimeError(
                "Symmetry is unsupported by CornersAnchoredPPO."
            )

        if self.schedule != "fixed":
            raise RuntimeError(
                "Use schedule='fixed' with separate actor/critic rates."
            )

        if (
            self.dagger_anchor_coef > 0.0
            and self._dagger_reference_student is None
        ):
            raise RuntimeError(
                "A positive anchor coefficient requires a "
                "frozen DAgger reference."
            )

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_anchor_loss = 0.0
        mean_anchor_mae = 0.0
        maximum_anchor_error = 0.0
        update_count = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            _hidden_states_batch,
            _masks_batch,
            _rnd_state_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (
                        advantages_batch
                        - advantages_batch.mean()
                    ) / (
                        advantages_batch.std()
                        + 1.0e-8
                    )

            # Current stochastic policy.
            self.policy.act(obs_batch)

            actions_log_prob_batch = (
                self.policy.get_actions_log_prob(
                    actions_batch
                )
            )

            value_batch = self.policy.evaluate(
                critic_obs_batch
            )

            action_mean = self.policy.action_mean
            entropy_batch = self.policy.entropy

            old_log_prob = old_actions_log_prob_batch.squeeze(-1)
            advantages = advantages_batch.squeeze(-1)

            ratio = torch.exp(
                actions_log_prob_batch - old_log_prob
            )

            surrogate = -advantages * ratio

            surrogate_clipped = (
                -advantages
                * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
            )

            surrogate_loss = torch.maximum(
                surrogate,
                surrogate_clipped,
            ).mean()

            if self.use_clipped_value_loss:
                value_clipped = (
                    target_values_batch
                    + (
                        value_batch
                        - target_values_batch
                    ).clamp(
                        -self.clip_param,
                        self.clip_param,
                    )
                )

                value_loss = torch.maximum(
                    (
                        value_batch
                        - returns_batch
                    ).pow(2),
                    (
                        value_clipped
                        - returns_batch
                    ).pow(2),
                ).mean()
            else:
                value_loss = (
                    returns_batch
                    - value_batch
                ).pow(2).mean()

            if self.dagger_anchor_coef > 0.0:
                reference_actions = (
                    self._reference_actions(obs_batch)
                )

                action_difference = (
                    action_mean - reference_actions
                )

                anchor_loss = (
                    action_difference.pow(2).mean()
                )

                anchor_mae = (
                    action_difference.abs().mean()
                )

                anchor_max = (
                    action_difference.abs().max()
                )
            else:
                anchor_loss = torch.zeros(
                    (),
                    device=self.device,
                )
                anchor_mae = torch.zeros(
                    (),
                    device=self.device,
                )
                anchor_max = torch.zeros(
                    (),
                    device=self.device,
                )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.dagger_anchor_coef * anchor_loss
            )

            self.optimizer.zero_grad()
            loss.backward()

            trainable_parameters = [
                parameter
                for parameter in self.policy.parameters()
                if parameter.requires_grad
            ]

            nn.utils.clip_grad_norm_(
                trainable_parameters,
                self.max_grad_norm,
            )

            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_anchor_loss += anchor_loss.item()
            mean_anchor_mae += anchor_mae.item()
            maximum_anchor_error = max(
                maximum_anchor_error,
                anchor_max.item(),
            )
            update_count += 1

        self.storage.clear()

        denominator = max(update_count, 1)

        return {
            "value_function": (
                mean_value_loss / denominator
            ),
            "surrogate": (
                mean_surrogate_loss / denominator
            ),
            "entropy": (
                mean_entropy / denominator
            ),
            "dagger_anchor": (
                mean_anchor_loss / denominator
            ),
            "dagger_action_mae": (
                mean_anchor_mae / denominator
            ),
            "dagger_action_max": maximum_anchor_error,
        }
