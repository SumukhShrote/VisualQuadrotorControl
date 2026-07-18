'''import torch
from rsl_rl.algorithms.ppo import PPO


class AuxPPO(PPO):
    def update(self):
        policy = getattr(self, "actor_critic", None)
        if policy is None:
            policy = getattr(self, "policy", None)

        has_aux = policy is not None and hasattr(policy, "get_aux_loss")

        if has_aux:
            policy._aux_meter_total = 0.0
            policy._aux_meter_bbox = 0.0
            policy._aux_meter_conf = 0.0
            policy._aux_meter_count = 0

            original_get_aux_loss = policy.get_aux_loss

            def wrapped_get_aux_loss(bbox_gt):
                out = original_get_aux_loss(bbox_gt)

                if isinstance(out, tuple):
                    aux_loss, bbox_loss, conf_loss = out
                else:
                    aux_loss = out
                    bbox_loss = getattr(policy, "_aux_last_bbox_loss", torch.tensor(0.0, device=bbox_gt.device))
                    conf_loss = getattr(policy, "_aux_last_conf_loss", torch.tensor(0.0, device=bbox_gt.device))

                policy._aux_meter_total += float(aux_loss.detach().item())
                policy._aux_meter_bbox += float(bbox_loss.detach().item())
                policy._aux_meter_conf += float(conf_loss.detach().item())
                policy._aux_meter_count += 1

                print("AUX CALLED", bbox_gt.shape)

                return out

            policy.get_aux_loss = wrapped_get_aux_loss

        try:
            loss_dict = super().update()
        finally:
            if has_aux:
                policy.get_aux_loss = original_get_aux_loss

        if has_aux:
            count = max(int(policy._aux_meter_count), 1)
            loss_dict["aux"] = policy._aux_meter_total / count
            loss_dict["bbox_reg"] = policy._aux_meter_bbox / count
            loss_dict["bbox_conf"] = policy._aux_meter_conf / count

        return loss_dict
'''

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO


class AuxPPO(PPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bbox_gt_buffer = []

    def process_env_step(self, rewards, dones, infos):
        bbox_gt = infos.get("observations", {}).get("bbox_gt", None)
        if bbox_gt is not None:
            self._bbox_gt_buffer.append(bbox_gt.clone())
        return super().process_env_step(rewards, dones, infos)

    def update(self):
        if not (
            hasattr(self.policy, "backbone")
            and hasattr(self.policy, "bbox_head")
        ):
            self._bbox_gt_buffer.clear()
            return super().update()

        # Existing RGB auxiliary-PPO implementation continues below.
        # At the start of update(), freeze backbone for PPO pass
        for param in self.policy.backbone.parameters():
            param.requires_grad_(False)
        for param in self.policy.bbox_head.parameters():
            param.requires_grad_(False)

        torch.cuda.empty_cache()
        # Stack buffer: (T, N, 5)
        if len(self._bbox_gt_buffer) > 0:
            bbox_gt_all = torch.stack(self._bbox_gt_buffer, dim=0)  # (T, N, 5)
        else:
            bbox_gt_all = None
        self._bbox_gt_buffer.clear()

        mean_value_loss     = 0
        mean_surrogate_loss = 0
        mean_entropy        = 0
        mean_aux_loss       = 0
        mean_bbox_loss      = 0
        mean_conf_loss      = 0
        aux_count           = 0

        num_updates = self.num_learning_epochs * self.num_mini_batches

        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for i, (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in enumerate(generator):

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch      = self.policy.action_mean
            sigma_batch   = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL-adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2)
                ).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss     += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy        += entropy_batch.mean().item()

        # ---- Aux loss pass: separate from recurrent PPO loop ----
        # Before aux loss loop
        for param in self.policy.backbone.parameters():
            param.requires_grad_(True)
        for param in self.policy.bbox_head.parameters():
            param.requires_grad_(True)
        # Done after the PPO loop but before storage.clear()
        if bbox_gt_all is not None and hasattr(self.policy, "get_aux_loss"):
            obs_all      = self.storage.observations.flatten(0, 1)  # (T*N, obs_dim)
            bbox_gt_flat = bbox_gt_all.flatten(0, 1)                # (T*N, 5)

            # Sample a random subset instead of all T*N — aux loss doesn't need full dataset
            total = obs_all.shape[0]
            num_aux_samples = min(256, total)
            perm = torch.randperm(total, device=self.device)[:num_aux_samples]
            obs_all      = obs_all[perm]
            bbox_gt_flat = bbox_gt_flat[perm]

            torch.cuda.empty_cache()

            chunk = 32
            for start in range(0, obs_all.shape[0], chunk):
                end           = min(start + chunk, obs_all.shape[0])
                obs_chunk     = obs_all[start:end]
                bbox_gt_chunk = bbox_gt_flat[start:end]

                self.policy.reset()
                self.policy.act_inference(obs_chunk)
                aux_loss = self.policy.get_aux_loss(bbox_gt_chunk)

                self.optimizer.zero_grad()
                aux_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_aux_loss  += aux_loss.item()
                mean_bbox_loss += float(getattr(self.policy, "_aux_last_bbox_loss", 0.0))
                mean_conf_loss += float(getattr(self.policy, "_aux_last_conf_loss", 0.0))
                aux_count      += 1

        self.policy.reset()
        torch.cuda.empty_cache()
        self.storage.clear()

        mean_value_loss     /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy        /= num_updates
        mean_aux_loss       /= max(aux_count, 1)
        mean_bbox_loss      /= max(aux_count, 1)
        mean_conf_loss      /= max(aux_count, 1)

        return {
            "value_function": mean_value_loss,
            "surrogate":      mean_surrogate_loss,
            "entropy":        mean_entropy,
            "aux":            mean_aux_loss,
            "bbox_reg":       mean_bbox_loss,
            "bbox_conf":      mean_conf_loss,
        }