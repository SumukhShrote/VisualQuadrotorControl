from rsl_rl.algorithms.distillation import Distillation
import torch
import torch.nn as nn


class AuxDistillation(Distillation):
    """
    Distillation + DAgger-style beta-mixing + auxiliary bbox loss.

    During collection, env is stepped with:
        mixed_action = beta * teacher_action + (1 - beta) * student_action

    beta anneals linearly from beta_start to beta_end over beta_anneal_iters.
    The teacher action is ALWAYS stored as the BC supervision target.
    """

    def __init__(self, *args,
                 beta_start: float = 1.0,
                 beta_end: float = 0.0,
                 beta_anneal_iters: int = 350,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_iters = beta_anneal_iters
        self._bbox_gt_buffer = []     # collects (N, 5) per step; cleared after update()

    '''@property
    def beta(self) -> float:
        t = min(self.num_updates, self.beta_anneal_iters)
        return self.beta_start + (self.beta_end - self.beta_start) * (t / self.beta_anneal_iters)
    '''

    @property
    def beta(self) -> float:
        """Fraction of environments controlled by the teacher."""

        warmup_updates = 25

        if self.num_updates < warmup_updates:
            return float(self.beta_start)

        anneal_step = min(
            self.num_updates - warmup_updates,
            self.beta_anneal_iters,
        )

        progress = anneal_step / max(self.beta_anneal_iters, 1)

        beta = (
            self.beta_start
            + progress * (self.beta_end - self.beta_start)
        )

        lower = min(self.beta_start, self.beta_end)
        upper = max(self.beta_start, self.beta_end)

        return float(max(lower, min(beta, upper)))

    def act(self, obs, privileged_obs):
        """
        Collect deterministic student-induced states while retaining the
        privileged teacher action as the imitation target.

        A complete environment is controlled by either the teacher or student
        for this step. Teacher and student actions are never numerically averaged.
        """
        with torch.no_grad():
            # Match the deterministic student used by offline collection
            # and deployment.
            student_action = self.policy.act_inference(obs)

            # This exact tensor is both the stored supervision label and the
            # action used when teacher intervention is selected.
            teacher_action = self.policy.evaluate(privileged_obs)

        self.transition.actions = student_action.detach()
        self.transition.privileged_actions = teacher_action.detach()
        self.transition.observations = obs
        self.transition.privileged_observations = privileged_obs

        teacher_mask = (
            torch.rand(
                student_action.shape[0],
                1,
                device=student_action.device,
            )
            < self.beta
        )

        return torch.where(
            teacher_mask,
            teacher_action,
            student_action,
        )
    
    def process_env_step(self, rewards, dones, infos):
        bbox_gt = infos.get("observations", {}).get("bbox_gt", None)
        if bbox_gt is not None:
            self._bbox_gt_buffer.append(bbox_gt.clone())
        return super().process_env_step(rewards, dones, infos)

    def update(self):
        # Beta that actually generated the rollout currently in storage.
        rollout_beta = self.beta
        self.num_updates += 1

        if len(self._bbox_gt_buffer) > 0:
            # Each element is (N, 5); stack -> (T, N, 5), then flatten -> (T*N, 5)
            # This matches the order storage.generator() uses to flatten obs
            bbox_gt_flat = torch.stack(self._bbox_gt_buffer, dim=0).view(-1, 5)
        else:
            bbox_gt_flat = None
        self._bbox_gt_buffer.clear()

        mean_behavior_loss = 0.0
        mean_aux_loss = 0.0
        loss = torch.tensor(0.0, device=self.device)
        cnt = 0
        sample_idx = 0
        accumulation_count = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            sample_idx = 0  # reset each epoch since generator restarts from beginning

            for obs, teacher_obs, _, privileged_actions, dones in self.storage.generator():

                actions = self.policy.act_inference(obs)
                behavior_loss = self.loss_fn(actions, privileged_actions)

                batch_size = obs.shape[0]
                if hasattr(self.policy, "get_aux_loss") and bbox_gt_flat is not None \
                        and sample_idx + batch_size <= bbox_gt_flat.shape[0]:
                    bbox_gt_batch = bbox_gt_flat[sample_idx : sample_idx + batch_size]
                    aux_loss = self.policy.get_aux_loss(bbox_gt_batch)
                    mean_aux_loss += aux_loss.item()
                else:
                    aux_loss = torch.tensor(0.0, device=obs.device)
                sample_idx += batch_size

                loss = loss + behavior_loss + aux_loss
                accumulation_count += 1
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if accumulation_count == self.gradient_length:
                    self.optimizer.zero_grad()

                    averaged_loss = (
                        loss / accumulation_count
                    )

                    averaged_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(
                            self.policy.student.parameters(), self.max_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = torch.tensor(0.0, device=obs.device)

                    accumulation_count = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        # Flush the final incomplete accumulation group.
        if accumulation_count > 0:
            self.optimizer.zero_grad()

            averaged_loss = (
                loss / accumulation_count
            )

            averaged_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.policy.student.parameters(),
                    self.max_grad_norm,
                )

            self.optimizer.step()
        mean_behavior_loss /= max(cnt, 1)
        mean_aux_loss /= max(cnt, 1)
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {
            "behavior": mean_behavior_loss,
            "aux": mean_aux_loss,
            "beta": rollout_beta,
        }