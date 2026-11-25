import torch
import torch.nn as nn
import torchvision.models as models
from rsl_rl.modules import StudentTeacher
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory
import torch.nn.functional as F


class CausalTCNBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
    ):
        super().__init__()

        self.left_padding = kernel_size - 1

        self.conv = nn.Conv1d(
            input_dim,
            output_dim,
            kernel_size=kernel_size,
        )

        self.residual = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=1,
            )
        )

        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        x = F.pad(
            x,
            (self.left_padding, 0),
        )

        x = self.conv(x)
        x = x + residual

        return self.activation(x)

class VisionStudentTeacher(StudentTeacher):
    """
    Student:
        32 timestamps x 8 normalized marker-corner coordinates
        → 3-layer causal TCN
        → 128-D temporal state
        → CTBR action

    Teacher:
        20-D privileged state.
    """

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        student_hidden_dims=[256, 256],
        teacher_hidden_dims=[64, 64],
        activation="elu",
        init_noise_std=0.01,
        history_length: int = 32,
        token_dim: int = 8,
        temporal_channels: int = 128,
        use_gru: bool = False,
        **kwargs,
    ):
        if use_gru:
            raise ValueError(
                "Corners-TCN must use use_gru=False."
            )

        # Ignore obsolete arguments still passed by train_rsl_rl.py.
        for key in (
            "imu_dropout_prob",
            "bbox_loss_weight",
            "conf_loss_weight",
            "ppo_bbox_grad_scale",
            "img_height",
            "img_width",
            "frame_stack",
            "visual_latent_dim",
            "previous_action_dim",
        ):
            kwargs.pop(key, None)

        self.history_length = history_length
        self.token_dim = token_dim
        self.temporal_channels = temporal_channels

        self.use_gru = False
        self.is_recurrent = False

        expected_num_obs = history_length * token_dim

        assert num_student_obs == expected_num_obs, (
            f"{num_student_obs=} != {expected_num_obs=}"
        )

        assert num_teacher_obs == 20, (
            f"Expected 20 teacher observations, "
            f"got {num_teacher_obs}"
        )

        super().__init__(
            num_student_obs=num_student_obs,
            num_teacher_obs=num_teacher_obs,
            num_actions=num_actions,
            student_hidden_dims=student_hidden_dims,
            teacher_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs,
        )

        self.teacher_recurrent = False

        # Three layers with kernel size 11 give a 31-step
        # causal receptive field.
        self.tcn = nn.Sequential(
            CausalTCNBlock(
                input_dim=token_dim,
                output_dim=temporal_channels,
                kernel_size=11,
            ),
            CausalTCNBlock(
                input_dim=temporal_channels,
                output_dim=temporal_channels,
                kernel_size=11,
            ),
            CausalTCNBlock(
                input_dim=temporal_channels,
                output_dim=temporal_channels,
                kernel_size=11,
            ),
        )

        self.vision_student = nn.Sequential(
            nn.Linear(
                temporal_channels,
                256,
            ),
            nn.ELU(),
            nn.Linear(
                256,
                256,
            ),
            nn.ELU(),
            nn.Linear(
                256,
                num_actions,
            ),
        )

        # Used by clipping, checkpointing and offline training.
        self.student = nn.ModuleList(
            [
                self.tcn,
                self.vision_student,
            ]
        )

        print(
            "[Corners-TCN student]",
            f"student_obs={num_student_obs}",
            f"history={history_length}",
            f"token_dim={token_dim}",
            f"temporal_channels={temporal_channels}",
        )

    def encode_temporal_state(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 128-D newest-time TCN latent without producing an action."""
        if observations.ndim != 2:
            raise RuntimeError(
                f"Expected observations with shape [B, {self.history_length * self.token_dim}], "
                f"got {tuple(observations.shape)}."
            )

        expected_dim = self.history_length * self.token_dim
        if observations.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Expected observation dimension {expected_dim}, "
                f"got {observations.shape[-1]}."
            )

        sequence = observations.reshape(
            observations.shape[0],
            self.history_length,
            self.token_dim,
        )

        # Conv1d requires B × channels × time.
        sequence = sequence.transpose(1, 2)
        temporal_features = self.tcn(sequence)

        # Newest causal representation. With three kernel-11 blocks and
        # dilation 1, this latent has a 31-frame receptive field.
        return temporal_features[:, :, -1]

    def _encode(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encode_temporal_state(observations)
        raw_action = self.vision_student(latent)
        return torch.tanh(raw_action)

    def update_distribution(
        self,
        observations: torch.Tensor,
    ):
        mean = self._encode(observations)

        self.distribution = Normal(
            mean,
            self.std.expand_as(mean),
        )

    def act_inference(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode(observations)

    def evaluate(
        self,
        teacher_observations: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.teacher(
                teacher_observations
            )

    def reset(
        self,
        dones=None,
        hidden_states=None,
    ):
        return

    def get_hidden_states(self):
        return None

    def detach_hidden_states(
        self,
        dones=None,
    ):
        return

    def get_aux_loss(
        self,
        bbox_gt=None,
    ):
        return torch.tensor(
            0.0,
            device=next(self.parameters()).device,
        )

class CornersTCNActorCritic(ActorCritic):
    """
    Asymmetric PPO policy.

    Actor:
        32 x 8 normalized ArUco-corner history
        -> three-layer causal TCN
        -> 256, 256 MLP
        -> tanh CTBR action

    Critic:
        20-D privileged teacher observation
        -> MLP
        -> scalar value

    The `student` ModuleList intentionally matches the offline DAgger
    checkpoint layout exactly:

        student[0] = TCN
        student[1] = action MLP
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=0.03,
        noise_std_type="scalar",
        history_length: int = 32,
        token_dim: int = 8,
        temporal_channels: int = 128,
        **kwargs,
    ):
        # Ignore fields that may still be injected by the older visual path.
        for key in (
            "use_gru",
            "gru_hidden_size",
            "rnn_type",
            "rnn_hidden_dim",
            "rnn_num_layers",
            "imu_dropout_prob",
            "bbox_loss_weight",
            "conf_loss_weight",
            "ppo_bbox_grad_scale",
            "img_height",
            "img_width",
        ):
            kwargs.pop(key, None)

        expected_actor_obs = history_length * token_dim

        if num_actor_obs != expected_actor_obs:
            raise ValueError(
                f"Expected {expected_actor_obs} actor observations, "
                f"got {num_actor_obs}."
            )

        if num_critic_obs != 20:
            raise ValueError(
                f"Expected 20 privileged critic observations, "
                f"got {num_critic_obs}."
            )

        # This constructs the critic and action-noise parameter.
        # The temporary MLP actor is replaced below by the exact corners TCN.
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        self.history_length = int(history_length)
        self.token_dim = int(token_dim)
        self.temporal_channels = int(temporal_channels)

        self.tcn = nn.Sequential(
            CausalTCNBlock(
                input_dim=self.token_dim,
                output_dim=self.temporal_channels,
                kernel_size=11,
            ),
            CausalTCNBlock(
                input_dim=self.temporal_channels,
                output_dim=self.temporal_channels,
                kernel_size=11,
            ),
            CausalTCNBlock(
                input_dim=self.temporal_channels,
                output_dim=self.temporal_channels,
                kernel_size=11,
            ),
        )

        # Replace ActorCritic's temporary actor with the exact DAgger head.
        self.actor = nn.Sequential(
            nn.Linear(self.temporal_channels, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, num_actions),
        )

        # This exact layout accepts the offline student_state_dict strictly.
        self.student = nn.ModuleList(
            [
                self.tcn,
                self.actor,
            ]
        )

        print(
            "[CornersTCNActorCritic]",
            f"actor_obs={num_actor_obs}",
            f"critic_obs={num_critic_obs}",
            f"history={self.history_length}",
            f"token_dim={self.token_dim}",
            f"temporal_channels={self.temporal_channels}",
            f"noise_std={init_noise_std}",
        )

    def encode_temporal_state(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 128-D newest-time TCN latent without producing an action."""
        if observations.ndim != 2:
            raise RuntimeError(
                f"Expected observations with shape [B, {self.history_length * self.token_dim}], "
                f"got {tuple(observations.shape)}."
            )

        expected_dim = self.history_length * self.token_dim
        if observations.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Expected observation dimension {expected_dim}, "
                f"got {observations.shape[-1]}."
            )

        sequence = observations.reshape(
            observations.shape[0],
            self.history_length,
            self.token_dim,
        )

        # Conv1d input: batch x channels x time.
        sequence = sequence.transpose(1, 2)
        temporal_features = self.tcn(sequence)
        return temporal_features[:, :, -1]

    def _encode_actor(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.encode_temporal_state(observations)
        raw_action = self.actor(latent)
        return torch.tanh(raw_action)

    def update_distribution(
        self,
        observations: torch.Tensor,
    ):
        mean = self._encode_actor(observations)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown noise_std_type: {self.noise_std_type}"
            )

        self.distribution = Normal(mean, std)

    def act_inference(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_actor(observations)

class _VisionEncoderMixin:
    def _build_vision_encoder(self, img_height, img_width, imu_dim,
                          imu_dropout_prob, bbox_loss_weight, conf_loss_weight,
                          ppo_bbox_grad_scale=0.05):
        self.H = img_height
        self.W = img_width
        self.imu_dim = imu_dim
        self.imu_dropout_prob = imu_dropout_prob
        self.bbox_loss_weight = bbox_loss_weight
        self.conf_loss_weight = conf_loss_weight
        self.ppo_bbox_grad_scale = ppo_bbox_grad_scale

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.backbone = backbone.features

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Sigmoid(),
        )

        self._last_marker = None

    def _encode_visual_features_flat(self, observations: torch.Tensor) -> torch.Tensor:
        imu = observations[:, :self.imu_dim]
        img = observations[:, self.imu_dim:].view(-1, 3, self.H, self.W)
        img = (img - self.imagenet_mean) / self.imagenet_std

        feats = self.backbone(img)
        marker = self.bbox_head(feats)
        self._last_marker = marker
        marker_for_policy = marker.detach() + self.ppo_bbox_grad_scale * (marker - marker.detach())

        if self.training and self.imu_dropout_prob > 0.0:
            dropout_mask = (
                torch.rand(imu.shape[0], 1, device=imu.device) > self.imu_dropout_prob
            ).float()
            imu = imu * dropout_mask

        return torch.cat([imu, marker_for_policy], dim=-1)

    def _get_aux_loss_impl(self, bbox_gt: torch.Tensor) -> torch.Tensor:
        if self._last_marker is None:
            return torch.tensor(0.0, device=bbox_gt.device)

        import torch.nn.functional as F

        if bbox_gt.dim() == 3:
            bbox_gt = bbox_gt.reshape(-1, bbox_gt.shape[-1])

        pred_bbox = self._last_marker[:, :4]
        pred_conf = self._last_marker[:, 4]
        gt_bbox = bbox_gt[:, :4]
        gt_vis = bbox_gt[:, 4]

        conf_loss = F.binary_cross_entropy(pred_conf, gt_vis)

        visible = gt_vis.bool()
        if visible.sum() == 0:
            bbox_loss = torch.tensor(0.0, device=bbox_gt.device)
        else:
            weights = torch.tensor([2.0, 2.0, 1.0, 1.0], device=bbox_gt.device)
            bbox_diff = pred_bbox[visible] - gt_bbox[visible]
            bbox_loss = (weights * (bbox_diff ** 2)).mean()

        total_aux_loss = self.bbox_loss_weight * bbox_loss + self.conf_loss_weight * conf_loss

        self._aux_last_bbox_loss = bbox_loss.detach()
        self._aux_last_conf_loss = conf_loss.detach()
        self._aux_last_total_loss = total_aux_loss.detach()

        return total_aux_loss

class VisionActorCritic(_VisionEncoderMixin, ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type="scalar",
        img_height=128,
        img_width=128,
        imu_dim=13,
        imu_dropout_prob=0.0,
        bbox_loss_weight=0.0,
        conf_loss_weight=0.0,
        ppo_bbox_grad_scale=0.05,
        use_gru: bool = False,
        gru_hidden_size: int | None = None,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 1,
        **kwargs,
    ):
        super().__init__(
            num_actor_obs=imu_dim + 5,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        self._build_vision_encoder(
            img_height=img_height,
            img_width=img_width,
            imu_dim=imu_dim,
            imu_dropout_prob=imu_dropout_prob,
            bbox_loss_weight=bbox_loss_weight,
            conf_loss_weight=conf_loss_weight,
            ppo_bbox_grad_scale=ppo_bbox_grad_scale,
        )

    def update_distribution(self, observations: torch.Tensor):
        actor_features = self._encode_visual_features_flat(observations)
        mean = self.actor(actor_features)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actor_features = self._encode_visual_features_flat(observations)
        return self.actor(actor_features)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def get_aux_loss(self, bbox_gt: torch.Tensor) -> torch.Tensor:
        return self._get_aux_loss_impl(bbox_gt)


class VisionActorCriticRecurrent(_VisionEncoderMixin, ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type="scalar",
        img_height=128,
        img_width=128,
        imu_dim=13,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        imu_dropout_prob=0.0,
        bbox_loss_weight=0.0,
        conf_loss_weight=0.0,
        ppo_bbox_grad_scale=0.05,
        use_gru: bool = True,
        gru_hidden_size: int | None = None,
        **kwargs,
    ):
        if not use_gru:
            raise ValueError("VisionActorCriticRecurrent must be constructed with use_gru=True.")

        if gru_hidden_size is not None:
            rnn_hidden_dim = gru_hidden_size

        super().__init__(
            num_actor_obs=rnn_hidden_dim,
            num_critic_obs=rnn_hidden_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        self._build_vision_encoder(
            img_height=img_height,
            img_width=img_width,
            imu_dim=imu_dim,
            imu_dropout_prob=imu_dropout_prob,
            bbox_loss_weight=bbox_loss_weight,
            conf_loss_weight=conf_loss_weight,
            ppo_bbox_grad_scale=ppo_bbox_grad_scale,
        )

        self.memory_a = Memory(
            imu_dim + 5,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )
        self.memory_c = Memory(
            num_critic_obs,
            type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def _encode_actor_features_sequence_aware(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 2:
            return self._encode_visual_features_flat(observations)

        T, B, D = observations.shape
        obs_flat = observations.reshape(T * B, D)
        features_flat = self._encode_visual_features_flat(obs_flat)
        return features_flat.view(T, B, -1)

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        actor_features = self._encode_actor_features_sequence_aware(observations)
        input_a = self.memory_a(actor_features, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        actor_features = self._encode_actor_features_sequence_aware(observations)
        input_a = self.memory_a(actor_features)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_aux_loss(self, bbox_gt: torch.Tensor) -> torch.Tensor:
        return self._get_aux_loss_impl(bbox_gt)