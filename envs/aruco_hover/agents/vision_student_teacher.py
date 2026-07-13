import torch
import torch.nn as nn
import torchvision.models as models
from rsl_rl.modules import StudentTeacher
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import Memory


class VisionStudentTeacher(StudentTeacher):
    """
    StudentTeacher where the student sees IMU + flattened RGB images.

    Student obs (flat): [imu(10), rgb(3*H*W)] = 12298-D
    Teacher obs:        17-D GT privileged state

    use_gru=True:  imu+marker(14) -> GRU(128) -> MLP -> actions
    use_gru=False: imu+marker(14) -> MLP -> actions   (no temporal memory)
    """

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        student_hidden_dims=[128, 128],
        teacher_hidden_dims=[64, 64],
        activation="elu",
        init_noise_std=0.1,
        img_height: int = 128,
        img_width: int = 128,
        imu_dim: int = 13,
        use_gru: bool = True,          # toggle recurrent memory
        gru_hidden_size: int = 128,    # configurable, only used if use_gru
        bbox_loss_weight: float = 0.0,    
        conf_loss_weight: float = 0.0,    
        imu_dropout_prob: float = 0.0,    
        **kwargs,
    ):
        self.H = img_height
        self.W = img_width
        self.imu_dim = imu_dim
        self.use_gru = use_gru      

        expected_num_obs = imu_dim + 3 * img_height * img_width
        assert num_student_obs == expected_num_obs, (
            f"{num_student_obs=} != {expected_num_obs=}"
        )

        # is_recurrent must be set as an INSTANCE attribute here (overriding
        # the class-level default) BEFORE super().__init__, since the base
        # class / runner may read it immediately after construction.
        self.is_recurrent = use_gru    

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
        self.teacher_recurrent = False   # teacher is always a plain MLP

        # ---- Vision backbone --------------------------------------------
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # used for GRU pathway only
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # Output layout: [cx, cy, w, h, confidence]
        # cx, cy, w, h are passed through Sigmoid -> constrained to [0, 1]
        # confidence is a separate Sigmoid -> [0, 1] visibility score
        # Spatial bbox head — operates on full (N, 576, 4, 4) feature map
        # preserving spatial extent so w/h can be predicted accurately
        self.bbox_head = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=3, padding=1),  # (N, 64, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                                   # (N, 1024)
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Sigmoid(),
        )

        act_fn = dict(elu=nn.ELU, relu=nn.ReLU, tanh=nn.Tanh)[activation]

        # ---- Optional GRU temporal memory --------------------------------
        if self.use_gru:
            self.gru_hidden_size = gru_hidden_size
            self.gru = nn.GRU(
                input_size=imu_dim + 5,
                hidden_size=self.gru_hidden_size,
                batch_first=True,
            )
            self.hidden_states = None
            mlp_input_dim = self.gru_hidden_size
        else:
            self.gru = None
            self.hidden_states = None
            mlp_input_dim = imu_dim + 5   # feedforward straight from imu+marker; mlp_input_dim = imu_dim + 5   # imu(13) + [cx, cy, w, h, conf](5)

        # ---- Student MLP --------------------------------------------------
        layers = []
        prev = mlp_input_dim
        for h in student_hidden_dims:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.vision_student = nn.Sequential(*layers)

        # Point self.student at real trainable modules (grad clipping target)
        modules = [self.backbone, self.pool, self.bbox_head, self.vision_student]
        if self.use_gru:
            modules.insert(3, self.gru)   # keep ordering readable; order doesn't matter functionally
        self.student = nn.ModuleList(modules)

        self.imu_dropout_prob = imu_dropout_prob
        self.bbox_loss_weight = bbox_loss_weight
        self.conf_loss_weight = conf_loss_weight

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    
    def _encode(self, observations: torch.Tensor, return_marker: bool = False) -> torch.Tensor:
        """Split flat obs, run vision encoder (+ optional GRU), return action mean."""
        imu = observations[:, :self.imu_dim]
        img = observations[:, self.imu_dim:].view(-1, 3, self.H, self.W)
        img = (img - self.imagenet_mean) / self.imagenet_std

        feats  = self.backbone(img)          # (N, 576, 4, 4)
        self._last_feature_map = feats
        marker = self.bbox_head(feats)       # spatial path — preserves w/h info
        pooled = self.pool(feats)            # (N, 576, 1, 1)
        
        # --- NEW: IMU Dropout during training ---
        # 30% of the time, zero out the IMU so the GRU MUST rely on the visual marker.
        # This breaks "causal confusion" where the network ignores the image.
        if self.training and self.imu_dropout_prob > 0.0:
            dropout_mask = (torch.rand(imu.shape[0], 1, device=imu.device) > self.imu_dropout_prob).float()
            imu = imu * dropout_mask

        combined = torch.cat([imu, marker], dim=-1)

        if not self.use_gru:
            actions = self.vision_student(combined)
            if return_marker:
                return actions, marker
            return actions

        gru_input = combined.unsqueeze(1)
        batch_size = observations.shape[0]
        if self.hidden_states is None or self.hidden_states.shape[1] != batch_size:
            self.hidden_states = torch.zeros(
                1, batch_size, self.gru_hidden_size, device=observations.device
            )

        gru_out, self.hidden_states = self.gru(gru_input, self.hidden_states)
        actions = self.vision_student(gru_out.squeeze(1))

        if return_marker:
            return actions, marker
        return actions

    # -------------------------------------------------------------------------
    # Recurrent-state overrides — all become safe no-ops when use_gru=False
    # -------------------------------------------------------------------------

    def reset(self, dones=None, hidden_states=None):
        if not self.use_gru:
            return
        if hidden_states is not None:
            self.hidden_states = hidden_states.clone()
        elif self.hidden_states is not None:
            if dones is not None:
                dones = dones.bool()
                if dones.any():
                    self.hidden_states[:, dones, :] = 0.0

    def get_hidden_states(self):
        if not self.use_gru or self.hidden_states is None:
            return None
        return self.hidden_states.clone()

    def detach_hidden_states(self, dones=None):
        if not self.use_gru or self.hidden_states is None:
            return
        self.hidden_states = self.hidden_states.detach().clone()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        self.update_distribution(observations)   # stashes _last_marker as side effect
        return self.distribution.mean            # deterministic mean, same as before

    def update_distribution(self, observations: torch.Tensor):
        mean, marker = self._encode(observations, return_marker=True)
        self.distribution = Normal(mean, self.std.expand_as(mean))
        self._last_marker = marker          # stash for get_aux_loss()

    def get_aux_loss(self, bbox_gt: torch.Tensor) -> torch.Tensor:
        """
        bbox_gt shape: (N, 5) — [cx, cy, w, h, visible_flag]
        visible_flag: 1.0 = marker in frame, 0.0 = not visible.

        Two losses:
          1. bbox_loss   — MSE on [cx, cy, w, h], computed ONLY on visible frames
          2. conf_loss   — BCE on the predicted confidence vs visible_flag (all frames)
        """
        if not hasattr(self, '_last_marker') or self._last_marker is None:
            return torch.tensor(0.0, device=bbox_gt.device)

        import torch.nn.functional as F

        pred_bbox = self._last_marker[:, :4]   # [cx, cy, w, h]
        pred_conf = self._last_marker[:, 4]    # visibility confidence
        gt_bbox   = bbox_gt[:, :4]
        gt_vis    = bbox_gt[:, 4]              # 1.0 or 0.0

        # Confidence loss — trained on ALL frames (teaches "is marker visible?")
        conf_loss = F.binary_cross_entropy(pred_conf, gt_vis)

        # Bbox loss — trained ONLY on visible frames (prevents learning sentinel)
        visible = gt_vis.bool()
        if visible.sum() == 0:
            bbox_loss = torch.tensor(0.0, device=bbox_gt.device)
        else:
            bbox_loss = F.mse_loss(pred_bbox[visible], gt_bbox[visible])

        # Replace the hardcoded return statement with this:
        return (self.bbox_loss_weight * bbox_loss) + (self.conf_loss_weight * conf_loss)

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        """Matches base class signature exactly — returns teacher actions."""
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

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