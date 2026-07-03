import torch
import torch.nn as nn
import torchvision.models as models
from rsl_rl.modules import StudentTeacher
from torch.distributions import Normal


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
        pooled = self.pool(feats)           # (N, 576, 1, 1)
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

        return 0.5 * bbox_loss + 0.05 * conf_loss

    def evaluate(self, teacher_observations: torch.Tensor) -> torch.Tensor:
        """Matches base class signature exactly — returns teacher actions."""
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions