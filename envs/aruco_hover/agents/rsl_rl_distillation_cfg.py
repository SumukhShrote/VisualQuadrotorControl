from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.distillation_cfg import (
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
)


@configclass
class VisionStudentTeacherCfg(
    RslRlDistillationStudentTeacherCfg
):
    use_gru: bool = False

    history_length: int = 32
    token_dim: int = 8
    temporal_channels: int = 128


@configclass
class QuadrotorStudentDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 128
    max_iterations: int = 500
    save_interval: int = 50
    experiment_name: str = "CTBR_Hover_Distill"
    resume: bool = False

    obs_groups = {
        "policy":  ["policy"],
        "teacher": ["teacher"],
    }

    policy: VisionStudentTeacherCfg = VisionStudentTeacherCfg(
        class_name="VisionStudentTeacher",
        init_noise_std=0.01,

        student_hidden_dims=[256, 256],
        teacher_hidden_dims=[64, 64],

        activation="elu",
        use_gru=False,

        history_length=32,
        token_dim=8,
        temporal_channels=128,
    )

    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=3e-4,
        gradient_length=16,
        max_grad_norm=1.0,
    )