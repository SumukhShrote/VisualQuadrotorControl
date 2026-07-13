from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg
from isaaclab_rl.rsl_rl.distillation_cfg import (
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
)


@configclass
class VisionStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Extends the base cfg with vision-specific fields."""
    use_gru: bool = True
    gru_hidden_size: int = 128


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
        student_hidden_dims=[128, 128],
        teacher_hidden_dims=[64, 64],
        activation="tanh",
        use_gru=False,        # flip to True to enable GRU
        gru_hidden_size=128,
    )

    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=3e-4,
        gradient_length=16,
        max_grad_norm=1.0,
    )