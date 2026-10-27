"""UFO original G1 M0 deployment parameters."""

from robojudo.config import ASSETS_DIR
from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.policy.policy_cfgs import PolicyCfg


UFO_EFFORT_LIMITS = [
    139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
    25.0, 25.0, 25.0, 25.0, 25.0, 13.4, 13.4,
]


class G1UfoM0DoF(G1_29DoF):
    """29 joint order and PD settings from UFO's g1_29dof.yaml."""

    stiffness: list[float] | None = [
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        99.09843, 99.09843, 40.17924, 99.09843, 28.50125, 28.50125,
        300.0, 300.0, 300.0,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 8.61103, 8.61103,
        14.25062, 14.25062, 14.25062, 14.25062, 14.25062, 8.61103, 8.61103,
    ]
    damping: list[float] | None = [
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        6.3088, 6.3088, 2.55789, 6.3088, 1.81445, 1.81445,
        5.0, 5.0, 5.0,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 0.5482, 0.5482,
        0.90722, 0.90722, 0.90722, 0.90722, 0.90722, 0.5482, 0.5482,
    ]
    torque_limits: list[float] | None = [0.8 * limit for limit in UFO_EFFORT_LIMITS]


class G1UfoM0PolicyCfg(PolicyCfg):
    policy_type: str = "UfoM0Policy"
    robot: str = "g1"
    disable_autoload: bool = True
    model_dir: str = (ASSETS_DIR / "models/g1/ufo_m0").as_posix()
    motion_path: str = "~/data/lafan/lafan_g1_npz/walk1_subject1.npz"
    start_frame: int = 0
    max_frames: int | None = 500
    freq: int = 50
    obs_dof: G1UfoM0DoF = G1UfoM0DoF()
    action_dof: G1UfoM0DoF = G1UfoM0DoF()
    effort_limits: list[float] = UFO_EFFORT_LIMITS
    history_length: int = 4

    @property
    def policy_file(self) -> str:
        return f"{self.model_dir}/FBcprAuxModel.onnx"
