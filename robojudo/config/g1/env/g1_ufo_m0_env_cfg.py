"""MuJoCo environment matching UFO M0's G1 model and 50 Hz control."""

from typing import Literal

from pydantic import Field

from robojudo.config import ASSETS_DIR
from robojudo.config.g1.policy.g1_ufo_m0_policy_cfg import G1UfoM0DoF
from robojudo.environment.env_cfgs import MujocoEnvCfg, UnitreeEnvCfg

from .g1_env_cfg import G1EnvCfg
from .g1_real_env_cfg import G1RealEnvCfg


class G1UfoM0MujocoEnvCfg(G1EnvCfg, MujocoEnvCfg):
    env_type: str = "MujocoEnv"
    is_sim: bool = True
    xml: str = (ASSETS_DIR / "robots/g1/g1_ufo_m0.xml").as_posix()
    dof: G1UfoM0DoF = G1UfoM0DoF()
    forward_kinematic: None = None
    update_with_fk: bool = False
    born_place_align: bool = False
    sim_dt: float = 0.005
    sim_decimation: int = 4
    render: bool = False
    visualize_extras: bool = False
    scene_style: Literal["model", "light"] = "light"


class G1UfoM0RealEnvCfg(G1RealEnvCfg):
    """RoboJuDo's tested G1 driver with UFO M0 joints and opt-in target guard."""

    dof: G1UfoM0DoF = G1UfoM0DoF()
    forward_kinematic: None = None
    update_with_fk: bool = False
    born_place_align: bool = False
    target_velocity_limits: list[float] = [
        32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
        32.0, 32.0, 32.0, 20.0, 37.0, 37.0,
        32.0, 37.0, 37.0,
        37.0, 37.0, 37.0, 37.0, 22.0, 22.0, 22.0,
        37.0, 37.0, 37.0, 37.0, 22.0, 22.0, 22.0,
    ]
    target_slew_safety_factor: float = 0.5


class G1UfoWorkstationEnvCfg(G1UfoM0RealEnvCfg):
    env_type: str = "UnitreeG1DdsEnv"
    unitree: UnitreeEnvCfg.UnitreeCfg = G1RealEnvCfg().unitree.model_copy(
        update={"net_if": "", "enable_odometry": False})
    odometry_type: Literal["NONE"] = "NONE"
    dds_domain_id: int = Field(default=0, ge=0, le=232)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    feedback_timeout_s: float = Field(default=0.1, gt=0)
    command_timeout_s: float = Field(default=0.1, gt=0)
