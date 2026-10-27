"""MuJoCo environment matching UFO M0's G1 model and 50 Hz control."""

from typing import Literal

from robojudo.config import ASSETS_DIR
from robojudo.config.g1.policy.g1_ufo_m0_policy_cfg import G1UfoM0DoF
from robojudo.environment.env_cfgs import MujocoEnvCfg

from .g1_env_cfg import G1EnvCfg


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
