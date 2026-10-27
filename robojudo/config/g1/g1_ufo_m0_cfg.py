"""Original UFO G1 M0, tracking a selected LAFAN motion in MuJoCo."""

from robojudo.config import cfg_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.g1_ufo_m0_env_cfg import G1UfoM0MujocoEnvCfg
from .policy.g1_ufo_m0_policy_cfg import G1UfoM0PolicyCfg


@cfg_registry.register
class g1_ufo_m0(RlPipelineCfg):
    pipeline_type: str = "UfoM0Pipeline"
    robot: str = "g1"
    env: G1UfoM0MujocoEnvCfg = G1UfoM0MujocoEnvCfg()
    ctrl: list = []
    policy: G1UfoM0PolicyCfg = G1UfoM0PolicyCfg()
    run_fullspeed: bool = True
