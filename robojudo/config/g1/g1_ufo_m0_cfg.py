"""Original UFO G1 M0, tracking a selected LAFAN motion in MuJoCo."""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.g1_ufo_m0_env_cfg import G1UfoM0MujocoEnvCfg, G1UfoM0RealEnvCfg
from .policy.g1_ufo_m0_policy_cfg import G1UfoM0PolicyCfg


@cfg_registry.register
class g1_ufo_m0(RlPipelineCfg):
    pipeline_type: str = "UfoM0Pipeline"
    robot: str = "g1"
    env: G1UfoM0MujocoEnvCfg = G1UfoM0MujocoEnvCfg()
    ctrl: list = []
    policy: G1UfoM0PolicyCfg = G1UfoM0PolicyCfg()
    run_fullspeed: bool = True


@cfg_registry.register
class g1_ufo_m0_real(RlPipelineCfg):
    pipeline_type: str = "UfoM0Pipeline"
    robot: str = "g1"
    env: G1UfoM0RealEnvCfg = G1UfoM0RealEnvCfg()
    # The pipeline consumes a raw button snapshot and owns the Real state machine.
    ctrl: list = [UnitreeCtrlCfg(triggers={}, combination_init_buttons=[])]
    policy: G1UfoM0PolicyCfg = G1UfoM0PolicyCfg()
    run_fullspeed: bool = False
    do_safety_check: bool = True
