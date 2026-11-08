"""Video-derived UFO M0 latents with shared file/socket controller."""

from robojudo.config import cfg_registry
from pydantic import Field

from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg, LatentCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.g1_ufo_m0_env_cfg import G1UfoM0MujocoEnvCfg, G1UfoM0RealEnvCfg, G1UfoWorkstationEnvCfg
from .policy.g1_ufo_m0_policy_cfg import G1UfoM0PolicyCfg
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg


@cfg_registry.register
class g1_ufo_video_file(RlPipelineCfg):
    pipeline_type: str = "UfoM0Pipeline"
    robot: str = "g1"
    env: G1UfoM0MujocoEnvCfg = G1UfoM0MujocoEnvCfg()
    ctrl: list = [LatentCtrlCfg(source="file")]
    policy: G1UfoM0PolicyCfg = G1UfoM0PolicyCfg(latent_source="controller")
    run_fullspeed: bool = True


@cfg_registry.register
class g1_ufo_video_stream(g1_ufo_video_file):
    ctrl: list = [LatentCtrlCfg(source="socket", stale_seconds=5.0, max_age_seconds=5.0)]
    run_fullspeed: bool = False


@cfg_registry.register
class g1_ufo_video_camera(g1_ufo_video_stream):
    """Live camera demo with the same latent age limit as G1 Real."""

    env: G1UfoM0MujocoEnvCfg = G1UfoM0MujocoEnvCfg(render=True)
    ctrl: list = [LatentCtrlCfg(source="socket", stale_seconds=5.0, max_age_seconds=5.0)]


@cfg_registry.register
class g1_ufo_video_real_file(g1_ufo_video_file):
    env: G1UfoM0RealEnvCfg = G1UfoM0RealEnvCfg()
    ctrl: list = [UnitreeCtrlCfg(triggers={}, combination_init_buttons=[]), LatentCtrlCfg(source="file")]
    run_fullspeed: bool = False
    do_safety_check: bool = True
    video_real_enabled: bool = False  # set only after the selected variant passes sim and latency review


@cfg_registry.register
class g1_ufo_video_real_stream(g1_ufo_video_real_file):
    auto_debug_mode: bool = True
    video_real_enabled: bool = True
    ctrl: list = [UnitreeCtrlCfg(triggers={}, combination_init_buttons=[]),
                  LatentCtrlCfg(source="tcp", stale_seconds=5.0, max_age_seconds=5.0)]


@cfg_registry.register
class g1_ufo_video_real_workstation(g1_ufo_video_real_stream):
    env: G1UfoWorkstationEnvCfg = G1UfoWorkstationEnvCfg()
    # Allow intentional lying poses. DDS validity/watchdogs, joint guards and
    # operator stop remain independent of this pipeline attitude check.
    do_safety_check: bool = False
    ctrl: list = [UnitreeCtrlCfg(triggers={}, combination_init_buttons=[]),
                  LatentCtrlCfg(source="socket", stale_seconds=5.0, max_age_seconds=5.0)]


@cfg_registry.register
class g1_ufo_video_real_workstation_amo(g1_ufo_video_real_workstation):
    pipeline_type: str = "UfoAmoPipeline"
    amo_policy: G1AmoPolicyCfg = G1AmoPolicyCfg()
    # AMO needs upright feedback. The original UFO-only lying-pose option is unchanged.
    do_safety_check: bool = True
    init_seconds: float = Field(default=3.0, gt=0)
    switch_seconds: float = Field(default=0.5, gt=0)
    settle_seconds: float = Field(default=0.5, gt=0)
    neutral_seconds: float = Field(default=0.5, gt=0)
    stick_deadzone: float = Field(default=0.1, gt=0, lt=1)


@cfg_registry.register
class g1_ufo_video_camera_amo(g1_ufo_video_real_workstation_amo):
    env: G1UfoM0MujocoEnvCfg = G1UfoM0MujocoEnvCfg(render=True)
    auto_debug_mode: bool = False
    video_real_enabled: bool = False
    sim_initial_support: bool = True
    ctrl: list = [KeyboardCtrlCfg(triggers={}),
                  LatentCtrlCfg(source="socket", stale_seconds=5.0, max_age_seconds=5.0)]
