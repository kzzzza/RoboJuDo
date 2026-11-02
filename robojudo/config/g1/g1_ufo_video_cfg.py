"""Video-derived UFO M0 latents with shared file/socket controller."""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import LatentCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.g1_ufo_m0_env_cfg import G1UfoM0MujocoEnvCfg, G1UfoM0RealEnvCfg
from .policy.g1_ufo_m0_policy_cfg import G1UfoM0PolicyCfg


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
    ctrl: list = [LatentCtrlCfg(source="socket", stale_seconds=0.75)]
    run_fullspeed: bool = False


@cfg_registry.register
class g1_ufo_video_camera(g1_ufo_video_stream):
    """Live camera demo: show the robot and allow delayed RGB inference."""

    env: G1UfoM0MujocoEnvCfg = G1UfoM0MujocoEnvCfg(render=True)
    ctrl: list = [LatentCtrlCfg(source="socket", stale_seconds=30.0, max_age_seconds=300.0)]


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
