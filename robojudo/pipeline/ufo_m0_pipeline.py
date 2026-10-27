"""RoboJuDo RL pipeline initialized from a UFO reference motion."""

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_pipeline import RlPipeline


@pipeline_registry.register
class UfoM0Pipeline(RlPipeline):
    def reset(self):
        super().reset()
        if self.cfg.env.is_sim:
            motion = self.policy.policy.motion
            self.env.reborn(motion.first_qpos(), motion.first_qvel())
            self.env.update()
