import logging
import os
import time

import mujoco
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_render import (
    MujocoVideoRecorder, ReferenceGhost, load_model,
)
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        if cfg_env.render and cfg_env.video_path:
            raise ValueError("Window rendering and headless recording cannot run together")
        if cfg_env.video_path and os.environ.get("MUJOCO_GL") != "egl":
            raise RuntimeError("Set MUJOCO_GL=egl before importing MuJoCo for headless video")

        self.model = load_model(cfg_env.xml, cfg_env.scene_style)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        needs_reference = (cfg_env.render and cfg_env.show_reference) or (
            cfg_env.video_path is not None and cfg_env.video_content == "comparison"
        )
        self.reference_ghost = (
            ReferenceGhost(self.model, cfg_env.reference_root_body) if needs_reference else None
        )
        self.viewer = None
        if cfg_env.render:
            import mujoco_viewer

            self.viewer = mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
                width=1200,
                height=900,
                hide_menus=True,
                diable_key_callbacks=True,
            )
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -10.0
            self.viewer.cam.azimuth = 180.0
            if cfg_env.show_reference:
                self.reference_ghost.option = self.viewer.vopt
                self.viewer.scene_hook = self.reference_ghost.append
        # self.viewer._paused = True

        self.video_recorder = None
        if cfg_env.video_path:
            self.video_recorder = MujocoVideoRecorder(
                self.model, cfg_env.video_path, 1.0 / self.control_dt,
                cfg_env.video_width, cfg_env.video_height,
                ghost=self.reference_ghost if cfg_env.video_content == "comparison" else None,
            )

        if cfg_env.visualize_extras and self.viewer is not None:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()

        self.update()  # get initial state

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None, init_qvel=None):
        if init_qpos is not None:
            qpos = np.asarray(init_qpos)
            if qpos.shape == (7,):
                self.data.qpos[0:7] = qpos
            elif qpos.shape == (self.model.nq,):
                self.data.qpos[:] = qpos
            else:
                raise ValueError(f"init_qpos must have 7 or {self.model.nq} elements")
            if init_qvel is None:
                self.data.qvel[:] = 0.0
            else:
                qvel = np.asarray(init_qvel)
                if qvel.shape != (self.model.nv,):
                    raise ValueError(f"init_qvel must have {self.model.nv} elements")
                self.data.qvel[:] = qvel
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
        dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        if self.reference_ghost is not None and not self.reference_ghost.has_pose:
            raise ValueError("Reference visualization requires reference_qpos on every frame")

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

        self.render_frame()
        if self.video_recorder is not None:
            self.video_recorder.write_frame(self.data)

    def render_frame(self):
        """Draw the current state without advancing physics (for source warmup)."""
        if self.viewer is not None and self.viewer.is_alive:
            self.viewer.cam.lookat = self.data.qpos[:3]
            self.viewer.render()

    def set_reference_qpos(self, qpos):
        if self.reference_ghost is not None:
            self.reference_ghost.set_pose(qpos)

    def shutdown(self):
        try:
            if self.viewer is not None and self.viewer.is_alive:
                self.viewer.close()
        finally:
            if self.video_recorder is not None:
                self.video_recorder.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
