"""UFO M0 G1 actor and backward encoder, using RoboJuDo's PD interface."""

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from robojudo.policy import Policy, policy_registry
from robojudo.utils.ufo_m0_motion import UfoM0Motion

logger = logging.getLogger(__name__)

HISTORY_KEYS = ("actions", "base_ang_vel", "dof_pos", "dof_vel", "projected_gravity")


@policy_registry.register
class UfoM0Policy(Policy):
    def __init__(self, cfg_policy, device="cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        model_dir = Path(cfg_policy.model_dir).expanduser()
        meta = json.loads((model_dir / "FBcprAuxModel.meta.json").read_text())
        if meta["control_joint_names"] != self.cfg_obs_dof.joint_names:
            raise ValueError("UFO model joint names do not match the policy's observation joint order")
        if self.cfg_action_dof.joint_names != self.cfg_obs_dof.joint_names:
            raise ValueError("UFO actor action and observation joint orders must agree")
        if meta["actor_obs_dim"] != 721 or meta["z_dim"] != 256:
            raise ValueError("Expected the original UFO G1 M0 actor with 721 inputs and 256 latent values")

        self.motion = UfoM0Motion(
            cfg_policy.motion_path,
            default_pos=self.default_dof_pos,
            start_frame=cfg_policy.start_frame,
            max_frames=cfg_policy.max_frames,
        )
        providers = ["CPUExecutionProvider"]
        self.backward_session = ort.InferenceSession(str(model_dir / "backward_encoder.onnx"), providers=providers)
        self.actor_session = ort.InferenceSession(str(model_dir / "FBcprAuxModel.onnx"), providers=providers)
        if {item.name: item.shape[-1] for item in self.backward_session.get_inputs()} != {
            "state": 64, "privileged_state": 463,
        }:
            raise ValueError("Unexpected UFO M0 backward encoder inputs")
        if [(item.name, item.shape[-1]) for item in self.actor_session.get_inputs()] != [("actor_obs", 721)]:
            raise ValueError("Unexpected UFO M0 actor input")

        # Reference B encoding is independent of live robot state. UFO tracks
        # reference frame t+1 at control step t, so omit reference frame zero.
        zs = []
        for start in range(1, self.motion.num_frames, 512):
            state, privileged = self.motion.backward_inputs(start, min(start + 512, self.motion.num_frames))
            z = self.backward_session.run(["z"], {"state": state, "privileged_state": privileged})[0]
            zs.append(np.asarray(z, dtype=np.float32))
        self.z = np.concatenate(zs, axis=0)
        if self.z.shape != (self.motion.num_frames - 1, 256) or not np.all(np.isfinite(self.z)):
            raise ValueError("Invalid UFO M0 reference embedding")

        self.effort_limits = np.asarray(cfg_policy.effort_limits, dtype=np.float32)
        self.kp = np.asarray(self.cfg_action_dof.stiffness, dtype=np.float32)
        if self.effort_limits.shape != (29,) or self.kp.shape != (29,) or np.any(self.kp <= 0):
            raise ValueError("UFO M0 requires positive 29-joint stiffness and effort limits")
        self.target_scale = 0.25 * self.effort_limits / self.kp
        self.reset()
        logger.info("Loaded UFO M0 motion %s (%d frames)", self.motion.path, self.motion.num_frames)

    def reset(self):
        self.frame = 0
        self.last_action = np.zeros(29, dtype=np.float32)
        self.history = {
            key: deque((np.zeros(size, dtype=np.float32) for _ in range(4)), maxlen=4)
            for key, size in (("actions", 29), ("base_ang_vel", 3), ("dof_pos", 29),
                              ("dof_vel", 29), ("projected_gravity", 3))
        }

    def post_step_callback(self, commands=None):
        if any(cmd in ("[MOTION_RESET]", "[MOTION_FADE_IN]") for cmd in commands or []):
            self.reset()
        else:
            self.frame = min(self.frame + 1, self.motion.num_frames - 2)

    def get_init_dof_pos(self) -> np.ndarray:
        return self.motion.dof_pos[0].copy()

    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32) - self.default_dof_pos
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        base_quat = np.asarray(env_data.base_quat, dtype=np.float32)
        gravity = Rotation.from_quat(base_quat).inv().apply([0.0, 0.0, -1.0]).astype(np.float32)
        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32) * 0.25
        values = {
            "actions": self.last_action,
            "base_ang_vel": base_ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "projected_gravity": gravity,
        }
        state = np.concatenate([dof_pos, dof_vel, gravity, base_ang_vel])
        history = np.concatenate([np.concatenate(list(self.history[key])) for key in HISTORY_KEYS])
        obs = np.concatenate([state, self.last_action, history, self.z[self.frame]])
        for key in HISTORY_KEYS:
            self.history[key].append(values[key].copy())
        if obs.shape != (721,) or not np.all(np.isfinite(obs)):
            raise ValueError("Invalid UFO M0 actor observation")
        return obs.astype(np.float32), {
            "reference_frame": self.frame + 1,
            "reference_qpos": self.motion.qpos_at(self.frame + 1),
        }

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        raw = self.actor_session.run(["action"], {"actor_obs": obs[None].astype(np.float32)})[0]
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.shape != (29,) or not np.all(np.isfinite(raw)):
            raise ValueError("Invalid UFO M0 actor action")
        # UFO stores the normalized and clipped environment action in its
        # last_action/history buffers, while its PD target uses that same value.
        self.last_action = np.clip(raw * 5.0, -5.0, 5.0)
        return self.last_action * self.target_scale
