"""LAFAN G1 reference motions in the observation convention of UFO M0."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# The source NPZ is ordered by kinematic depth, with left and right interleaved.
# Value i is the UFO/RoboJuDo joint index represented by source column i.
NPZ_TO_JOINT = np.asarray(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
     5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

NPZ_BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link",
    "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link", "left_shoulder_roll_link",
    "right_shoulder_roll_link", "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link", "left_elbow_link",
    "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link", "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

UFO_BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
    "right_wrist_pitch_link", "right_wrist_yaw_link",
)


class UfoM0Motion:
    """Loads a retargeted G1 NPZ and constructs the B encoder's raw inputs."""

    def __init__(self, path: str | Path, default_pos: np.ndarray, start_frame: int = 0, max_frames: int | None = None):
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        if start_frame < 0 or (max_frames is not None and max_frames < 2):
            raise ValueError("start_frame must be >= 0 and max_frames must be >= 2")
        stop = None if max_frames is None else start_frame + max_frames
        with np.load(path, allow_pickle=False) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if fps != 50.0:
                raise ValueError(f"UFO M0 expects a 50 Hz reference; got {fps} Hz from {path}")
            dof_pos_src = np.asarray(data["joint_pos"][start_frame:stop], dtype=np.float32)
            dof_vel_src = np.asarray(data["joint_vel"][start_frame:stop], dtype=np.float32)
            arrays = [
                np.asarray(data[key][start_frame:stop], dtype=np.float32)
                for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
            ]
        n = dof_pos_src.shape[0]
        if n < 2 or dof_pos_src.shape != (n, 29) or dof_vel_src.shape != (n, 29):
            raise ValueError(f"Expected at least two frames of [29] joint positions and velocities in {path}")
        expected = ((n, 30, 3), (n, 30, 4), (n, 30, 3), (n, 30, 3))
        if any(value.shape != shape for value, shape in zip(arrays, expected)):
            raise ValueError(f"G1 body array shapes differ from {expected} in {path}")
        if not all(np.all(np.isfinite(value)) for value in [dof_pos_src, dof_vel_src, *arrays]):
            raise ValueError(f"Non-finite reference values in {path}")
        if not np.allclose(np.linalg.norm(arrays[1], axis=-1), 1.0, atol=2e-3):
            raise ValueError(f"Non-unit body quaternions in {path}")

        self.path = path
        self.fps = fps
        self.num_frames = n
        self.default_pos = np.asarray(default_pos, dtype=np.float32)
        if self.default_pos.shape != (29,):
            raise ValueError("UFO M0 needs 29 default joint positions")
        self.dof_pos = np.empty_like(dof_pos_src)
        self.dof_vel = np.empty_like(dof_vel_src)
        self.dof_pos[:, NPZ_TO_JOINT] = dof_pos_src
        self.dof_vel[:, NPZ_TO_JOINT] = dof_vel_src

        body_indices = [NPZ_BODY_NAMES.index(name) for name in UFO_BODY_NAMES]
        body_pos, body_quat_wxyz, body_vel, body_ang_vel = (value[:, body_indices] for value in arrays)
        body_quat = body_quat_wxyz[..., [1, 2, 3, 0]]

        # UFO adds a virtual head 0.35 m above the torso in the torso frame.
        torso_rot = Rotation.from_quat(body_quat[:, 15])
        head_offset = torso_rot.apply(np.array([0.0, 0.0, 0.35]))
        head_pos = body_pos[:, 15] + head_offset
        head_vel = body_vel[:, 15] + np.cross(body_ang_vel[:, 15], head_offset)
        self.body_pos = np.concatenate([body_pos, head_pos[:, None]], axis=1).astype(np.float32)
        self.body_quat = np.concatenate([body_quat, body_quat[:, 15:16]], axis=1).astype(np.float32)
        self.body_vel = np.concatenate([body_vel, head_vel[:, None]], axis=1).astype(np.float32)
        self.body_ang_vel = np.concatenate([body_ang_vel, body_ang_vel[:, 15:16]], axis=1).astype(np.float32)

    def first_qpos(self) -> np.ndarray:
        return self.qpos_at(0)

    def qpos_at(self, frame: int) -> np.ndarray:
        """Return a complete model-order MuJoCo pose for visualization."""
        root_quat_wxyz = self.body_quat[frame, 0, [3, 0, 1, 2]]
        return np.concatenate([self.body_pos[frame, 0], root_quat_wxyz,
                               self.dof_pos[frame]]).astype(np.float64)

    def first_qvel(self) -> np.ndarray:
        local_ang = Rotation.from_quat(self.body_quat[0, 0]).inv().apply(self.body_ang_vel[0, 0])
        return np.concatenate([self.body_vel[0, 0], local_ang, self.dof_vel[0]]).astype(np.float64)

    def backward_inputs(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Return raw [N,64] state and [N,463] privileged state for B."""
        pos = self.body_pos[start:stop]
        quat = self.body_quat[start:stop]
        vel = self.body_vel[start:stop]
        ang = self.body_ang_vel[start:stop]
        n = len(pos)
        root_rot = Rotation.from_quat(quat[:, 0])
        forward = root_rot.apply(np.tile([1.0, 0.0, 0.0], (n, 1)))
        yaw = np.arctan2(forward[:, 1], forward[:, 0])
        heading_inv = Rotation.from_euler("z", (-yaw)[:, None])
        gravity = root_rot.inv().apply(np.tile([0.0, 0.0, -1.0], (n, 1)))
        root_ang_local = root_rot.inv().apply(ang[:, 0]) * 0.25
        state = np.concatenate([self.dof_pos[start:stop] - self.default_pos,
                                self.dof_vel[start:stop], gravity, root_ang_local], axis=1)

        def yaw_rotate(values: np.ndarray) -> np.ndarray:
            count = values.shape[1]
            rotations = Rotation.from_quat(np.repeat(heading_inv.as_quat(), count, axis=0))
            return rotations.apply(values.reshape(-1, 3)).reshape(n, count, 3)

        local_pos = yaw_rotate(pos[:, 1:] - pos[:, :1])
        heading_all = Rotation.from_quat(np.repeat(heading_inv.as_quat(), 31, axis=0))
        matrix = (heading_all * Rotation.from_quat(quat.reshape(-1, 4))).as_matrix().reshape(n, 31, 3, 3)
        rot6 = np.concatenate([matrix[..., :, 0], matrix[..., :, 2]], axis=-1)
        local_vel = yaw_rotate(vel)
        local_ang = yaw_rotate(ang)
        privileged = np.concatenate([
            pos[:, 0, 2:3], local_pos.reshape(n, -1), rot6.reshape(n, -1),
            local_vel.reshape(n, -1), local_ang.reshape(n, -1),
        ], axis=1)
        if state.shape != (n, 64) or privileged.shape != (n, 463):
            raise AssertionError("UFO M0 reference observation dimensions changed")
        return state.astype(np.float32), privileged.astype(np.float32)
