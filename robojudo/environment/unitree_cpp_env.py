import logging
import time

import numpy as np
from unitree_cpp import RobotState, SportState, UnitreeController  # type: ignore

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeEnvCfg
from robojudo.environment.utils.real_target_guard import RealTargetGuard
from robojudo.tools.retarget import HandRetarget
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@env_registry.register
class UnitreeCppEnv(Environment):
    cfg_env: UnitreeEnvCfg

    def __init__(self, cfg_env: UnitreeEnvCfg, device="cpu"):
        self.enabled: bool = cfg_env.act
        super().__init__(cfg_env=cfg_env, device=device)
        self.RemoteControllerHandler = None
        self.target_guard = (
            RealTargetGuard(
                self.position_limits,
                cfg_env.target_velocity_limits,
                cfg_env.unitree.control_dt,
                cfg_env.target_slew_safety_factor,
            )
            if cfg_env.target_velocity_limits is not None
            else None
        )

        cfg_unitree: UnitreeEnvCfg.UnitreeCfg = cfg_env.unitree

        cfg_unitree_dict: dict = cfg_unitree.to_dict()
        cfg_unitree_dict["num_dofs"] = self.num_dofs
        cfg_unitree_dict["stiffness"] = self.stiffness
        cfg_unitree_dict["damping"] = self.damping

        self.robot = cfg_unitree.robot
        self._dof_idx = cfg_env.joint2motor_idx
        self._odometry_type = cfg_env.odometry_type
        if self._odometry_type == "ZED":
            assert self.cfg_env.zed_cfg is not None, "zed_cfg must be set if odometry_type is 'ZED'"
            from robojudo.tools.zed_odometry import ZedOdometry

            self.zed_odometry = ZedOdometry(self.cfg_env.zed_cfg)
        elif self._odometry_type == "DUMMY":
            pass
        elif self._odometry_type == "UNITREE":
            pass

        self.hand_type = cfg_unitree.hand_type
        if self.hand_type == "Inspire":
            self.hand_retarget = HandRetarget(cfg_env.hand_retarget)
        elif self.hand_type == "Dex-3":
            self.hand_retarget = None  # TODO
        else:
            self.hand_retarget = None

        self.sport_state: SportState = None  # pyright: ignore[reportAttributeAccessIssue]
        self.robot_state: RobotState = None  # pyright: ignore[reportAttributeAccessIssue]

        self.unitree = UnitreeController(cfg_unitree_dict)

        # born place alignment extra for h1 torso
        if self.robot == "h1":
            self.torso_align = TransformAlignment()

        # time.sleep(1)  # wait for unitree init
        try:
            self.self_check()
        except BaseException:
            try:
                self.unitree.shutdown()
            except Exception:
                logger.exception("UnitreeCppEnv cleanup failed after initialization error")
            raise

    def self_check(self):
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                logger.info("UnitreeCppEnv self check passed!")
                return
        logger.critical("UnitreeCppEnv self check failed!")
        raise RuntimeError("UnitreeCppEnv self check failed")

    def update_dof_cfg(self, override_cfg=None):
        super().update_dof_cfg(override_cfg=override_cfg)
        if getattr(self, "target_guard", None) is not None:
            # PolicyWrapper may override limits after environment construction.
            last_command = self.target_guard.last_command
            self.target_guard = RealTargetGuard(
                self.position_limits,
                self.cfg_env.target_velocity_limits,
                self.cfg_env.unitree.control_dt,
                self.cfg_env.target_slew_safety_factor,
            )
            self.target_guard.last_command = last_command

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

        if self.robot == "h1":
            self.torso_align.set_base(quat=self.torso_quat)

        if self._odometry_type == "ZED":
            self.zed_odometry.set_zreo()

    def update(self):
        # robot state
        self.robot_state = self.unitree.get_robot_state()
        if self._dof_idx is None:
            self._dof_pos = np.array(self.robot_state.motor_state.q, dtype=np.float32)
            self._dof_vel = np.array(self.robot_state.motor_state.dq, dtype=np.float32)
        else:
            self._dof_pos = np.array(
                [self.robot_state.motor_state.q[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )
            self._dof_vel = np.array(
                [self.robot_state.motor_state.dq[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )

        if self.robot == "g1":
            quat = np.array(self.robot_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            ang_vel = np.array(self.robot_state.imu_state.gyroscope, dtype=np.float32)
            rpy = np.array(self.robot_state.imu_state.rpy, dtype=np.float32)

            if self.born_place_align:
                quat = self.base_align.align_quat(quat)

            self._base_quat = quat
            self._base_ang_vel = ang_vel
            self._base_rpy = rpy

        elif self.robot == "h1":
            raise NotImplementedError("H1 robot with unitree_cpp not supported yet.")

        # odometry
        if self._odometry_type == "ZED":
            self.zed_odometry.update()
            if self.zed_odometry.is_valid:
                # born place aligned in zed_odometry
                self._base_pos = self.zed_odometry.pos
                self._base_lin_vel = self.zed_odometry.lin_vel
        elif self._odometry_type == "DUMMY":
            self._base_pos = np.array([0.0, 0.0, 0.8])
            self._base_lin_vel = np.array([0.0, 0.0, 0.0])
        elif self._odometry_type == "UNITREE":
            self.sport_state = self.unitree.get_sport_state()
            base_pos = np.asarray(self.sport_state.position, dtype=np.float32)
            lin_vel = np.asarray(self.sport_state.velocity, dtype=np.float32)
            self._base_lin_vel = quat_rotate_inverse_np(self.base_quat, lin_vel)
            self._base_pos = self.base_align.align_pos(base_pos) if self.born_place_align else base_pos

        # FK
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            if self.robot != "h1":
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        # controller
        if self.RemoteControllerHandler:
            self.RemoteControllerHandler(self.robot_state.wireless_remote)

    def step(self, pd_target, hand_pose=None):
        if self.target_guard is not None:
            if self.robot_state is None:
                raise ValueError("Cannot send a guarded target before receiving robot state")
            positions = self.target_guard.filter(pd_target, self.dof_pos)
        else:
            assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
            positions = pd_target
        if self.enabled:
            self.unitree.step(positions.tolist())

        if hand_pose is not None:
            assert type(hand_pose) is np.ndarray, "hand_pose should be a numpy array"
            assert hand_pose.shape[0] == 2, "hand_pose should be of shape (2, -1)"
            if self.hand_retarget is not None:
                hand_pose = self.hand_retarget.from_pose_to_cmd(hand_pose)
                logger.debug(f"Hand pose retargeted: {hand_pose}")
            hand_pose = hand_pose.tolist()

            if self.enabled:
                self.unitree.step_hands(hand_pose[0], hand_pose[1])

    def send_hold(self) -> bool:
        """Send the latest measured joint positions when feedback is valid."""
        if not self.enabled or self.robot_state is None:
            return False
        try:
            if self.target_guard is not None:
                positions = self.target_guard.hold(self.dof_pos)
            else:
                positions = np.asarray(self.dof_pos, dtype=np.float64)
                if positions.shape != (self.num_dofs,) or not np.all(np.isfinite(positions)):
                    return False
                positions = np.clip(positions, self.position_limits[:, 0], self.position_limits[:, 1])
            self.unitree.step(positions.tolist())
            return True
        except Exception as exc:
            logger.warning("Could not send measured-position hold: %s", exc)
            return False

    def shutdown(self):
        # self.set_damping_mode()
        self.enabled = False
        self.unitree.shutdown()

    def set_gains(self, stiffness, damping):
        if not hasattr(self, "unitree"):  # TODO
            return
        if not self.enabled:
            return
        self.unitree.set_gains(stiffness, damping)


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg

    env = UnitreeCppEnv(cfg_env=G1RealEnvCfg())
    env.set_gains(
        stiffness=[kp * 0.0 for kp in env.stiffness],
        damping=[kd * 0.1 for kd in env.damping],
    )
    while 1:
        # env.step(np.zeros(29), np.ones((2, 7)) * -0)
        env.step(np.zeros(29), None)
        # if controller.remote_controller("A"):
        #     controller.shutdown()
        print(env.base_rpy)
        print(env.dof_pos)
        print(env.base_pos)
        env.update()
        # print(env.base_pos)
        time.sleep(0.1)
    # print("Exit")
    # from robojudo.controller import UnitreeCtrl
    # ctrl = UnitreeCtrl(env=env)

    # while True:
    #     env.update()
    #     state = ctrl.get_state()
    #     events = ctrl.get_events()
    #     print("State:", state)
    #     print("Events:", events)
    #     time.sleep(0.1)  # Simulate a control loop
