"""Workstation G1 control: one guarded HG command per explicit step, no replay thread."""

import os
import time
from collections import deque

import numpy as np

from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.environment import Environment, env_registry
from robojudo.environment.utils.g1_dds import G1DdsError, G1StateReader
from robojudo.environment.utils.real_target_guard import RealTargetGuard


@env_registry.register
class UnitreeG1DdsEnv(Environment):
    def __init__(self, cfg_env, device="cpu"):
        if cfg_env.act and os.environ.get("UFO_REAL_ROBOT_OK") != "1":
            raise RuntimeError("G1 DDS actuation requires UFO_REAL_ROBOT_OK=1")
        super().__init__(cfg_env, device)
        names = G1_29DoF().joint_names
        if self.num_dofs != 29 or set(self.joint_names) != set(names):
            raise ValueError("Workstation DDS requires all 29 G1 joints")
        self._indices = np.array([names.index(name) for name in self.joint_names])
        if cfg_env.joint2motor_idx is not None and list(self._indices) != cfg_env.joint2motor_idx:
            raise ValueError("joint2motor_idx differs from the named G1 motor mapping")
        self.enabled = cfg_env.act
        self.RemoteControllerHandler = None
        self.target_guard = self._make_guard()
        self.fault_reason = None
        self._closed = False
        self._cycle_started = None
        self._last_update = None
        self._reader = None
        self._publisher = None
        self._has_sent = False
        self._samples = deque(maxlen=4096)
        self._stats = dict(commands_sent=0, holds_sent=0, send_failures=0)
        try:
            self._reader = G1StateReader(
                cfg_env.unitree.net_if, domain_id=cfg_env.dds_domain_id,
                feedback_timeout_s=cfg_env.feedback_timeout_s, topic=cfg_env.unitree.lowstate_topic)
            self.self_check()
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.utils.crc import CRC

            self._new_command = unitree_hg_msg_dds__LowCmd_
            self._crc = CRC()
            self._publisher = ChannelPublisher(cfg_env.unitree.lowcmd_topic, LowCmd_)
            self._publisher.Init()
        except BaseException:
            self.shutdown()
            raise

    def _make_guard(self):
        return RealTargetGuard(self.position_limits, self.cfg_env.target_velocity_limits,
                               self.cfg_env.unitree.control_dt, self.cfg_env.target_slew_safety_factor)

    def update_dof_cfg(self, override_cfg=None):
        super().update_dof_cfg(override_cfg)
        if getattr(self, "target_guard", None) is not None:
            previous = self.target_guard.last_command
            self.target_guard = self._make_guard()
            self.target_guard.last_command = previous

    def set_gains(self, stiffness, damping):
        self._kp = np.asarray(stiffness, dtype=np.float64).copy()
        self._kd = np.asarray(damping, dtype=np.float64).copy()
        if (self._kp.shape != (29,) or self._kd.shape != (29,)
                or not np.isfinite(self._kp).all() or not np.isfinite(self._kd).all()
                or np.any(self._kp <= 0) or np.any(self._kd < 0)):
            raise ValueError("Expected finite positive 29-joint stiffness and nonnegative damping")

    def self_check(self):
        self._reader.wait(self.cfg_env.connect_timeout_s)

    def reset(self):
        # Never clear a transport/control fault by resetting policy history.
        self._cycle_started = None

    def update(self):
        now = time.monotonic()
        try:
            state = self._reader.latest()
        except Exception as exc:
            self._fault(str(exc))
            raise
        self._cycle_started = now
        self._cycle_state = state
        self._cycle_period = None if self._last_update is None else now - self._last_update
        self._last_update = now
        self._dof_pos = state.q[self._indices].copy()
        self._dof_vel = state.dq[self._indices].copy()
        self._base_quat = state.quaternion[[1, 2, 3, 0]].copy()
        self._base_ang_vel = state.gyroscope.copy()
        if self.RemoteControllerHandler is not None:
            self.RemoteControllerHandler(state.remote)

    def _fault(self, reason):
        if self.fault_reason is None:
            self.fault_reason = reason

    def _publish(self, target, *, hold=False):
        if self._closed or not self.enabled:
            return False
        state = self._reader.latest()
        cmd = self._new_command()
        cmd.mode_pr = 0
        cmd.mode_machine = state.mode_machine
        for logical, motor in enumerate(self._indices):
            out = cmd.motor_cmd[int(motor)]
            out.mode = 1
            out.q = float(target[logical])
            out.dq = 0.0
            out.tau = 0.0
            out.kp = float(self._kp[logical])
            out.kd = float(self._kd[logical])
        cmd.crc = self._crc.Crc(cmd)
        # Recheck after packing/CRC; never submit a late policy result.
        self._reader.latest()
        now = time.monotonic()
        if not hold and (self._cycle_started is None
                         or now - self._cycle_started > self.cfg_env.command_timeout_s):
            raise G1DdsError("G1 command computation exceeded its deadline")
        # The SDK's timeout option polls matching in 100ms sleeps; do not use
        # that blocking path in a 20ms control cycle. Feedback is our watchdog.
        if not self._publisher.Write(cmd):
            self._stats["send_failures"] += 1
            raise G1DdsError("G1 DDS command write failed")
        self._has_sent = True
        self._stats["commands_sent"] += 1
        self._stats["holds_sent"] += int(hold)
        self._samples.append({"sent_at": now, "feedback_age_s": now - state.received_at,
                              "cycle_period_s": getattr(self, "_cycle_period", None),
                              "compute_s": None if self._cycle_started is None else now - self._cycle_started,
                              "hold": hold, "tick": state.tick})
        return True

    def step(self, pd_target, hand_pose=None):
        try:
            if self.fault_reason:
                raise G1DdsError(f"G1 DDS fault is latched: {self.fault_reason}")
            if hand_pose is not None:
                raise ValueError("Workstation DDS does not control hands")
            if self._cycle_started is None:
                raise G1DdsError("Call update before sending a G1 target")
            self._reader.latest()
            target = self.target_guard.filter(pd_target, self._cycle_state.q[self._indices])
            self._publish(target)
        except Exception as exc:
            self._fault(str(exc))
            raise

    def send_hold(self):
        if not self._has_sent:
            return False
        try:
            state = self._reader.latest()
            return self._publish(self.target_guard.hold(state.q[self._indices]), hold=True)
        except Exception as exc:
            self._fault(str(exc))
            return False

    def metrics(self):
        return {**self._stats, "fault_reason": self.fault_reason,
                "telemetry": self._reader.metrics() if self._reader else {},
                "samples": list(self._samples)}

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self.enabled = False
        # The pipeline's safe_stop owns the best-effort fresh-feedback hold.
        # Closing a never-armed environment must not emit any motor command.
        try:
            if self._publisher is not None:
                self._publisher.Close()
        finally:
            if self._reader is not None:
                self._reader.close()
