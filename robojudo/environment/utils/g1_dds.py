"""Read-only, timestamped G1 telemetry using the official SDK's HG messages."""

import threading
import time
from dataclasses import dataclass

import numpy as np


class G1DdsError(RuntimeError):
    """A latched control fault or unavailable/faulty G1 feedback."""


@dataclass(frozen=True)
class G1Snapshot:
    tick: int
    mode_machine: int
    received_at: float
    q: np.ndarray
    dq: np.ndarray
    quaternion: np.ndarray
    gyroscope: np.ndarray
    remote: bytes


class G1StateReader:
    """Never publishes commands or switches the robot's motion mode.

    mode_machine 2 is the legacy 29DOF ID, 5 the unlocked rev_1_0 ID.
    Newer hardware variants, locked-waist/23DOF/unknown machines are
    deliberately rejected until their model/actuator parameters are checked.
    """

    def __init__(self, net_if, *, domain_id=0, feedback_timeout_s=0.1,
                 topic="rt/lowstate", clock=time.monotonic):
        if not net_if:
            raise ValueError("An explicit G1 network interface is required")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC

        self.clock = clock
        self.feedback_timeout_s = feedback_timeout_s
        self._condition = threading.Condition()
        self._snapshot = None
        self._progress = 0
        self._closed = False
        self._fatal = None
        self._stats = dict(accepted=0, crc_errors=0, old_ticks=0, invalid_states=0)
        self._crc = CRC()
        ChannelFactoryInitialize(domain_id, net_if)
        self._subscriber = ChannelSubscriber(topic, LowState_)
        try:
            self._subscriber.Init(self._receive, 1)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _array(value, size):
        result = np.asarray(value, dtype=np.float64).copy()
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"Expected {size} finite feedback values")
        result.setflags(write=False)
        return result

    def _receive(self, msg):
        with self._condition:
            if self._closed or self._fatal:
                return
            try:
                if self._crc.Crc(msg) != msg.crc:
                    self._stats["crc_errors"] += 1
                    return
                if msg.mode_machine not in (2, 5):
                    raise ValueError(f"Expected unlocked 29DOF G1, mode_machine={msg.mode_machine}")
                if msg.mode_pr != 0:
                    raise ValueError("G1 feedback must be in PR mode")
                if self._snapshot and msg.mode_machine != self._snapshot.mode_machine:
                    raise ValueError("G1 machine type changed during the session")
                tick = int(msg.tick)
                if self._snapshot:
                    delta = (tick - self._snapshot.tick) & 0xFFFFFFFF
                    if not 0 < delta < 0x80000000:
                        self._stats["old_ticks"] += 1
                        return
                q = self._array([m.q for m in msg.motor_state[:29]], 29)
                dq = self._array([m.dq for m in msg.motor_state[:29]], 29)
                quat = self._array(msg.imu_state.quaternion, 4)
                if abs(np.linalg.norm(quat) - 1.0) > 0.05:
                    raise ValueError("Invalid G1 IMU quaternion norm")
                gyro = self._array(msg.imu_state.gyroscope, 3)
                remote = bytes(msg.wireless_remote)
                if len(remote) != 40:
                    raise ValueError("Expected 40 remote-control bytes")
                if any(m.motorstate for m in msg.motor_state[:29]):
                    raise ValueError("G1 reports a joint motor fault")
                self._snapshot = G1Snapshot(tick, int(msg.mode_machine), self.clock(), q, dq, quat, gyro, remote)
                self._progress += 1
                self._stats["accepted"] += 1
            except Exception as exc:
                self._fatal = str(exc)
                self._stats["invalid_states"] += 1
            self._condition.notify_all()

    def latest(self):
        with self._condition:
            if self._closed:
                raise G1DdsError("G1 telemetry reader is closed")
            if self._fatal:
                raise G1DdsError(self._fatal)
            if self._snapshot is None or self._progress < 2:
                raise G1DdsError("Waiting for advancing G1 lowstate ticks")
            age = self.clock() - self._snapshot.received_at
            if age > self.feedback_timeout_s:
                raise G1DdsError(f"G1 lowstate stale: {age:.3f}s > {self.feedback_timeout_s:.3f}s")
            return self._snapshot

    def wait(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._closed:
                    raise G1DdsError("G1 telemetry reader is closed")
                if self._fatal:
                    raise G1DdsError(self._fatal)
                try:
                    return self.latest()
                except G1DdsError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise G1DdsError("G1 lowstate connection timed out; check interface and DDS discovery")
                    self._condition.wait(min(remaining, 0.02))

    def metrics(self):
        with self._condition:
            return {**self._stats, "feedback_fault": self._fatal}

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        if getattr(self, "_subscriber", None) is not None:
            self._subscriber.Close()
