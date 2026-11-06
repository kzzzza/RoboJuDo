"""Official HG message/CRC contracts, with transport replaced by local fakes."""

import importlib.util
import os
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from robojudo.config.config_manager import ConfigManager
from robojudo.environment.unitree_g1_dds_env import UnitreeG1DdsEnv
from robojudo.environment.utils.g1_dds import G1DdsError, G1StateReader


def state(tick=1, mode=5):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.utils.crc import CRC

    msg = unitree_hg_msg_dds__LowState_()
    msg.mode_machine = mode
    msg.tick = tick
    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    for i, motor in enumerate(msg.motor_state):
        motor.q = i * 0.001
        motor.dq = i * 0.002
    msg.crc = CRC().Crc(msg)
    return msg


class Subscriber:
    def __init__(self, *args):
        self.closed = False

    def Init(self, handler, queue_length):
        self.handler = handler
        handler(state(1))
        handler(state(2))

    def Close(self):
        self.closed = True


class Publisher:
    def __init__(self, *args):
        self.commands = []
        self.succeed = True
        self.closed = False

    def Init(self):
        pass

    def Write(self, msg):
        if self.succeed:
            self.commands.append(msg)
        return self.succeed

    def Close(self):
        self.closed = True


@unittest.skipUnless(importlib.util.find_spec("unitree_sdk2py"), "Install the optional workstation SDK")
class G1DdsTests(unittest.TestCase):
    def setUp(self):
        self.patches = [patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": "1"}),
                        patch("unitree_sdk2py.core.channel.ChannelFactoryInitialize"),
                        patch("unitree_sdk2py.core.channel.ChannelSubscriber", Subscriber),
                        patch("unitree_sdk2py.core.channel.ChannelPublisher", Publisher)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.cfg = ConfigManager("g1_ufo_video_real_workstation").get_cfg().env
        self.cfg.unitree.net_if = "lo"
        self.cfg.connect_timeout_s = 0.05

    def environment(self):
        env = UnitreeG1DdsEnv(self.cfg)
        self.addCleanup(env.shutdown)
        return env

    def test_no_commands_during_initialization_update_reset_or_shutdown(self):
        env = self.environment()
        env.update()
        env.reset()
        self.assertFalse(env.send_hold())
        env.shutdown()
        env.shutdown()
        self.assertEqual(env._publisher.commands, [])

    def test_official_crc_and_29_motor_command_contract(self):
        from unitree_sdk2py.utils.crc import CRC

        env = self.environment()
        env.update()
        env.step(env.dof_pos)
        cmd = env._publisher.commands[-1]
        self.assertEqual(cmd.crc, CRC().Crc(cmd))
        self.assertEqual((cmd.mode_pr, cmd.mode_machine, len(cmd.motor_cmd)), (0, 5, 35))
        np.testing.assert_allclose([m.q for m in cmd.motor_cmd[:29]], env.dof_pos)
        for i, motor in enumerate(cmd.motor_cmd[:29]):
            self.assertEqual((motor.mode, motor.dq, motor.tau), (1, 0, 0))
            self.assertAlmostEqual(motor.kp, env.stiffness[i])
            self.assertAlmostEqual(motor.kd, env.damping[i])
        self.assertTrue(all(m.mode == 0 for m in cmd.motor_cmd[29:]))
        time.sleep(0.03)
        self.assertEqual(len(env._publisher.commands), 1, "No background resend is allowed")

    def test_motor_mapping_uses_names_for_both_feedback_and_commands(self):
        dof = self.cfg.dof
        for name in ("joint_names", "default_pos", "stiffness", "damping", "torque_limits", "position_limits"):
            setattr(dof, name, list(reversed(getattr(dof, name))))
        env = self.environment()
        env.update()
        np.testing.assert_allclose(env.dof_pos, np.arange(28, -1, -1) * 0.001)
        env.step(env.dof_pos)
        np.testing.assert_allclose([m.q for m in env._publisher.commands[-1].motor_cmd[:29]],
                                   np.arange(29) * 0.001)

    def test_bad_crc_duplicate_and_old_ticks_do_not_refresh_age(self):
        reader = G1StateReader("lo")
        self.addCleanup(reader.close)
        before = reader.latest()
        bad = state(3)
        bad.crc ^= 1
        for msg in (bad, state(2), state(1)):
            reader._receive(msg)
        self.assertIs(reader.latest(), before)
        reader.clock = lambda: before.received_at + 0.101
        with self.assertRaisesRegex(G1DdsError, "stale"):
            reader.latest()
        self.assertEqual(reader.metrics()["crc_errors"], 1)
        self.assertEqual(reader.metrics()["old_ticks"], 2)

    def test_tick_wrap_is_valid_and_callback_snapshot_is_owned(self):
        reader = G1StateReader("lo")
        self.addCleanup(reader.close)
        reader._snapshot = replace(reader.latest(), tick=0xFFFFFFFF)
        msg = state(0)
        reader._receive(msg)
        snapshot = reader.latest()
        msg.motor_state[0].q = 99
        self.assertEqual(snapshot.tick, 0)
        self.assertEqual(snapshot.q[0], 0)
        self.assertFalse(snapshot.q.flags.writeable)

    def test_rejects_incompatible_machines_and_invalid_feedback(self):
        from unitree_sdk2py.utils.crc import CRC

        for mode in (1, 4, 6, 9, 99):
            reader = G1StateReader("lo")
            reader._receive(state(3, mode))
            with self.assertRaisesRegex(G1DdsError, "29DOF"):
                reader.latest()
            reader.close()
        reader = G1StateReader("lo")
        self.addCleanup(reader.close)
        bad = state(3)
        bad.imu_state.quaternion = [0.0] * 4
        bad.crc = CRC().Crc(bad)
        reader._receive(bad)
        with self.assertRaisesRegex(G1DdsError, "quaternion"):
            reader.latest()
        reader._receive(state(4))
        with self.assertRaises(G1DdsError):
            reader.latest()

    def test_target_guard_clips_and_limits_increment(self):
        env = self.environment()
        env.update()
        env.step(np.full(29, 1000.0))
        q = np.array([m.q for m in env._publisher.commands[-1].motor_cmd[:29]])
        self.assertTrue(np.all(q <= env.position_limits[:, 1]))
        self.assertTrue(np.all(q >= env.position_limits[:, 0]))
        self.assertTrue(np.all(np.abs(q - env.dof_pos) <= env.target_guard.max_delta + 1e-6))

    def test_raw_actor_output_is_scaled_and_offset_before_dds(self):
        from types import SimpleNamespace

        from robojudo.environment.utils.real_target_guard import RealTargetGuard
        from robojudo.pipeline.rl_pipeline import PolicyWrapper

        cfg = ConfigManager("g1_ufo_video_real_workstation").get_cfg()
        env = self.environment()
        wrapper = PolicyWrapper(cfg.policy, env.dof_cfg, "cpu")
        raw = np.linspace(-2, 2, 29, dtype=np.float32)
        wrapper.policy.actor_session = SimpleNamespace(run=lambda *args: [raw[None]])
        target = wrapper.get_pd_target(np.zeros(721, np.float32))
        normalized = np.clip(raw * 5, -5, 5)
        expected = np.asarray(cfg.env.dof.default_pos) + normalized * (
            0.25 * np.asarray(cfg.policy.effort_limits) / np.asarray(cfg.env.dof.stiffness))
        np.testing.assert_allclose(target, expected, rtol=1e-6)
        np.testing.assert_allclose(wrapper.policy.last_action, normalized)
        # Model loading takes longer than the feedback watchdog. The transport
        # fake must provide another packet, as the live subscriber would.
        env._reader._receive(state(3))
        env.update()
        guard = RealTargetGuard(env.position_limits, env.cfg_env.target_velocity_limits,
                                env.cfg_env.unitree.control_dt, env.cfg_env.target_slew_safety_factor)
        guarded = guard.filter(expected, env.dof_pos)
        env.step(target)
        np.testing.assert_allclose([m.q for m in env._publisher.commands[-1].motor_cmd[:29]],
                                   guarded, atol=1e-6)

    def test_expired_cycle_is_not_sent_and_fault_is_latched(self):
        env = self.environment()
        env.update()
        env._cycle_started -= 0.101
        with self.assertRaisesRegex(G1DdsError, "deadline"):
            env.step(env.dof_pos)
        env.update()
        with self.assertRaisesRegex(G1DdsError, "latched"):
            env.step(env.dof_pos)
        self.assertEqual(env._publisher.commands, [])

    def test_stale_feedback_never_sends_target_or_hold(self):
        env = self.environment()
        env.update()
        env.step(env.dof_pos)
        before = len(env._publisher.commands)
        snapshot = env._reader.latest()
        env._reader.clock = lambda: snapshot.received_at + 0.101
        with self.assertRaisesRegex(G1DdsError, "stale"):
            env.step(env.dof_pos)
        self.assertFalse(env.send_hold())
        self.assertEqual(len(env._publisher.commands), before)

    def test_write_failure_latches_fault_but_fresh_hold_is_available(self):
        env = self.environment()
        env.update()
        env.step(env.dof_pos)
        env._publisher.succeed = False
        with self.assertRaisesRegex(G1DdsError, "write failed"):
            env.step(env.dof_pos)
        env._publisher.succeed = True
        env._reader._receive(state(3))
        env.update()
        with self.assertRaisesRegex(G1DdsError, "latched"):
            env.step(env.dof_pos)
        self.assertTrue(env.send_hold())
        self.assertEqual(env.metrics()["send_failures"], 1)

    def test_missing_explicit_interface_and_robot_gate_fail_closed(self):
        with self.assertRaises(ValueError):
            G1StateReader("")
        with patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": ""}):
            with self.assertRaisesRegex(RuntimeError, "UFO_REAL_ROBOT_OK"):
                UnitreeG1DdsEnv(self.cfg)


if __name__ == "__main__":
    unittest.main()
