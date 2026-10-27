"""Hardware-free contract checks for UFO M0 Real control."""

import importlib
import struct
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from box import Box

from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg
from robojudo.controller.unitree_ctrl import UnitreeCtrl
from robojudo.config.config_manager import ConfigManager
import robojudo.environment
from robojudo.environment.base_env import Environment
from robojudo.environment.utils.real_target_guard import RealTargetGuard
from robojudo.pipeline.ufo_m0_pipeline import UfoM0Pipeline
from robojudo.policy.ufo_m0_policy import HISTORY_KEYS, UfoM0Policy


class FakePolicy:
    def __init__(self):
        self.policy = self
        self.motion_done = False
        self.frame = 0
        self.reset_count = 0
        self.tracking_enabled = False

    def reset(self):
        self.frame = 0
        self.motion_done = False
        self.reset_count += 1

    def reset_reference(self):
        self.frame = 0
        self.motion_done = False

    def set_tracking_enabled(self, enabled):
        self.tracking_enabled = enabled


class FakeEnv:
    def __init__(self):
        self.dof_pos = np.zeros(29)
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.dof_cfg = SimpleNamespace(default_pos=np.full(29, 0.2))
        self.target_guard = SimpleNamespace(last_command=None)
        self.targets = []
        self.holds = 0

    def step(self, target):
        target = np.asarray(target).copy()
        self.targets.append(target)
        self.target_guard.last_command = target

    def send_hold(self):
        self.holds += 1
        return True


def fake_pipeline():
    pipeline = UfoM0Pipeline.__new__(UfoM0Pipeline)
    pipeline._real_mode = True
    pipeline.real_state = "DISARMED"
    pipeline._fault_stop = False
    pipeline._released_after_stop = False
    pipeline._has_actuated = False
    pipeline._standing_ready = False
    pipeline._previous_buttons = {name: False for name in pipeline.BUTTONS}
    pipeline._feedback_valid_cycle = False
    pipeline._init_step = 0
    pipeline._init_start = None
    pipeline._return_step = 0
    pipeline._return_start = None
    pipeline.cfg = SimpleNamespace(do_safety_check=True)
    pipeline.freq = 50
    pipeline.timestep = 0
    pipeline.env = FakeEnv()
    pipeline.policy = FakePolicy()
    pipeline.ctrl_manager = SimpleNamespace(post_step_callback=lambda _: None)
    return pipeline


def control_cycle(pipeline, **buttons):
    state = {name: bool(buttons.get(name, False)) for name in pipeline.BUTTONS}
    ctrl_data = Box({"UnitreeCtrl": {"buttons": state}, "COMMANDS": []})
    env_data = Box({
        "dof_pos": pipeline.env.dof_pos,
        "dof_vel": np.zeros(29),
        "base_quat": pipeline.env.base_quat,
        "base_ang_vel": np.zeros(3),
    })
    return pipeline.pre_policy_step(env_data, ctrl_data)


class RealTargetGuardTests(unittest.TestCase):
    def test_limits_slew_and_hold(self):
        guard = RealTargetGuard([[-1, 1], [-1, 1]], [1, 2], 0.02, 0.5)
        np.testing.assert_allclose(guard.filter([4, -4], [0, 0]), [0.01, -0.02])
        np.testing.assert_allclose(guard.filter([4, -4], [0, 0]), [0.02, -0.04])
        np.testing.assert_allclose(guard.hold([0.8, -0.8]), [0.8, -0.8])
        np.testing.assert_allclose(guard.filter([4, -4], [0, 0]), [0.81, -0.82])
        guard.hold([0.995, -0.995])
        np.testing.assert_allclose(guard.filter([4, -4], [0, 0]), [1.0, -1.0])

    def test_rejects_invalid_targets_and_first_feedback(self):
        guard = RealTargetGuard([[-1, 1], [-1, 1]], [1, 2], 0.02, 0.5)
        for bad in ([0], [np.nan, 0], [0, np.inf]):
            with self.assertRaises(ValueError):
                guard.filter(bad, [0, 0])
        with self.assertRaises(ValueError):
            guard.filter([0, 0], [np.nan, 0])
        guard.hold([0, 0])
        with self.assertRaises(ValueError):
            guard.filter([0, 0], [np.nan, 0])
        with self.assertRaises(ValueError):
            guard.hold([np.inf, 0])

    def test_optional_guard_is_used_at_cpp_send_boundary(self):
        # The local machine has no real unitree_cpp extension. Substitute only
        # its import surface; never construct the actual hardware controller.
        fake_extension = types.ModuleType("unitree_cpp")
        fake_extension.RobotState = object
        fake_extension.SportState = object
        fake_extension.UnitreeController = object
        with patch.dict(sys.modules, {"unitree_cpp": fake_extension}):
            module = importlib.import_module("robojudo.environment.unitree_cpp_env")

        class Sender:
            def __init__(self):
                self.commands = []

            def step(self, target):
                self.commands.append(target)

        env = module.UnitreeCppEnv.__new__(module.UnitreeCppEnv)
        env.enabled = True
        env.robot_state = object()
        env.num_dofs = 2
        env._dof_pos = np.zeros(2)
        env.target_guard = RealTargetGuard([[-1, 1], [-1, 1]], [1, 2], 0.02, 0.5)
        env.unitree = Sender()
        env.step([5, -5])
        np.testing.assert_allclose(env.unitree.commands[-1], [0.01, -0.02])
        with self.assertRaises(ValueError):
            env.step([np.nan, 0])
        self.assertEqual(len(env.unitree.commands), 1)
        env._dof_pos = np.array([0.4, -0.4])
        self.assertTrue(env.send_hold())
        np.testing.assert_allclose(env.unitree.commands[-1], [0.4, -0.4])


class UfoM0HistoryTests(unittest.TestCase):
    def test_four_prior_frames_are_newest_first(self):
        policy = UfoM0Policy.__new__(UfoM0Policy)
        policy.default_dof_pos = np.zeros(29)
        policy.motion = SimpleNamespace(num_frames=3, qpos_at=lambda _: np.zeros(36))
        policy.z = np.zeros((2, 256), dtype=np.float32)
        policy.reset()
        self.assertEqual(tuple(policy.history), HISTORY_KEYS)
        for value in range(1, 7):
            env = SimpleNamespace(
                dof_pos=np.full(29, value, dtype=np.float32),
                dof_vel=np.full(29, value, dtype=np.float32),
                base_quat=np.array([0, 0, 0, 1], dtype=np.float32),
                base_ang_vel=np.full(3, value, dtype=np.float32),
            )
            obs, _ = policy.get_observation(env, {})
            self.assertEqual(obs.shape, (721,))
        # State 64 + last action 29 + actions history 116 + angular history 12.
        pos_history = obs[64 + 29 + 116 + 12:][:116].reshape(4, 29)
        np.testing.assert_allclose(pos_history[:, 0], [5, 4, 3, 2])
        self.assertEqual(len(policy.history["dof_pos"]), 4)


class UfoM0StateTests(unittest.TestCase):
    def test_real_gate_precedes_environment_construction(self):
        cfg = ConfigManager("g1_ufo_m0_real").get_cfg()
        with patch.dict("os.environ", {"UFO_REAL_ROBOT_OK": ""}):
            with self.assertRaisesRegex(RuntimeError, "UFO_REAL_ROBOT_OK=1"):
                UfoM0Pipeline(cfg)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "assets/models/g1/ufo_m0/FBcprAuxModel.onnx").exists()
        and (Path.home() / "data/lafan/lafan_g1_npz/walk1_subject1.npz").exists(),
        "local M0 model and motion are needed for the integration check",
    )
    def test_real_config_with_actual_onnx_and_fake_environment(self):
        class SimulatedDriver(Environment):
            def __init__(self, cfg_env, device="cpu"):
                super().__init__(cfg_env, device)
                self.RemoteControllerHandler = None
                self.commands = []
                self.holds = 0
                self.closed = False
                self.target_guard = RealTargetGuard(
                    self.position_limits, cfg_env.target_velocity_limits,
                    cfg_env.unitree.control_dt, cfg_env.target_slew_safety_factor,
                )

            def self_check(self):
                pass

            def reset(self):
                pass

            def update(self):
                pass

            def step(self, target, hand_pose=None):
                command = self.target_guard.filter(target, self.dof_pos)
                self.commands.append(command)
                self._dof_pos = command.copy()

            def send_hold(self):
                self.holds += 1
                self.commands.append(self.target_guard.hold(self.dof_pos))
                return True

            def shutdown(self):
                self.closed = True

            def set_gains(self, stiffness, damping):
                pass

        cfg = ConfigManager("g1_ufo_m0_real").get_cfg()
        with patch.dict("os.environ", {"UFO_REAL_ROBOT_OK": "1"}), patch.dict(
            robojudo.environment.__dict__, {"UnitreeCppEnv": SimulatedDriver}
        ):
            pipeline = UfoM0Pipeline(cfg)
        try:
            controller = pipeline.ctrl_manager.controllers["UnitreeCtrl"].inst
            button_state = controller.unitree_remote_controller.last_button_state
            pipeline.step()
            self.assertEqual(pipeline.env.commands, [])
            button_state[8] = True  # A
            pipeline.step()
            button_state[8] = False
            for _ in range(499):
                pipeline.step()
            self.assertEqual(pipeline.real_state, "STANDING")
            button_state[0] = True  # R1
            pipeline.step()
            self.assertEqual(pipeline.real_state, "ENABLED")
            button_state[0] = False
            button_state[9] = True  # B
            pipeline.step()
            self.assertEqual(pipeline.real_state, "TRACKING")
            button_state[9] = False
            pipeline.step()
            self.assertGreater(pipeline.policy.policy.frame, 0)
            button_state[4] = True  # R2
            pipeline.step()
            self.assertEqual(pipeline.real_state, "STOPPED")
            self.assertGreater(pipeline.env.holds, 0)
        finally:
            pipeline.env.shutdown()
            self.assertTrue(pipeline.env.closed)

    def test_common_rl_loop_skips_inference_while_disarmed_or_stopped(self):
        pipeline = fake_pipeline()
        pipeline.env.update = lambda: None
        pipeline.env.get_data = lambda: Box({
            "dof_pos": pipeline.env.dof_pos,
            "dof_vel": np.zeros(29),
            "base_quat": pipeline.env.base_quat,
            "base_ang_vel": np.zeros(3),
        })
        buttons = {name: False for name in pipeline.BUTTONS}
        pipeline.ctrl_manager.get_ctrl_data = lambda _: Box({
            "UnitreeCtrl": {"buttons": buttons}, "COMMANDS": [],
        })
        pipeline.policy.get_observation = lambda *_: self.fail("inference reached while disarmed")
        pipeline.step()
        self.assertEqual(pipeline.env.targets, [])
        buttons["R2"] = True
        pipeline.step()
        self.assertEqual(pipeline.real_state, "STOPPED")
        self.assertEqual(pipeline.env.targets, [])
        self.assertEqual(pipeline.env.holds, 0)
        buttons["R2"] = False
        pipeline.step()
        buttons["R1"] = True
        buttons["B"] = True
        pipeline.step()
        self.assertEqual(pipeline.real_state, "STOPPED")
        self.assertEqual(pipeline.env.targets, [])
        buttons["R1"] = False
        buttons["B"] = False
        buttons["A"] = True
        pipeline.step()
        self.assertEqual(pipeline.real_state, "INITIALIZING")
        self.assertEqual(len(pipeline.env.targets), 1)

    def test_remote_sequence_and_stop_latch(self):
        pipeline = fake_pipeline()
        self.assertTrue(control_cycle(pipeline))
        self.assertEqual(pipeline.env.targets, [])
        self.assertTrue(control_cycle(pipeline, A=True))
        for _ in range(499):
            self.assertTrue(control_cycle(pipeline))
        self.assertEqual(pipeline.real_state, "STANDING")
        self.assertEqual(len(pipeline.env.targets), 500)
        self.assertFalse(control_cycle(pipeline, R1=True))
        self.assertEqual(pipeline.real_state, "ENABLED")
        self.assertFalse(pipeline.policy.tracking_enabled)
        self.assertFalse(control_cycle(pipeline, B=True))
        self.assertEqual(pipeline.real_state, "TRACKING")
        self.assertTrue(pipeline.policy.tracking_enabled)
        self.assertFalse(control_cycle(pipeline, X=True))
        self.assertEqual(pipeline.real_state, "ENABLED")
        self.assertFalse(control_cycle(pipeline, B=True))
        self.assertTrue(control_cycle(pipeline, R2=True, A=True))
        self.assertEqual(pipeline.real_state, "STOPPED")
        self.assertGreater(pipeline.env.holds, 0)
        self.assertTrue(control_cycle(pipeline, R1=True, B=True))
        self.assertEqual(pipeline.real_state, "STOPPED")
        self.assertTrue(control_cycle(pipeline))
        self.assertFalse(control_cycle(pipeline, R1=True, B=True))
        self.assertEqual(pipeline.real_state, "TRACKING")
        self.assertEqual(pipeline.policy.frame, 0)

    def test_fault_cannot_rearm_and_motion_returns_to_stand(self):
        pipeline = fake_pipeline()
        pipeline.real_state = "TRACKING"
        pipeline._enter_stop("bad observation", fault=True)
        control_cycle(pipeline)
        self.assertTrue(control_cycle(pipeline, R1=True, B=True))
        self.assertEqual(pipeline.real_state, "STOPPED")

        pipeline = fake_pipeline()
        pipeline.real_state = "TRACKING"
        pipeline.env.target_guard.last_command = np.full(29, 0.3)
        pipeline._begin_return()
        self.assertEqual(pipeline.real_state, "RETURNING")
        for _ in range(250):
            self.assertTrue(control_cycle(pipeline))
        self.assertEqual(pipeline.real_state, "STANDING")
        np.testing.assert_allclose(pipeline.env.targets[-1], np.full(29, 0.2))

    def test_invalid_quaternion_and_dry_run_never_send_target(self):
        pipeline = fake_pipeline()
        pipeline.env.base_quat = np.array([0.0, 0.0, 0.0, 2.0])
        pipeline._dry_run_cycle = True
        pipeline.safety_check()
        self.assertEqual(pipeline.real_state, "DISARMED")
        pipeline._dry_run_cycle = False
        self.assertTrue(control_cycle(pipeline))
        self.assertEqual(pipeline.real_state, "STOPPED")
        self.assertTrue(pipeline._fault_stop)
        self.assertEqual(pipeline.env.targets, [])
        self.assertEqual(pipeline.env.holds, 0)


class UnitreeButtonSnapshotTests(unittest.TestCase):
    def test_snapshot_tracks_press_and_release(self):
        ctrl = UnitreeCtrl(
            cfg_ctrl=UnitreeCtrlCfg(triggers={}, combination_init_buttons=[]),
            env=SimpleNamespace(),
        )
        remote = bytearray(24)
        struct.pack_into("<H", remote, 2, (1 << 0) | (1 << 4) | (1 << 9))
        ctrl.unitree_remote_controller.parse(remote)
        buttons = ctrl.get_data()["buttons"]
        self.assertTrue(buttons["R1"])
        self.assertTrue(buttons["R2"])
        self.assertTrue(buttons["B"])
        ctrl.unitree_remote_controller.parse(bytearray(24))
        self.assertFalse(ctrl.get_data()["buttons"]["R2"])


if __name__ == "__main__":
    unittest.main()
