"""Hardware-free checks for SSH latent transport and G1 video state."""

import os
import socket
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from box import Box

from robojudo.controller.ctrl_cfgs import LatentCtrlCfg
from robojudo.controller.latent_ctrl import LatentCtrl
from robojudo.environment.utils.g1_motion_mode import _release_with_sdk
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.pipeline.ufo_m0_pipeline import UfoM0Pipeline
from robojudo.video.latent_packet import encode_packet
from robojudo.video.latent_transport import LatestTcpSender
from test_ufo_m0_real import fake_pipeline


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    raise TimeoutError("Timed out waiting for latent transport")


class TcpLatentTests(unittest.TestCase):
    def test_host_can_start_before_receiver_and_deliver_latest(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserve:
            reserve.bind(("127.0.0.1", 0))
            port = reserve.getsockname()[1]
        sender = LatestTcpSender("127.0.0.1", port)
        try:
            z = np.zeros(256, np.float32)
            z[0] = 16
            sender.send(encode_packet("tcn", 1, time.monotonic() - 1, z))
            sender.send(encode_packet("tcn", 2, time.monotonic(), z))
            ctrl = LatentCtrl(LatentCtrlCfg(source="tcp", tcp_port=port, variant="tcn",
                                           stale_seconds=5.0, max_age_seconds=5.0))
            try:
                def valid_data():
                    data = ctrl.get_data()
                    return data if data["valid"] else None

                data = _wait_for(valid_data)
                self.assertEqual(data["frame_id"], 2)
                self.assertLess(data["age"], 1.0)
                sender.send(encode_packet("mlp", 3, time.monotonic(), z))
                _wait_for(lambda: ctrl.get_data()["frame_id"] == 3)
                self.assertFalse(ctrl.get_data()["valid"])
                sender.send(encode_packet("tcn", 4, time.monotonic() - 6, z))
                _wait_for(lambda: ctrl.get_data()["frame_id"] == 4)
                self.assertFalse(ctrl.get_data()["valid"])
                sender.send(encode_packet("tcn", 5, time.monotonic(), z))
                self.assertTrue(_wait_for(lambda: ctrl.get_data()["valid"]))
                sender.close()
                _wait_for(lambda: not ctrl.tcp_receiver.link_ready())
                self.assertFalse(ctrl.get_data()["valid"])
            finally:
                ctrl.shutdown()
        finally:
            sender.close()


class VideoRealStateTests(unittest.TestCase):
    def _cycle(self, pipeline, *, valid=True, **buttons):
        state = {name: bool(buttons.get(name, False)) for name in pipeline.BUTTONS}
        ctrl_data = Box({"UnitreeCtrl": {"buttons": state},
                         "LatentCtrl": {"valid": valid, "ended": False}, "COMMANDS": []})
        feedback = Box({"dof_pos": pipeline.env.dof_pos, "dof_vel": np.zeros(29),
                        "base_quat": pipeline.env.base_quat, "base_ang_vel": np.zeros(3)})
        return pipeline.pre_policy_step(feedback, ctrl_data)

    def test_a_b_r2_and_explicit_a_rearm(self):
        pipeline = fake_pipeline()
        pipeline._video_mode = True
        latent = SimpleNamespace(cfg_ctrl=SimpleNamespace(source="tcp"), ended=False,
                                 set_playing=lambda _: None, reset=lambda: None)
        pipeline.ctrl_manager.controllers = {"LatentCtrl": SimpleNamespace(inst=latent)}
        self.assertTrue(self._cycle(pipeline, B=True))
        self.assertEqual(pipeline.env.targets, [])
        self._cycle(pipeline, A=True)
        for _ in range(pipeline.INIT_STEPS - 1):
            self._cycle(pipeline)
        self.assertEqual(pipeline.real_state, "STANDING")
        self.assertTrue(self._cycle(pipeline, valid=False, B=True))
        self.assertEqual(pipeline.real_state, "STANDING")
        self._cycle(pipeline)
        self.assertFalse(self._cycle(pipeline, B=True))
        self.assertEqual(pipeline.real_state, "TRACKING")
        self.assertTrue(self._cycle(pipeline, R2=True, A=True, B=True))
        self.assertEqual(pipeline.real_state, "STOPPED")
        self.assertGreater(pipeline.env.holds, 0)
        self.assertTrue(self._cycle(pipeline, A=True))
        self.assertEqual(pipeline.real_state, "STOPPED")
        self._cycle(pipeline)
        self.assertTrue(self._cycle(pipeline, B=True))
        self.assertEqual(pipeline.real_state, "STOPPED")
        self._cycle(pipeline)
        self.assertTrue(self._cycle(pipeline, A=True))
        self.assertEqual(pipeline.real_state, "INITIALIZING")
        for _ in range(pipeline.INIT_STEPS - 1):
            self._cycle(pipeline)
        self.assertEqual(pipeline.real_state, "STANDING")

    def test_auto_debug_failure_precedes_driver_construction(self):
        cfg = SimpleNamespace(env=SimpleNamespace(is_sim=False,
                                                  unitree=SimpleNamespace(net_if="eth0")),
                              policy=SimpleNamespace(latent_source="controller"),
                              video_real_enabled=True, auto_debug_mode=True)
        with patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": "1"}), \
             patch("robojudo.environment.utils.g1_motion_mode.ensure_debug_mode",
                   side_effect=RuntimeError("mode switch failed")), \
             patch.object(RlPipeline, "__init__") as initialize:
            with self.assertRaisesRegex(RuntimeError, "mode switch failed"):
                UfoM0Pipeline(cfg)
            initialize.assert_not_called()

    def test_capture_age_is_rechecked_after_actor_inference(self):
        pipeline = fake_pipeline()
        pipeline._video_mode = True
        pipeline.real_state = "TRACKING"
        ctrl_data = {"LatentCtrl": {"capture_time": time.monotonic() - 6.0}}
        with self.assertRaisesRegex(ValueError, "exceeded 5 seconds"):
            pipeline.before_env_step(None, ctrl_data, {}, np.zeros(29))
        ctrl_data["LatentCtrl"]["capture_time"] = time.monotonic()
        pipeline.before_env_step(None, ctrl_data, {}, np.zeros(29))


class DebugModeTests(unittest.TestCase):
    def test_official_switcher_releases_active_mode_and_confirms(self):
        calls = []

        class FakeSwitcher:
            active = True

            def SetTimeout(self, seconds):
                calls.append(("timeout", seconds))

            def Init(self):
                calls.append(("init",))

            def CheckMode(self):
                calls.append(("check",))
                return 0, {"name": "ai" if self.active else ""}

            def ReleaseMode(self):
                calls.append(("release",))
                self.active = False
                return 0, None

        channel = types.ModuleType("unitree_sdk2py.core.channel")
        channel.ChannelFactoryInitialize = lambda domain, interface: calls.append(
            ("channel", domain, interface))
        switcher = types.ModuleType("unitree_sdk2py.comm.motion_switcher.motion_switcher_client")
        switcher.MotionSwitcherClient = FakeSwitcher
        modules = {
            "unitree_sdk2py": types.ModuleType("unitree_sdk2py"),
            "unitree_sdk2py.core": types.ModuleType("unitree_sdk2py.core"),
            "unitree_sdk2py.core.channel": channel,
            "unitree_sdk2py.comm": types.ModuleType("unitree_sdk2py.comm"),
            "unitree_sdk2py.comm.motion_switcher": types.ModuleType("unitree_sdk2py.comm.motion_switcher"),
            "unitree_sdk2py.comm.motion_switcher.motion_switcher_client": switcher,
        }
        with patch.dict(sys.modules, modules), patch(
            "robojudo.environment.utils.g1_motion_mode.time.sleep", return_value=None
        ):
            _release_with_sdk("eth0")
        self.assertEqual(calls[0], ("channel", 0, "eth0"))
        self.assertEqual([call[0] for call in calls].count("release"), 1)
        self.assertEqual([call[0] for call in calls].count("check"), 2)


if __name__ == "__main__":
    unittest.main()
