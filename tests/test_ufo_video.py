"""Hardware-free tests for the video latent boundary and safety state."""

import json
import socket
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from box import Box

from robojudo.controller.ctrl_cfgs import LatentCtrlCfg
from robojudo.controller.latent_ctrl import LatentCtrl
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.ufo_m0_pipeline import UfoM0Pipeline
from robojudo.video.latent_packet import decode_packet, encode_packet
from test_ufo_m0_real import fake_pipeline


class LatentCtrlTests(unittest.TestCase):
    def test_file_timeline_end_and_variant(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            z = np.tile(np.eye(1, 256, dtype=np.float32)[0] * 16, (3, 1))
            np.save(root / "z.npy", z)
            np.save(root / "timestamps.npy", [0.0, 0.04, 0.08])
            (root / "latent_metadata.json").write_text(json.dumps({"variant": "tcn"}))
            with self.assertRaises(ValueError):
                LatentCtrl(LatentCtrlCfg(source="file", z_dir=str(root), variant="mlp"))
            ctrl = LatentCtrl(LatentCtrlCfg(source="file", z_dir=str(root), variant="tcn", control_dt=0.02))
            ctrl.set_playing(True)
            for _ in range(5):
                ctrl.post_step_callback()
            self.assertEqual(ctrl.get_data()["frame_id"], 2)
            self.assertTrue(ctrl.get_data()["ended"])
            ctrl.reset()
            self.assertEqual(ctrl.get_data()["frame_id"], 0)
            self.assertFalse(ctrl.get_data()["ended"])

    def test_socket_latest_packet_and_staleness(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "latent.sock")
            ctrl = LatentCtrl(LatentCtrlCfg(source="socket", socket_path=path,
                                           variant="mlp", stale_seconds=0.02),
                              env=SimpleNamespace(cfg_env=SimpleNamespace(is_sim=True)))
            z = np.zeros(256, np.float32)
            z[0] = 16
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.sendto(encode_packet("tcn", 1, time.monotonic(), z), path)
                sender.sendto(encode_packet("mlp", 2, time.monotonic(), z), path)
                sender.sendto(encode_packet("mlp", 1, time.monotonic(), z), path)
            data = ctrl.get_data()
            self.assertEqual(data["frame_id"], 2)
            self.assertTrue(data["valid"])
            self.assertEqual(decode_packet(encode_packet("mlp", 2, time.monotonic(), z))[0], "mlp")
            time.sleep(0.03)
            self.assertFalse(ctrl.get_data()["valid"])
            ctrl.shutdown()
            self.assertFalse(Path(path).exists())


class RealVideoStateTests(unittest.TestCase):
    def test_real_video_requires_qualification_before_environment_creation(self):
        cfg = ConfigManager("g1_ufo_video_real_stream").get_cfg()
        with self.assertRaisesRegex(RuntimeError, "disabled until sim and latency"):
            UfoM0Pipeline(cfg)

    def test_stream_does_not_infer_before_b_or_after_stale_fault(self):
        pipeline = fake_pipeline()
        pipeline._video_mode = True
        latent = SimpleNamespace(cfg_ctrl=SimpleNamespace(source="socket"), ended=False,
                                 set_playing=lambda enabled: None, reset=lambda: None)
        pipeline.ctrl_manager.controllers = {"LatentCtrl": SimpleNamespace(inst=latent)}
        pipeline.policy.motion = None
        standing = len(pipeline.env.targets)

        def cycle(*, valid=True, **buttons):
            state = {name: bool(buttons.get(name, False)) for name in pipeline.BUTTONS}
            ctrl_data = Box({"UnitreeCtrl": {"buttons": state},
                             "LatentCtrl": {"valid": valid, "ended": False}, "COMMANDS": []})
            feedback = Box({"dof_pos": pipeline.env.dof_pos, "dof_vel": np.zeros(29),
                            "base_quat": pipeline.env.base_quat, "base_ang_vel": np.zeros(3)})
            return pipeline.pre_policy_step(feedback, ctrl_data)

        self.assertTrue(cycle())
        self.assertEqual(len(pipeline.env.targets), standing)
        self.assertTrue(cycle(A=True))
        for _ in range(pipeline.INIT_STEPS - 1):
            cycle()
        self.assertEqual(pipeline.real_state, "STANDING")
        cycle(R1=True)
        self.assertEqual(pipeline.real_state, "ENABLED")
        self.assertTrue(cycle())
        self.assertFalse(cycle(B=True))
        self.assertEqual(pipeline.real_state, "TRACKING")
        self.assertTrue(cycle(valid=False))
        self.assertEqual(pipeline.real_state, "STOPPED")
        self.assertTrue(pipeline._fault_stop)
        self.assertTrue(cycle(R1=True, B=True))
        self.assertEqual(pipeline.real_state, "STOPPED")


if __name__ == "__main__":
    unittest.main()
