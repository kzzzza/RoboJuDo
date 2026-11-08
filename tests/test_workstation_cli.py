"""Workstation routing and preflight must never silently fall back to Jetson."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robojudo.config.config_manager import ConfigManager


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkstationCliTests(unittest.TestCase):
    def test_amo_config_and_launcher_are_opt_in(self):
        from tempfile import TemporaryDirectory

        cfg = ConfigManager("g1_ufo_video_real_workstation_amo").get_cfg()
        self.assertEqual(cfg.pipeline_type, "UfoAmoPipeline")
        self.assertEqual((cfg.init_seconds, cfg.switch_seconds), (3., .5))
        self.assertTrue(cfg.do_safety_check)
        cli = load_script("run_ufo_workstation_real")
        commands = []
        control = SimpleNamespace(pid=100, poll=lambda: None, returncode=0)
        vision = SimpleNamespace(pid=101, poll=lambda: 0, returncode=0)

        def spawn(command, **kwargs):
            commands.append(command)
            if len(commands) == 1:
                Path(command[command.index("--latent-socket") + 1]).touch()
                return control
            return vision

        with TemporaryDirectory() as directory, \
                patch.object(sys, "argv", ["runner", "--net-if", "lo", "--with-amo", "--no-preview", "--output", directory]), \
                patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": "1"}), \
                patch.object(cli, "checked") as checked, \
                patch.object(cli.subprocess, "Popen", side_effect=spawn), patch.object(cli, "stop"):
            cli.main()
        self.assertIn("g1_ufo_video_real_workstation_amo", commands[0])
        self.assertIn("g1_ufo_video_real_workstation_amo", checked.call_args_list[-1].args[0])

    def test_amo_vision_crash_keeps_control_until_operator_exit(self):
        from tempfile import TemporaryDirectory

        cli = load_script("run_ufo_workstation_real")
        commands = []
        control = SimpleNamespace(pid=100, poll=lambda: None, returncode=None)
        vision = SimpleNamespace(pid=101, poll=lambda: 9, returncode=9)

        def spawn(command, **kwargs):
            commands.append(command)
            if len(commands) == 1:
                Path(command[command.index("--latent-socket") + 1]).touch()
                return control
            return vision

        with TemporaryDirectory() as directory, \
                patch.object(sys, "argv", ["runner", "--net-if", "lo", "--with-amo", "--no-preview", "--output", directory]), \
                patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": "1"}), \
                patch.object(cli, "checked"), patch.object(cli.subprocess, "Popen", side_effect=spawn), \
                patch.object(cli, "stop") as stop, \
                patch.object(cli.time, "sleep", side_effect=KeyboardInterrupt) as waiting:
            cli.main()
            waiting.assert_called_once_with(.1)
            stop.assert_any_call(control)
            summaries = list(Path(directory).glob("*/summary.json"))
            import json
            summary = json.loads(summaries[0].read_text())
            self.assertTrue(summary["vision_failed"])
            self.assertEqual(summary["status"], "interrupted")

    def test_new_config_is_local_and_does_not_mutate_jetson_defaults(self):
        local = ConfigManager("g1_ufo_video_real_workstation").get_cfg()
        local.env.unitree.net_if = "wlan0"
        old = ConfigManager("g1_ufo_video_real_stream").get_cfg()
        self.assertEqual(local.ctrl[-1].source, "socket")
        self.assertEqual(local.env.env_type, "UnitreeG1DdsEnv")
        self.assertEqual(local.env.odometry_type, "NONE")
        self.assertFalse(local.env.unitree.enable_odometry)
        self.assertEqual(old.ctrl[-1].source, "tcp")
        self.assertEqual(old.env.unitree.net_if, "eth0")
        self.assertEqual(old.env.env_type, "UnitreeCppEnv")

    def test_robot_gate_and_interface_checked_before_starting_processes(self):
        cli = load_script("run_ufo_workstation_real")
        with patch.object(cli.subprocess, "Popen") as spawn:
            with patch.object(sys, "argv", ["runner", "--net-if", "lo", "--no-preview"]), \
                    patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": ""}):
                with self.assertRaises(SystemExit):
                    cli.main()
            with patch.object(sys, "argv", ["runner"]):
                with self.assertRaises(SystemExit):
                    cli.main()
            spawn.assert_not_called()

    def test_launcher_routes_two_local_processes_without_ssh(self):
        from tempfile import TemporaryDirectory

        cli = load_script("run_ufo_workstation_real")
        control = SimpleNamespace(pid=100, poll=lambda: None, returncode=0)
        vision = SimpleNamespace(pid=101, poll=lambda: 0, returncode=0)
        commands = []

        def spawn(command, **kwargs):
            commands.append(command)
            if len(commands) == 1:
                Path(command[command.index("--latent-socket") + 1]).touch()
                return control
            return vision

        with TemporaryDirectory() as directory, \
                patch.object(sys, "argv", ["runner", "--net-if", "wlan0", "--no-preview", "--output", directory]), \
                patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": "1"}), \
                patch.object(cli, "checked") as checked, \
                patch.object(cli.subprocess, "Popen", side_effect=spawn), patch.object(cli, "stop"):
            cli.main()
        self.assertEqual(len(commands), 2)
        self.assertIn("g1_ufo_video_real_workstation", commands[0])
        self.assertEqual(commands[0][commands[0].index("--net-if") + 1], "wlan0")
        self.assertIn("--read-only", checked.call_args_list[-1].args[0])
        self.assertEqual(commands[0][commands[0].index("--latent-socket") + 1],
                         commands[1][commands[1].index("--socket") + 1])
        self.assertEqual(commands[1][commands[1].index("--variant") + 1], "tcn")
        self.assertTrue(all("--ssh-target" not in command for command in commands))

    def test_preflight_failure_does_not_start_control(self):
        from tempfile import TemporaryDirectory

        cli = load_script("run_ufo_workstation_real")
        with TemporaryDirectory() as directory, \
                patch.object(sys, "argv", ["runner", "--net-if", "lo", "--no-preview", "--output", directory]), \
                patch.dict(os.environ, {"UFO_REAL_ROBOT_OK": "1"}), \
                patch.object(cli, "checked", side_effect=RuntimeError("model corrupt")), \
                patch.object(cli.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "model corrupt"):
                cli.main()
            spawn.assert_not_called()

    def test_metrics_write_failure_still_stops_and_closes_control(self):
        import robojudo.pipeline

        runner = load_script("run_pipeline")
        cfg = ConfigManager("g1_ufo_video_real_workstation").get_cfg()
        cfg.env.unitree.net_if = "lo"
        pipeline = SimpleNamespace(
            timestep=0, _video_mode=True, video_metrics=[], freq=50,
            real_state="DISARMED", _fault_stop=False, stop_reason=None,
            prepare=Mock(), safe_stop=Mock(),
            ctrl_manager=SimpleNamespace(shutdown=Mock()),
            env=SimpleNamespace(metrics=Mock(return_value={}), shutdown=Mock()))
        with patch.object(sys, "argv", ["runner", "-c", "g1_ufo_video_real_workstation",
                                       "--max-steps", "0", "--metrics-json", "/tmp/workstation-test.json"]), \
                patch("robojudo.config.config_manager.ConfigManager") as manager, \
                patch.object(robojudo.pipeline, "UfoM0Pipeline", return_value=pipeline), \
                patch.object(Path, "write_text", side_effect=OSError("disk full")):
            manager.return_value.get_cfg.return_value = cfg
            with self.assertRaisesRegex(OSError, "disk full"):
                runner.main()
        pipeline.safe_stop.assert_called_once()
        pipeline.ctrl_manager.shutdown.assert_called_once()
        pipeline.env.shutdown.assert_called_once()

    def test_amo_exit_holds_before_metrics_even_when_disk_write_fails(self):
        import robojudo.pipeline

        runner = load_script("run_pipeline")
        cfg = ConfigManager("g1_ufo_video_real_workstation_amo").get_cfg()
        cfg.env.unitree.net_if = "lo"
        events = []
        pipeline = SimpleNamespace(timestep=0, _video_mode=True, video_metrics=[], freq=50,
            real_state="STOPPED", _fault_stop=False, stop_reason="runner exit",
            prepare=Mock(), counts={"video_policy_steps": 0},
            metrics=lambda: events.append("metrics") or {},
            safe_stop=lambda: events.append("stop"),
            ctrl_manager=SimpleNamespace(shutdown=Mock()),
            env=SimpleNamespace(metrics=lambda: {}, shutdown=Mock()))
        with patch.object(sys, "argv", ["runner", "-c", "g1_ufo_video_real_workstation_amo",
                                       "--max-steps", "0", "--metrics-json", "/tmp/amo-test.json"]), \
                patch("robojudo.config.config_manager.ConfigManager") as manager, \
                patch.object(robojudo.pipeline, "UfoAmoPipeline", return_value=pipeline), \
                patch.object(Path, "write_text", side_effect=OSError("disk full")):
            manager.return_value.get_cfg.return_value = cfg
            with self.assertRaisesRegex(OSError, "disk full"):
                runner.main()
        self.assertEqual(events, ["stop", "metrics"])
        pipeline.ctrl_manager.shutdown.assert_called_once()
        pipeline.env.shutdown.assert_called_once()

    def test_read_only_preflight_never_creates_publisher_or_releases_mode(self):
        from test_g1_workstation_dds import Subscriber

        if importlib.util.find_spec("unitree_sdk2py") is None:
            self.skipTest("Install the workstation SDK")
        cli = load_script("check_ufo_m0_real")
        with patch.object(sys, "argv", ["check", "--config", "g1_ufo_video_real_workstation",
                                       "--net-if", "lo", "--latent-variant", "tcn", "--read-only"]), \
                patch.object(cli, "check_host"), \
                patch("unitree_sdk2py.core.channel.ChannelFactoryInitialize"), \
                patch("unitree_sdk2py.core.channel.ChannelSubscriber", Subscriber), \
                patch("unitree_sdk2py.core.channel.ChannelPublisher") as publisher, \
                patch("robojudo.environment.utils.g1_motion_mode.ensure_debug_mode") as release:
            cli.main()
            publisher.assert_not_called()
            release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
