"""Camera CLI startup must initialize HighGUI before PyAV/model imports."""

import importlib.util
import os
import sys
import unittest
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_ufo_video_source.py"


class CameraStartupTests(unittest.TestCase):
    def run_cli(self, worker_class, preview_class, profile="legacy", fast_class=None, extra=()):
        spec = importlib.util.spec_from_file_location("ufo_video_source_cli", SCRIPT)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with TemporaryDirectory() as directory, chdir(directory), \
                patch.object(sys, "path", sys.path.copy()), \
                patch.object(sys, "argv", [str(SCRIPT), "--camera", "0", "--variant", "tcn",
                                           "--vision-profile", profile, "--show-camera", "--show-detection-box", *extra]), \
                patch.dict(sys.modules, {
                    "vision_worker": SimpleNamespace(VisionWorker=worker_class),
                    "vision_fast": SimpleNamespace(FastVisionWorker=fast_class),
                    "vision_source": SimpleNamespace(CameraPreview=preview_class),
                }), patch.dict(os.environ):
            os.environ.pop("PYAV_LOGGING", None)
            cli.main()

    def test_preview_created_before_model_loading_and_reused(self):
        events = []
        preview = Mock()
        worker = Mock()
        worker.run.return_value = 4

        def create_preview(**kwargs):
            events.append("preview")
            self.assertTrue(kwargs["show_detection_box"])
            return preview

        def load_worker(*args):
            self.assertEqual(events, ["preview"])
            self.assertEqual(os.environ["PYAV_LOGGING"], "off")
            events.append("models")
            return worker

        self.run_cli(load_worker, create_preview)
        self.assertIs(worker.run.call_args.kwargs["preview"], preview)
        preview.close.assert_called_once_with()

    def test_preview_closed_when_model_loading_fails(self):
        preview = Mock()
        with self.assertRaisesRegex(RuntimeError, "model load failed"):
            self.run_cli(Mock(side_effect=RuntimeError("model load failed")), Mock(return_value=preview))
        preview.close.assert_called_once_with()

    def test_runtime_check_never_opens_source_or_transport(self):
        worker = Mock()
        preview = Mock()
        self.run_cli(Mock(return_value=worker), Mock(return_value=preview), extra=("--check-runtime",))
        worker.run.assert_not_called()
        preview.close.assert_called_once_with()

    def test_cuda_camera_auto_uses_fast_worker_after_preview_creation(self):
        legacy = Mock(side_effect=AssertionError("unexpected legacy worker"))
        preview = Mock()
        fast_worker = Mock()

        def create_fast(*args, **kwargs):
            preview_class.assert_called_once()
            self.assertEqual(kwargs["hmr_backend"], "auto")
            self.assertEqual(kwargs["cache_dir"], ".local/vision_accel")
            return fast_worker

        preview_class = Mock(return_value=preview)
        self.run_cli(legacy, preview_class, profile="auto", fast_class=create_fast)
        fast_worker.run.assert_called_once()
        preview.close.assert_called_once_with()

    def test_auto_preserves_file_original_and_cpu_paths(self):
        for arguments in (["--video", "example.mp4", "--variant", "tcn"],
                          ["--camera", "0", "--variant", "original"],
                          ["--camera", "0", "--variant", "tcn", "--device", "cpu"]):
            with self.subTest(arguments=arguments), TemporaryDirectory() as directory, chdir(directory):
                spec = importlib.util.spec_from_file_location("legacy_video_cli", SCRIPT)
                cli = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cli)
                legacy = Mock()
                fast = Mock(side_effect=AssertionError("unexpected accelerated worker"))
                with patch.object(sys, "argv", [str(SCRIPT), *arguments]), \
                        patch.object(sys, "path", sys.path.copy()), patch.dict(os.environ), \
                        patch.dict(sys.modules, {"vision_worker": SimpleNamespace(VisionWorker=legacy),
                                                 "vision_fast": SimpleNamespace(FastVisionWorker=fast)}):
                    cli.main()
                legacy.assert_called_once()
                fast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
