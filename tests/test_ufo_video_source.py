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
    def run_cli(self, worker_class, preview_class):
        spec = importlib.util.spec_from_file_location("ufo_video_source_cli", SCRIPT)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with TemporaryDirectory() as directory, chdir(directory), \
                patch.object(sys, "path", sys.path.copy()), \
                patch.object(sys, "argv", [str(SCRIPT), "--camera", "0", "--variant", "tcn",
                                           "--show-camera", "--show-detection-box"]), \
                patch.dict(sys.modules, {
                    "vision_worker": SimpleNamespace(VisionWorker=worker_class),
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


if __name__ == "__main__":
    unittest.main()
