"""Optional GPU regression for TensorRT output reallocation (no camera/DDS)."""

import importlib.util
import unittest
from pathlib import Path

import torch

from robojudo.video.vision_accel import trt_detector_backend


ENGINE = Path(__file__).resolve().parents[1] / ".local/vision_accel/yolov8x.engine"
AVAILABLE = importlib.util.find_spec("tensorrt") is not None and torch.cuda.is_available() and ENGINE.is_file()


@unittest.skipUnless(AVAILABLE, "Requires the project TensorRT cache and a CUDA device")
class DetectorCudaTests(unittest.TestCase):
    def test_output_stays_valid_after_repeated_shrink_and_growth(self):
        torch.cuda.set_device(0)
        detector = trt_detector_backend(ENGINE, "cuda:0")
        with torch.inference_mode():
            for height, width in ((256, 256), (640, 640), (384, 640), (640, 480), (640, 640)):
                image = torch.ones(1, 3, height, width, device="cuda", dtype=torch.float16)
                output = detector(image)
                anchors = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
                self.assertEqual(tuple(output.shape), (1, 84, anchors))
                self.assertTrue(torch.isfinite(output).all().item())
                self.assertTrue((output[:, 2:4] >= 0).all().item())
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
