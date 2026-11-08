"""Causal cadence, retained history and cache integrity of the accelerated stream."""

import importlib.util
import json
import sys
import unittest
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from robojudo.video.vision_accel import sha256, validated_cache
from robojudo.video.vision_fast import FastVisionWorker
from robojudo.video.vision_source import Frame
from robojudo.video.vision_worker import Sample


def bare_worker(window=50):
    worker = object.__new__(FastVisionWorker)
    worker.WINDOW = window
    worker.variant = "tcn"
    worker.device = "cpu"
    worker.resets = worker.gap_samples = worker.samples_skipped = 0
    worker.result_count = worker.frame_count = 0
    worker.first_capture_time = None
    worker.feature_seconds = deque(maxlen=100)
    worker._head_events = deque(maxlen=100)
    worker.person_loss_reported = False
    worker.reset_stream()
    worker._window_results = lambda count: [(worker.samples[-1], np.ones(256), False)]
    return worker


def frame(index, source_time=None):
    return Frame(index, None, 1000. + index / 30, index / 30 if source_time is None else source_time)


class FastTimelineTests(unittest.TestCase):
    def test_outputs_each_new_image_without_waiting_for_context_stride(self):
        worker = bare_worker(window=4)
        outputs = []
        context_at_output = []

        def output(count):
            outputs.append((worker.samples[-1].index, count))
            context_at_output.append([value.clone() for value in worker.context])
            return []

        worker._window_results = output
        for index in range(7):
            worker._consume(frame(index), torch.full((1024,), float(index)))
        self.assertEqual(outputs, [(3, 4), (4, 1), (5, 1), (6, 1)])
        self.assertEqual(context_at_output[0], [])  # current window is never its own past context
        torch.testing.assert_close(context_at_output[1][0], torch.full((1024,), 1.5))
        self.assertEqual(len(worker.context), 1)  # context cadence is independent of output cadence

    def test_held_gap_features_keep_original_capture_time_and_process_all_new_tokens(self):
        worker = bare_worker(window=4)
        for index in range(4):
            worker._consume(frame(index), torch.full((1024,), float(index)))
        counts = []
        worker._window_results = lambda count: counts.append(count) or []
        worker._consume(frame(6), torch.full((1024,), 6.))
        self.assertEqual(counts, [3])
        self.assertEqual(worker.gap_samples, 2)
        self.assertEqual([sample.source_frame_id for sample in worker.samples], [3, 3, 3, 6])
        self.assertEqual([sample.captured_at for sample in worker.samples][:-1], [frame(3).captured_at] * 3)
        self.assertEqual(worker.samples_skipped, 0)
        before = list(worker.samples)
        self.assertEqual(worker._consume(frame(7, 5.999 / 30), torch.zeros(1024)), [])
        self.assertEqual([s.index for s in worker.samples], [s.index for s in before])

    def test_large_gap_restarts_warmup_and_history(self):
        worker = bare_worker(window=4)
        for index in range(4):
            worker._consume(frame(index), torch.ones(1024))
        worker.tcn_history.append(torch.ones(1024))
        session = worker.session_id
        self.assertEqual(worker._consume(frame(100), torch.ones(1024)), [])
        self.assertNotEqual(worker.session_id, session)
        self.assertEqual(len(worker.samples), 1)
        self.assertFalse(worker.tcn_history)
        self.assertFalse(worker.context)
        self.assertEqual(worker.resets, 1)

    def test_person_loss_clears_gpu_state_and_recovery_requires_fresh_window(self):
        worker = bare_worker(window=4)
        worker._feature = lambda value, **kwargs: None if value.index in (4, 5) else torch.ones(1024)
        batches = list(worker.process([frame(i) for i in range(10)], finalize=False))
        self.assertEqual([batch[-1][0].index for batch in batches], [3, 4, 9])
        self.assertIsNone(batches[1][0][1])
        self.assertEqual(worker.resets, 1)
        self.assertEqual([value.index for value in worker.samples], [6, 7, 8, 9])

    def test_empty_detection_never_runs_crop_or_reuses_box(self):
        worker = bare_worker()
        worker.last_box = np.ones(4)
        worker._processed_start = None
        worker.detector_seconds = deque()
        worker._feature_events = deque()
        worker.detector = SimpleNamespace(predict=Mock(return_value=[
            SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.empty(0, 4)))]))
        crops = Mock(side_effect=AssertionError("stale box reached HMR2"))
        modules = {"lib.util.hmr2_preprocess": SimpleNamespace(
            get_bbx_xys_from_xyxy=crops, prepare_crops=crops),
            "cv2": SimpleNamespace(COLOR_RGB2BGR=0, cvtColor=lambda image, _: image)}
        with patch.dict(sys.modules, modules):
            value = Frame(0, np.zeros((4, 4, 3), np.uint8), 1000., 0.)
            self.assertIsNone(worker._feature(value, allow_missing=True))
        self.assertIsNone(worker.last_box)
        self.assertIsNone(worker.detected_box)
        crops.assert_not_called()

    def test_tcam_never_populates_future_context_slots(self):
        worker = bare_worker(window=50)
        worker._start_event = lambda: None
        worker._temporal_events = deque()
        for index in range(50):
            worker.samples.append(Sample(index, index, index / 30, 1000., torch.zeros(1024)))
        worker.context.extend(torch.ones(1024) for _ in range(3))

        def frontend(x, context, mask):
            self.assertEqual(tuple(x.shape), (1, 50, 1024))
            self.assertEqual(mask.tolist(), [[True] * 3 + [False] * 13])
            self.assertEqual(torch.count_nonzero(context[:, 3:]).item(), 0)
            return torch.zeros(50, 1024)

        worker._temporal_forward = frontend
        worker._window_u()
        self.assertEqual(len(worker.context), 3)


class IncrementalHeadTests(unittest.TestCase):
    def test_mlp_emits_only_latest_sample_and_matches_pointwise_reference(self):
        path = Path(__file__).resolve().parents[1] / ".local/rgb_bfm/mlp_pipeline/code/uz_head.py"
        if not path.exists():
            self.skipTest("Local MLP package unavailable")
        spec = importlib.util.spec_from_file_location("test_fast_mlp", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(19)
        values = torch.randn(50, 1024)
        worker = bare_worker()
        worker.variant = "mlp"
        worker.head = module.UZHead(kind="mlp", width=32, dropout=0).eval()
        worker._start_event = lambda: None
        worker._window_u = lambda: (values, None)
        worker.samples.append(Sample(49, 49, 49 / 30, 1001., values[-1]))
        with torch.inference_mode():
            expected = worker.head(values[None])[0, -1].numpy()
        for count in (1, 2, 50):
            result = FastVisionWorker._window_results(worker, count=count)
            self.assertEqual(len(result), 1)
            np.testing.assert_allclose(result[0][1], expected, atol=3e-6)
        self.assertFalse(worker.tcn_history)

    def test_invalid_z_is_rejected_before_packet_production(self):
        worker = bare_worker()
        worker.samples.append(Sample(0, 0, 0., 1000., torch.ones(1024)))
        worker._window_u = lambda: (torch.ones(50, 1024), None)
        worker._start_event = lambda: None
        for value in (0., float("nan"), 2.):
            with self.subTest(value=value):
                worker.head = lambda x, mask: torch.full((1, x.shape[1], 256), value)
                with self.assertRaisesRegex(ValueError, "latent"):
                    FastVisionWorker._window_results(worker, count=1)
        self.assertEqual(worker.result_count, 0)

    def test_tcn_matches_continuous_reference_across_variable_frame_gaps(self):
        path = Path(__file__).resolve().parents[1] / ".local/rgb_bfm/tcn_pipeline/code/uz_head.py"
        if not path.exists():
            self.skipTest("Local model package unavailable")
        spec = importlib.util.spec_from_file_location("test_fast_head", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        torch.manual_seed(17)
        values = torch.randn(85, 1024)
        head = module.UZHead(kind="causal_tcn", width=32, dilations=(1, 2, 4), dropout=0).eval()
        worker = bare_worker()
        worker.head = head
        worker._start_event = lambda: None
        with torch.inference_mode():
            expected = head(values[None], torch.ones(1, len(values), dtype=torch.bool))[0]
        previous = 0
        for end in (50, 51, 53, 59, 60, 79, 85):
            worker.samples = deque((Sample(i, i, i / 30, 1000., values[i])
                                    for i in range(end - 50, end)), maxlen=50)
            worker._window_u = lambda end=end: (values[end - 50:end], None)
            result = FastVisionWorker._window_results(worker, count=end - previous)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0][0].index, end - 1)
            np.testing.assert_allclose(result[0][1], expected[end - 1].numpy(), atol=3e-6)
            self.assertAlmostEqual(np.linalg.norm(result[0][1]), 16, places=4)
            self.assertEqual(len(worker.tcn_history), 14)
            previous = end


class CacheIntegrityTests(unittest.TestCase):
    def test_cache_rejects_changed_artifact_source_and_runtime(self):
        with TemporaryDirectory() as temporary, \
                patch("robojudo.video.vision_accel.fingerprint", return_value={"source": "expected"}), \
                patch("robojudo.video.vision_accel.gpu_identity", return_value={"gpu": "test"}):
            directory = Path(temporary)
            self.assertIsNone(validated_cache(directory, directory, "cuda:0"))
            artifact = directory / "tcam_plain.pt"
            artifact.write_bytes(b"test artifact")
            contextual = directory / "tcam_context.pt"
            contextual.write_bytes(b"test contextual artifact")
            manifest = dict(version=1, sources={"source": "expected"}, torch=torch.__version__,
                            gpu={"gpu": "test"}, artifacts={artifact.name: sha256(artifact),
                                                         contextual.name: sha256(contextual)})
            def write():
                (directory / "manifest.json").write_text(json.dumps(manifest))
            write()
            self.assertEqual(validated_cache(directory, directory, "cuda:0"), manifest)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact"):
                validated_cache(directory, directory, "cuda:0")
            manifest["sources"] = {}
            write()
            with self.assertRaisesRegex(RuntimeError, "source weights"):
                validated_cache(directory, directory, "cuda:0")
            manifest["sources"] = {"source": "expected"}
            manifest["torch"] = "other"
            write()
            with self.assertRaisesRegex(RuntimeError, "Torch/GPU"):
                validated_cache(directory, directory, "cuda:0")


if __name__ == "__main__":
    unittest.main()
