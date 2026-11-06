"""Streaming window, incremental head and bounded Original handoff checks."""

import importlib.util
import sys
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from robojudo.video.latent_packet import decode_packet, encode_packet
from robojudo.video.vision_source import Frame, LatestFrameSource
from robojudo.video.vision_worker import Sample, VisionWorker


ROOT = Path(__file__).resolve().parents[1]
HEAD_PATH = ROOT / ".local/rgb_bfm/tcn_pipeline/code/uz_head.py"
module = None
if HEAD_PATH.is_file():
    spec = importlib.util.spec_from_file_location("test_uz_head", HEAD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class StreamingTests(unittest.TestCase):
    def test_latest_source_discards_backlog(self):
        source = LatestFrameSource((Frame(i, None, time.monotonic(), i / 30) for i in range(100)))
        source.thread.join(timeout=2)
        self.assertEqual([frame.index for frame in source], [99])
        self.assertEqual(source.dropped, 99)
        source.close()

    def test_packet_preserves_session_and_timing(self):
        z = np.zeros(256, np.float32)
        z[0] = 16
        now = time.monotonic()
        item = decode_packet(encode_packet("tcn", 52, now - 0.5, z,
                                           session_id=123, source_frame_id=47,
                                           sample_time=52 / 30, produced_time=now))
        self.assertEqual((item.session_id, item.sequence, item.source_frame_id), (123, 52, 47))
        self.assertAlmostEqual(item.sample_time, 52 / 30)
        self.assertAlmostEqual(item.produced_time, now)

    def test_person_lost_packet_is_explicit_and_has_no_latent(self):
        now = time.monotonic()
        lost = decode_packet(encode_packet("tcn", 53, now, np.zeros(256, np.float32),
                                           person_lost=True))
        self.assertTrue(lost.person_lost)
        self.assertFalse(lost.eof)
        with self.assertRaises(ValueError):
            encode_packet("tcn", 53, now, np.ones(256, np.float32), person_lost=True)

    def test_live_person_loss_is_reported_once_and_resets_history(self):
        worker = object.__new__(VisionWorker)
        worker.frame_count = 0
        worker.first_capture_time = None
        worker.feature_seconds = deque(maxlen=10)
        worker.person_loss_reported = False
        worker.resets = 0
        worker.reset_stream()
        worker._feature = lambda frame, **_: None
        frames = [Frame(i, None, time.monotonic(), i / 30) for i in range(3)]
        batches = list(worker.process(frames, finalize=False))
        self.assertEqual(len(batches), 1)
        self.assertIsNone(batches[0][0][1])
        self.assertEqual(worker.resets, 1)
        self.assertEqual(len(worker.samples), 0)

    def test_live_missing_detection_never_reuses_the_old_box(self):
        worker = object.__new__(VisionWorker)
        worker.device = "cpu"
        worker.last_box = np.array([0, 0, 10, 10], np.float32)
        worker.detected_box = None
        worker.missing_boxes = 0
        worker.detector = SimpleNamespace(track=lambda *args, **kwargs: [
            SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.empty((0, 4))))])
        worker._visual_feature = lambda *args: self.fail("stale box reached HMR2")
        frame = Frame(0, np.zeros((2, 2, 3), np.uint8), time.monotonic())
        with patch.dict(sys.modules, {"cv2": SimpleNamespace(COLOR_RGB2BGR=0,
                                                             cvtColor=lambda image, _: image)}):
            self.assertIsNone(worker._feature(frame, allow_missing=True))
        self.assertIsNone(worker.last_box)

    def test_tcam_uses_only_past_context_slots(self):
        worker = object.__new__(VisionWorker)
        worker.torch = torch
        worker.device = "cpu"
        feature = np.zeros(1024, np.float32)
        worker.samples = deque((Sample(i, i, i / 30, float(i + 1), feature)
                                for i in range(50)), maxlen=50)
        worker.context = deque((np.ones(1024, np.float32) for _ in range(3)), maxlen=8)

        def frontend(x, *, context_kv, window_mask, return_motion_feature):
            self.assertEqual(tuple(context_kv["img_kv"].shape), (1, 16, 1024))
            self.assertEqual(window_mask[0].tolist(), [True] * 3 + [False] * 13)
            self.assertTrue(return_motion_feature)
            return torch.zeros((50, 1024))

        worker.frontend = frontend
        u, _ = worker._window_u()
        self.assertEqual(u.shape, (50, 1024))
        self.assertEqual(len(worker.context), 4)

    @unittest.skipIf(module is None, "Local model package is unavailable")
    def test_student_heads_keep_correct_history(self):
        torch.manual_seed(2)
        u = np.random.default_rng(2).standard_normal((54, 1024)).astype(np.float32)
        for kind in ("causal_tcn", "mlp"):
            head = module.UZHead(kind=kind, width=32, dilations=(1, 2, 4), dropout=0).eval()
            worker = object.__new__(VisionWorker)
            worker.variant = "tcn" if kind == "causal_tcn" else "mlp"
            worker.torch = torch
            worker.device = "cpu"
            worker.head = head
            worker.tcn_history = deque(maxlen=14)
            worker.tcam_seconds = deque(maxlen=10)
            worker.backend_seconds = deque(maxlen=10)
            worker.result_count = 0
            worker.last_emitted = -1
            for start in (0, 4):
                worker.samples = deque((Sample(i, i, i / 30, float(i + 1), u[i])
                                        for i in range(start, start + 50)), maxlen=50)
                worker._window_u = lambda start=start: (u[start:start + 50], None)
                result = worker._window_results()
                self.assertEqual([row[0].index for row in result], list(range(start + 46, start + 50)))
                with torch.inference_mode():
                    if kind == "causal_tcn":
                        reference = head(torch.from_numpy(u[32:start + 50][None]))[0, -4:].numpy()
                    else:
                        reference = head(torch.from_numpy(u[start + 46:start + 50][None]))[0].numpy()
                np.testing.assert_allclose(np.stack([row[1] for row in result]), reference,
                                           atol=2e-5, rtol=2e-5)

    def test_original_waits_for_right_context_and_emits_once(self):
        worker = object.__new__(VisionWorker)
        worker.original_raw = deque(maxlen=120)
        worker.original_first = 46
        worker.last_emitted = -1
        worker.backend_seconds = deque(maxlen=10)
        worker.result_count = 0
        worker.robot_spec = object()
        worker.final_robot_postprocess = lambda raw, *args, **kwargs: raw
        worker.adapter = SimpleNamespace(observations=lambda s, t: (
            np.zeros((len(t), 64), np.float32), np.zeros((len(t), 463), np.float32)))
        worker.backward = SimpleNamespace(encode=lambda state, priv: np.tile(
            np.r_[16., np.zeros(255)].astype(np.float32), (len(state), 1)))

        def append(start, stop):
            for i in range(start, stop):
                sample = Sample(i, i, i / 30, float(i + 1), np.zeros(1024, np.float32))
                worker.original_raw.append((sample, np.zeros(38, np.float32),
                                            np.array([1., 0., 0., 0.], np.float32),
                                            np.zeros(2, np.float32)))

        append(0, 89)
        self.assertEqual(worker._original_ready(), [])
        append(89, 90)
        self.assertEqual([row[0].index for row in worker._original_ready()], list(range(46, 58)))
        append(90, 102)
        self.assertEqual([row[0].index for row in worker._original_ready()], list(range(58, 70)))
        self.assertEqual(worker._original_ready(), [])
        self.assertEqual([row[0].index for row in worker._original_ready(final=True)],
                         list(range(70, 102)))

    def test_late_capture_runs_one_window_not_every_missing_sample(self):
        worker = object.__new__(VisionWorker)
        worker.variant = "original"
        feature = np.zeros(1024, np.float32)
        worker.samples = deque((Sample(i, i, i / 30, float(i + 1), feature)
                                for i in range(50)), maxlen=50)
        worker.last_sample = worker.samples[-1]
        worker.total_samples = 50
        worker.last_window_total = 50
        worker.gap_samples = 0
        calls = []
        worker._window_results = lambda count: calls.append(count) or []
        worker._consume(Frame(70, None, time.monotonic(), 70 / 30), feature)
        self.assertEqual(calls, [21])
        self.assertEqual(worker.gap_samples, 20)
        self.assertEqual(worker.samples[-1].index, 70)

    def test_twenty_fps_frame_never_fills_an_earlier_sample(self):
        worker = object.__new__(VisionWorker)
        worker.variant = "mlp"
        feature = np.zeros(1024, np.float32)
        worker.samples = deque(maxlen=50)
        worker.last_sample = None
        worker.total_samples = 0
        worker.last_window_total = 0
        worker.gap_samples = 0
        worker._consume(Frame(0, None, 1.0, 0.0), feature)
        worker._consume(Frame(1, None, 1.05, 0.05), feature)
        self.assertEqual([(s.index, s.source_frame_id) for s in worker.samples],
                         [(0, 0), (1, 0), (2, 1)])


if __name__ == "__main__":
    unittest.main()
