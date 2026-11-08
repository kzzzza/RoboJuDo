"""Per-processed-frame causal inference, with GPU-resident visual/temporal state.

Inspired by robojudo_ly/scripts/run_rgb2sm_vision.py. Retains this project's
YOLOv8x, HMR2, TCAM and student weights, and per-frame person-loss detection.
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

try:
    from .vision_accel import (DEFAULT_CACHE, CachedTemporal, HalfVisual, TensorRTVisual,
                               trt_detector_predictor, validated_cache)
    from .vision_worker import Sample, VisionWorker
except ImportError:
    from vision_accel import (DEFAULT_CACHE, CachedTemporal, HalfVisual, TensorRTVisual,
                              trt_detector_predictor, validated_cache)
    from vision_worker import Sample, VisionWorker


class FastVisionWorker(VisionWorker):
    CONTEXT_STRIDE = 4

    def __init__(self, packages, variant, device, *, cache_dir=DEFAULT_CACHE, hmr_backend="auto",
                 detector_backend="auto"):
        if variant not in ("tcn", "mlp") or torch.device(device).type != "cuda":
            raise ValueError("Low-latency vision requires CUDA and tcn/mlp; use legacy for other modes")
        torch.cuda.set_device(device)
        self.cache_dir = Path(cache_dir)
        self.cache = validated_cache(self.cache_dir, Path(packages) / "original_pipeline", device)
        self.requested_hmr_backend = hmr_backend
        self.requested_detector_backend = detector_backend
        super().__init__(packages, variant, device)
        self.detector_seconds = deque(maxlen=4096)
        self.preprocess_seconds = deque(maxlen=4096)
        self.hmr_seconds = deque(maxlen=4096)
        self._feature_events = deque(maxlen=4096)
        self._temporal_events = deque(maxlen=4096)
        self._head_events = deque(maxlen=4096)
        self._processed_start = self._processed_end = None
        self._source_size = None
        began = time.monotonic()
        # Warm up before camera capture starts: cold kernels must not become video latency.
        with torch.inference_mode():
            for _ in range(5):
                self.detector.predict(np.zeros((480, 640, 3), np.uint8), device=device, half=True,
                                      predictor=self._detector_predictor,
                                      classes=[0], conf=.5, verbose=False)
                self.visual({"img": torch.zeros(1, 3, 256, 256, device=device)})
                self._temporal_forward(torch.zeros(1, 50, 1024, device=device), None, None)
                self._temporal_forward(torch.zeros(1, 50, 1024, device=device),
                                       torch.zeros(1, 16, 1024, device=device),
                                       torch.zeros(1, 16, dtype=torch.bool, device=device))
                self.head(torch.zeros(1, 64, 1024, device=device),
                          torch.ones(1, 64, dtype=torch.bool, device=device))
        torch.cuda.synchronize(device)
        self.warmup_s = time.monotonic() - began
        self.reset_stream()
        print(f"Low-latency vision: HMR2={self.hmr_backend}, TCAM={self.temporal_backend}, "
              f"YOLOv8x={self.detector_backend} every frame; one new z per processed image", flush=True)

    def _load_detector(self):
        from ultralytics import YOLO

        available = self.cache and "yolov8x.engine" in self.cache["artifacts"]
        if self.requested_detector_backend == "tensorrt" and not available:
            raise RuntimeError("No validated YOLOv8x TensorRT cache; rebuild the acceleration cache")
        if available and self.requested_detector_backend != "torch":
            import tensorrt
            import ultralytics

            if tensorrt.__version__ != self.cache["tensorrt"] or ultralytics.__version__ != self.cache["ultralytics"]:
                raise RuntimeError("Detector acceleration runtime changed; rebuild the acceleration cache")
            self.detector_backend = "tensorrt-fp16"
            self._detector_predictor = trt_detector_predictor(self.device)
            return YOLO(str(self.cache_dir / "yolov8x.engine"), task="detect")
        self.detector_backend = "torch-fp16"
        self._detector_predictor = None
        return super()._load_detector()

    def _load_visual(self):
        engine_available = self.cache and "hmr2.engine" in self.cache["artifacts"]
        if self.requested_hmr_backend == "tensorrt" and not engine_available:
            raise RuntimeError("No validated HMR2 TensorRT cache; run scripts/prepare_ufo_vision_accel.py")
        if engine_available and self.requested_hmr_backend != "torch":
            self.hmr_backend = "tensorrt-fp16"
            return TensorRTVisual(self.cache_dir / "hmr2.engine", self.device, self.cache["tensorrt"])
        self.hmr_backend = "torch-fp16"
        return HalfVisual(self.bundle.load_visual(self.device))

    def _load_frontend(self):
        if self.cache:
            self.temporal_backend = "torchscript-" + self.cache.get("temporal_precision", "fp32")
            return CachedTemporal(self.cache_dir, self.device, self.cache.get("temporal_precision", "fp32"))
        self.temporal_backend = "torch-fp16"
        return self.bundle.load_frontend(self.device).half().eval()

    def _temporal_forward(self, x, context, mask):
        if isinstance(self.frontend, CachedTemporal):
            return self.frontend(x, context, mask)
        with torch.autocast("cuda", dtype=torch.float16):
            return self.frontend(x, context_kv=None if context is None else {"img_kv": context},
                                 window_mask=mask, return_motion_feature=True).reshape(50, 1024).float()

    @staticmethod
    def _start_event():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _feature(self, frame, *, allow_missing=False):
        import cv2
        if self._processed_start is None:
            self._processed_start = time.monotonic()
        self._source_size = [int(frame.rgb.shape[1]), int(frame.rgb.shape[0])]
        self.detected_box = None
        began = time.perf_counter()
        result = self.detector.predict(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), device=self.device,
                                       half=True, classes=[0], conf=.5, verbose=False,
                                       predictor=getattr(self, "_detector_predictor", None))[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        detector_s = time.perf_counter() - began
        self.detector_seconds.append(detector_s)
        if not len(boxes):
            self.last_box = None
            self._feature_events.append((None, None, detector_s, 0.0))
            self._processed_end = time.monotonic()
            return None  # never reuse a box when no person was detected
        area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        self.last_box = boxes[int(np.argmax(area))].astype(np.float32)
        self.detected_box = self.last_box.copy()
        self.missing_boxes = 0
        feature = self._encode_box(frame, self.last_box, detector_s)
        self._processed_end = time.monotonic()
        return feature

    def _visual_feature(self, frame, box):
        # Finite-file warm-up may backfill initial frames after the first detection.
        return self._encode_box(frame, box, 0.)

    def _encode_box(self, frame, box, detector_s):
        import cv2
        from lib.util.hmr2_preprocess import get_bbx_xys_from_xyxy, prepare_crops

        began = time.perf_counter()
        # Preserve the current project's exact interpolation, crop and normalization.
        half = cv2.resize(frame.rgb, (frame.rgb.shape[1] // 2, frame.rgb.shape[0] // 2))
        square = get_bbx_xys_from_xyxy(torch.from_numpy(box[None]), base_enlarge=1.2)
        crop = prepare_crops(half[None], square, image_scale=.5)
        preprocess_s = time.perf_counter() - began
        self.preprocess_seconds.append(preprocess_s)
        start = self._start_event()
        with torch.inference_mode():
            feature = self.visual({"img": crop.to(self.device)})[0]
        end = self._start_event()
        self._feature_events.append((start, end, detector_s, preprocess_s))
        if feature.shape != (1024,):
            raise ValueError("Invalid accelerated HMR2 feature shape")
        return feature  # no GPU -> NumPy -> GPU round trip

    def _remember_context(self):
        features = torch.stack([item.feature for item in self.samples])
        # Match a fully held 50-token window during initial warm-up.
        mean = (features.sum(0) + features[0] * (self.WINDOW - len(features))) / self.WINDOW
        self.context.append(mean)

    def _consume(self, frame, feature):
        source_time = frame.source_time if frame.source_time is not None else frame.index / 30.
        index = int(np.ceil(source_time * 30 - 1e-6))
        if self.last_sample is not None and index <= self.last_sample.index:
            return []
        if self.last_sample is not None and index - self.last_sample.index - 1 > self.MAX_GAP:
            self.reset_stream()
            self.resets += 1
        if self.last_sample is not None:
            previous = self.last_sample
            for missing in range(previous.index + 1, index):
                self.gap_samples += 1
                self._append(Sample(missing, previous.source_frame_id, missing / 30.,
                                    previous.captured_at, previous.feature))
                if (missing + 1) % self.CONTEXT_STRIDE == 0:
                    self._remember_context()
        self._append(Sample(index, frame.index, index / 30., frame.captured_at, feature))
        result = []
        if len(self.samples) == self.WINDOW:
            delta = self.total_samples - self.last_window_total
            count = min(delta, self.WINDOW)
            self.last_window_total = self.total_samples
            result = self._window_results(count=count)
        # Context stride is independent of output cadence; no current/future context leakage.
        if (index + 1) % self.CONTEXT_STRIDE == 0:
            self._remember_context()
        return result

    def _window_u(self):
        start = self._start_event()
        x = torch.stack([item.feature for item in self.samples])[None].float()
        context = mask = None
        if self.context:
            context = torch.zeros(1, 16, 1024, device=self.device)
            mask = torch.zeros(1, 16, dtype=torch.bool, device=self.device)
            context[0, :len(self.context)] = torch.stack(list(self.context))
            mask[0, :len(self.context)] = True
        with torch.inference_mode():
            u = self._temporal_forward(x, context, mask).float()
        self._temporal_events.append((start, self._start_event()))
        return u, x

    def _window_results(self, count=None):
        u, _ = self._window_u()
        count = min(count or 1, self.WINDOW)
        new_u = u[-count:]
        start = self._start_event()
        with torch.inference_mode():
            if self.variant == "tcn":
                if not self.tcn_history:
                    self.tcn_history.extend(u[max(0, len(u) - count - 14):len(u) - count])
                history = list(self.tcn_history)
                values = torch.zeros(1, 14 + count, 1024, device=self.device)
                valid = torch.zeros(1, 14 + count, dtype=torch.bool, device=self.device)
                if history:
                    values[0, 14 - len(history):14] = torch.stack(history)
                    valid[0, 14 - len(history):14] = True
                values[0, 14:] = new_u
                valid[0, 14:] = True
                z = self.head(values, valid)[0, -1]
                self.tcn_history.extend(new_u)
            else:
                z = self.head(new_u[-1:][None])[0, -1]
        self._head_events.append((start, self._start_event()))
        latent = z.float().detach().cpu().numpy().copy()
        if latent.shape != (256,) or not np.isfinite(latent).all() or not np.isclose(np.linalg.norm(latent), 16, atol=1e-3):
            raise ValueError("Invalid low-latency latent")
        sample = self.samples[-1]
        self.last_emitted = sample.index
        self.last_output_sample = sample
        self.result_count += 1
        self._processed_end = time.monotonic()
        return [(sample, latent, False)]  # send the latest true frame, never a synthetic FPS repeat

    def _finish_metrics(self):
        torch.cuda.synchronize(self.device)
        feature = []
        for start, end, detector_s, preprocess_s in self._feature_events:
            hmr_s = 0.0 if start is None else start.elapsed_time(end) / 1000.
            if start is not None:
                self.hmr_seconds.append(hmr_s)
            feature.append(detector_s + preprocess_s + hmr_s)
        self.feature_seconds = deque(feature, maxlen=4096)
        self.tcam_seconds = deque((a.elapsed_time(b) / 1000. for a, b in self._temporal_events), maxlen=4096)
        self.backend_seconds = deque((a.elapsed_time(b) / 1000. for a, b in self._head_events), maxlen=4096)

    def _extra_metrics(self):
        extra = dict(vision_profile="low-latency", hmr_backend=self.hmr_backend,
                     detector_backend=self.detector_backend,
                     temporal_backend=self.temporal_backend, detector_weights="yolov8x.pt",
                     detector_every=1, gpu_resident_features=True, logical_hz=30,
                     context_stride=self.CONTEXT_STRIDE, output_stride="each processed image",
                     source_size=self._source_size, warmup_s=self.warmup_s,
                     timing_sample_capacity=4096,
                     timing_method="CUDA events resolved after final output synchronization")
        if self._processed_end is not None and self._processed_end > self._processed_start:
            extra["processed_frame_rate_hz"] = (self.frame_count - 1) / (self._processed_end - self._processed_start)
        for name in ("detector", "preprocess", "hmr"):
            values = getattr(self, name + "_seconds")
            for percentile in (50, 95):
                extra[f"{name}_p{percentile}_s"] = float(np.percentile(values, percentile)) if values else None
        return extra
