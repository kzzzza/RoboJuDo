"""Bounded RGB-to-z streaming worker shared by MuJoCo and G1 Real."""

import importlib.util
import json
import secrets
import socket
import sys
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .latent_packet import VARIANTS, encode_packet
    from .latent_transport import LatestTcpSender
    from .vision_source import CameraPreview, LatestFrameSource, camera_frames, replay_video
except ImportError:  # standalone vision process does not import robot dependencies
    from latent_packet import VARIANTS, encode_packet
    from latent_transport import LatestTcpSender
    from vision_source import CameraPreview, LatestFrameSource, camera_frames, replay_video


def _load_bundle(root: Path):
    root = root.resolve()
    if not (root / "package.json").is_file():
        raise FileNotFoundError(f"Missing model package: {root}")
    sys.path.insert(0, str(root / "code"))
    spec = importlib.util.spec_from_file_location("rgb_bfm_bundle_pipeline", root / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Sample:
    index: int
    source_frame_id: int
    sample_time: float
    captured_at: float
    feature: np.ndarray


class VisionWorker:
    WINDOW = 50
    STRIDE = 4
    CONTEXT = 8
    ORIGINAL_BUFFER = 120
    ORIGINAL_RIGHT = 32
    ORIGINAL_EMIT_STEP = 12
    MAX_GAP = 49  # one window can reconstruct at most fifty missing samples

    def __init__(self, packages: Path, variant: str, device: str):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.device = device
        self.packages = Path(packages).expanduser().resolve()
        self.bundle = _load_bundle(self.packages / "original_pipeline")
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(1)
        self.torch = torch
        self.detector = YOLO(str(self.bundle.ROOT / "models/yolov8x.pt"))
        self.visual = self.bundle.load_visual(device)
        self.frontend = self.bundle.load_frontend(device)
        if variant == "original":
            from lib.robot.robot_spec import load_robot_spec
            from lib.util.robot_rotation import rot6d_to_quat_wxyz
            from lib.util.ufo_bfm import BackwardEncoder, source_adapter
            from postprocess.contact_infer import predict_contact_prob
            from postprocess.final_robot_postprocess import final_robot_postprocess

            self.teacher = self.bundle.Teacher(self.frontend, device)
            self.rot6d_to_quat = rot6d_to_quat_wxyz
            self.predict_contact_prob = predict_contact_prob
            self.robot_spec = load_robot_spec(
                str(self.bundle.ROOT / "code/config/robot/g1.yaml"),
                project_root=str(self.bundle.ROOT / "code"))
            self.final_robot_postprocess = final_robot_postprocess
            self.adapter = source_adapter()
            self.backward = BackwardEncoder()
        else:
            student = self.packages / f"{variant}_pipeline"
            spec = importlib.util.spec_from_file_location("rgb_bfm_uz_head", student / "code/uz_head.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            payload = torch.load(student / "models/distilled_head.pt", map_location="cpu", weights_only=True)
            kind = "causal_tcn" if variant == "tcn" else "mlp"
            if payload["model_config"]["kind"] != kind:
                raise ValueError(f"Expected {kind} head in {student}")
            self.head = module.UZHead(**payload["model_config"]).to(device).eval()
            self.head.load_state_dict(payload["state_dict"], strict=True)
        self.last_box = None
        self.detected_box = None
        self.missing_boxes = 0
        self.person_loss_reported = False
        self.reset_stream()
        self.feature_seconds = deque(maxlen=4096)
        self.tcam_seconds = deque(maxlen=4096)
        self.backend_seconds = deque(maxlen=4096)
        self.capture_to_send = deque(maxlen=4096)
        self.frame_count = 0
        self.first_capture_time = None
        self.first_send_time = None
        self.last_send_time = None
        self.result_count = 0
        self.packets_sent = 0
        self.latent_packets_sent = 0
        self.send_failures = 0
        self.gap_samples = 0
        self.samples_skipped = 0
        self.resets = 0
        self.sent_ids = []
        self.sent_z = []
        self.sent_times = []
        self.record_original = False
        self.original_s = {name: [] for name in ("root_trans", "root_rot_quat", "dof")}

    def reset_stream(self):
        self.session_id = secrets.randbits(64)
        self.samples = deque(maxlen=self.WINDOW)
        self.context = deque(maxlen=self.CONTEXT)
        self.original_raw = deque(maxlen=self.ORIGINAL_BUFFER)
        self.tcn_history = deque(maxlen=14)
        self.total_samples = 0
        self.last_window_total = 0
        self.last_sample = None
        self.last_emitted = -1
        self.last_output_sample = None
        self.original_first = None

    def _feature(self, frame, *, allow_missing=False):
        import cv2

        self.detected_box = None
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        result = self.detector.track(bgr, device=self.device, persist=True, conf=0.5,
                                     classes=[0], verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        if len(boxes):
            area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
            self.last_box = boxes[int(np.argmax(area))].astype(np.float32)
            self.detected_box = self.last_box.copy()
            self.missing_boxes = 0
        else:
            self.missing_boxes += 1
            if allow_missing:
                # Live control must not run HMR2 on an empty crop using a stale box.
                self.last_box = None
                return None
            if self.last_box is None:
                return None
            if self.missing_boxes > 5:
                raise RuntimeError(f"No person detected at frame {frame.index}")
        return self._visual_feature(frame, self.last_box)

    def _visual_feature(self, frame, box):
        import cv2
        from lib.util.hmr2_preprocess import get_bbx_xys_from_xyxy, prepare_crops

        half = cv2.resize(frame.rgb, (frame.rgb.shape[1] // 2, frame.rgb.shape[0] // 2))
        square = get_bbx_xys_from_xyxy(self.torch.from_numpy(box[None]), base_enlarge=1.2)
        crop = prepare_crops(half[None], square, image_scale=0.5).to(self.device)
        with self.torch.inference_mode():
            feature = self.visual({"img": crop})[0].detach().cpu().numpy()
        value = np.asarray(feature, dtype=np.float32)
        if value.shape != (1024,) or not np.isfinite(value).all():
            raise ValueError("Invalid HMR2 feature")
        return value

    def _window_u(self):
        features = np.stack([item.feature for item in self.samples]).astype(np.float32)
        x = self.torch.from_numpy(features[None]).to(self.device)
        context = mask = None
        if self.context:
            # The checkpoint has 8 past + 8 future attention slots. Only fill
            # the past slots; never ask it for a window that has not arrived.
            values = np.zeros((1, self.CONTEXT * 2, 1024), dtype=np.float32)
            valid = np.zeros((1, self.CONTEXT * 2), dtype=bool)
            history = list(self.context)
            values[0, :len(history)] = history
            valid[0, :len(history)] = True
            context = {"img_kv": self.torch.from_numpy(values).to(self.device)}
            mask = self.torch.from_numpy(valid).to(self.device)
        with self.torch.inference_mode():
            u = self.frontend(x, context_kv=context, window_mask=mask,
                              return_motion_feature=True).reshape(self.WINDOW, 1024)
        self.context.append(features.mean(axis=0))
        value = u.detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("Invalid TCAM motion feature")
        return value, x

    def _original_motion(self, u, x):
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(u[None]).to(self.device)
            motion = self.teacher.predict_from_feature(tensor, 1, self.WINDOW, "g1")["motion"][0]
            probability = self.predict_contact_prob(
                self.teacher.contact, tensor, x)["prob"][0]
            quaternion = self.rot6d_to_quat(motion[:, 3:9]).cpu().numpy()
        raw = motion.detach().cpu().numpy().astype(np.float32)
        prob = probability.detach().cpu().numpy().astype(np.float32)
        return raw, quaternion.astype(np.float32), prob

    def _original_ready(self, *, final=False):
        if not self.original_raw:
            return []
        cutoff = self.original_raw[-1][0].index - (0 if final else self.ORIGINAL_RIGHT)
        ready = [(i, row[0]) for i, row in enumerate(self.original_raw)
                 if self.last_emitted < row[0].index <= cutoff
                 and row[0].index >= self.original_first]
        if not ready or (not final and len(ready) < self.ORIGINAL_EMIT_STEP):
            return []
        began = time.monotonic()
        rows = list(self.original_raw)
        raw = np.stack([row[1] for row in rows])
        quaternion = np.stack([row[2] for row in rows])
        probability = np.stack([row[3] for row in rows])
        motion = {"fps": np.array([30], dtype=np.int64),
                  "root_trans": raw[:, :3], "root_rot_6d": raw[:, 3:9],
                  "root_rot_quat": quaternion, "dof": raw[:, 9:]}
        s = self.final_robot_postprocess(
            motion, self.robot_spec, probability,
            (probability > 0.5).astype(np.int64), fps=30)
        times = np.arange(len(rows), dtype=np.float64) / 30.0
        state, privileged = self.adapter.observations(s, times)
        indices = np.asarray([i for i, _ in ready])
        z = self.backward.encode(state[indices], privileged[indices])
        if getattr(self, "record_original", False):
            for name in self.original_s:
                self.original_s[name].extend(s[name][indices].copy())
        self.backend_seconds.append(time.monotonic() - began)
        self.last_emitted = ready[-1][1].index
        self.last_output_sample = ready[-1][1]
        self.result_count += len(ready)
        return [(sample, latent, False) for (_, sample), latent in zip(ready, z)]

    def _window_results(self, count=None):
        count = self.STRIDE if count is None else count
        began = time.monotonic()
        u, x = self._window_u()
        self.tcam_seconds.append(time.monotonic() - began)
        current = list(self.samples)[-count:]
        if self.variant == "original":
            began = time.monotonic()
            raw, quat, prob = self._original_motion(u, x)
            self.backend_seconds.append(time.monotonic() - began)
            if self.original_first is None:
                self.original_first = list(self.samples)[-self.STRIDE].index
                entries = zip(self.samples, raw, quat, prob)
            else:
                entries = zip(current, raw[-count:], quat[-count:], prob[-count:])
            self.original_raw.extend(entries)
            return self._original_ready()

        began = time.monotonic()
        new_u = u[-count:]
        if self.variant == "tcn":
            if not self.tcn_history:
                self.tcn_history.extend(u[-count - 14:-count])
            history = list(self.tcn_history)
            values = np.zeros((14 + count, 1024), dtype=np.float32)
            valid = np.zeros((14 + count,), dtype=bool)
            values[14 - len(history):14] = history
            valid[14 - len(history):14] = True
            values[14:] = new_u
            valid[14:] = True
            with self.torch.inference_mode():
                z = self.head(self.torch.from_numpy(values[None]).to(self.device),
                              self.torch.from_numpy(valid[None]).to(self.device))[0, -count:]
            self.tcn_history.extend(new_u)
        else:
            with self.torch.inference_mode():
                z = self.head(self.torch.from_numpy(new_u[None]).to(self.device))[0]
        latent = z.detach().cpu().numpy().astype(np.float32)
        self.backend_seconds.append(time.monotonic() - began)
        if latent.shape != (count, 256) or not np.isfinite(latent).all():
            raise ValueError("Invalid streaming z")
        self.last_emitted = current[-1].index
        self.last_output_sample = current[-1]
        self.result_count += len(current)
        return [(sample, value, False) for sample, value in zip(current, latent)]

    def _append(self, sample):
        self.samples.append(sample)
        self.last_sample = sample
        self.total_samples += 1

    def _consume(self, frame, feature):
        source_time = frame.source_time
        if source_time is None:
            source_time = frame.index / 30.0
        # The sample time must never precede the source frame it uses.
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
                self._append(Sample(
                    missing, previous.source_frame_id, missing / 30.0,
                    previous.captured_at, previous.feature))
        self._append(Sample(index, frame.index, index / 30.0, frame.captured_at, feature))
        if len(self.samples) < self.WINDOW:
            return []
        delta = self.total_samples - self.last_window_total
        if self.last_window_total and delta < self.STRIDE:
            return []
        self.last_window_total = self.total_samples
        if self.variant != "original" and delta > self.STRIDE:
            # The newest four samples are useful; an older skipped batch must
            # not be replayed or silently entered into the TCN history.
            self.samples_skipped += delta - self.STRIDE
            self.tcn_history.clear()
        count = min(delta, self.WINDOW) if self.variant == "original" and delta >= self.WINDOW else (
            min(delta, self.STRIDE) if self.variant != "original" else delta)
        return self._window_results(count=count)

    def process(self, source, *, finalize=True, on_frame=None):
        pending = []
        for frame in source:
            self.frame_count += 1
            if self.first_capture_time is None:
                self.first_capture_time = frame.captured_at
            began = time.monotonic()
            feature = self._feature(frame, allow_missing=not finalize)
            self.feature_seconds.append(time.monotonic() - began)
            if on_frame is not None and not on_frame(frame, self.detected_box):
                break
            if feature is None:
                if not finalize:
                    pending.clear()
                    if not self.person_loss_reported:
                        self.reset_stream()
                        self.resets += 1
                        self.person_loss_reported = True
                        source_time = frame.source_time
                        if source_time is None:
                            source_time = frame.index / 30.0
                        sample = Sample(int(np.ceil(source_time * 30 - 1e-6)), frame.index,
                                        source_time, frame.captured_at, np.zeros(1024, np.float32))
                        yield [(sample, None, False)]
                else:
                    pending.append(frame)
                continue
            self.person_loss_reported = False
            ready = [(old, self._visual_feature(old, self.last_box)) for old in pending]
            pending.clear()
            ready.append((frame, feature))
            for current, value in ready:
                batch = self._consume(current, value)
                if batch:
                    yield batch
        if not finalize:
            return
        if pending:
            raise RuntimeError("No person was detected in the video")
        if len(self.samples) == self.WINDOW:
            remainder = self.total_samples - self.last_window_total
            if remainder:
                batch = self._window_results(count=remainder)
                if batch:
                    yield batch
        if self.variant == "original":
            batch = self._original_ready(final=True)
            if batch:
                batch[-1] = (batch[-1][0], batch[-1][1], True)
                yield batch
            if self.result_count == 0:
                raise ValueError("Streaming input needs at least 50 valid 30 Hz samples")
        elif self.result_count == 0:
            raise ValueError("Streaming input needs at least 50 valid 30 Hz samples")
        else:
            # The last emitted latent is sufficient to signal EOF; no old
            # batch is replayed to the controller when the source closes.
            yield [(self.last_output_sample, None, True)]

    def run(self, *, video: str | None, camera: int | None, socket_path: str,
            metrics_path: str | None, output_dir: str | None = None,
            camera_backend: str = "opencv", show_camera: bool = False,
            show_detection_box: bool = False, tcp_host: str | None = None,
            tcp_port: int = 5541, healthcheck=None, preview=None):
        if (video is None) == (camera is None):
            raise ValueError("Choose exactly one video or camera source")
        if camera is not None and output_dir:
            raise ValueError("--output-dir records finite videos only; live camera memory must stay bounded")
        if show_detection_box and (video is not None or not show_camera):
            raise ValueError("Detection boxes require a camera preview")
        raw_source = replay_video(video) if video is not None else camera_frames(camera, backend=camera_backend)
        endpoint = Path(socket_path)
        if tcp_host is None:
            deadline = time.monotonic() + 30.0
            while not endpoint.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Control socket did not appear: {endpoint}")
                time.sleep(0.05)
        source = LatestFrameSource(raw_source)
        self.record_original = bool(output_dir and self.variant == "original")
        owns_preview = preview is None
        if show_camera and preview is None:
            preview = CameraPreview(show_detection_box=show_detection_box)

        def on_frame(frame, detected_box):
            if healthcheck is not None:
                healthcheck()
            return preview.show(frame, detected_box) if preview is not None else True

        network_sender = None
        last_z = None
        try:
            with ExitStack() as stack:
                if tcp_host is None:
                    sock = stack.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM))
                else:
                    network_sender = LatestTcpSender(tcp_host, tcp_port)
                    stack.callback(network_sender.close)
                for batch in self.process(source, finalize=video is not None,
                                          on_frame=on_frame if preview is not None or healthcheck is not None else None):
                    for sample, z, _ in batch:
                        if z is not None:
                            last_z = z
                            if output_dir:
                                self.sent_ids.append(sample.index)
                                self.sent_z.append(z.copy())
                                self.sent_times.append(sample.sample_time)
                    sample, z, eof = batch[-1]
                    person_lost = z is None and not eof
                    if person_lost:
                        last_z = None
                        z = np.zeros(256, dtype=np.float32)
                    else:
                        z = last_z if z is None else z
                    if z is None:
                        continue
                    try:
                        packet = encode_packet(self.variant, sample.index, sample.captured_at, z,
                                               eof=eof, session_id=self.session_id,
                                               source_frame_id=sample.source_frame_id,
                                               sample_time=sample.sample_time,
                                               produced_time=time.monotonic(),
                                               person_lost=person_lost)
                        if network_sender is None:
                            sock.sendto(packet, str(endpoint))
                        else:
                            network_sender.send(packet)
                        self.packets_sent += 1
                        sent_at = time.monotonic()
                        if not person_lost:
                            self.latent_packets_sent += 1
                            if self.first_send_time is None:
                                self.first_send_time = sent_at
                            self.last_send_time = sent_at
                            self.capture_to_send.append(sent_at - sample.captured_at)
                    except OSError:
                        self.send_failures += 1
        except KeyboardInterrupt:
            if video is not None:
                raise
        finally:
            source.close()
            if preview is not None and owns_preview:
                preview.close()
        if output_dir:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / "frame_id.npy", np.asarray(self.sent_ids, dtype=np.int64))
            np.save(target / "z.npy", np.asarray(self.sent_z, dtype=np.float32).reshape(-1, 256))
            np.save(target / "timestamps.npy", np.asarray(self.sent_times, dtype=np.float64))
            if self.record_original:
                np.savez_compressed(target / "s.npz", **{
                    name: np.asarray(rows, dtype=np.float32)
                    for name, rows in self.original_s.items()})
        if metrics_path:
            def percentile(values, p):
                return float(np.percentile(values, p)) if values else None

            data = {"variant": self.variant, "frames_read": self.frame_count,
                    "capture_frames_dropped": source.dropped,
                    "missing_samples_reused": self.gap_samples,
                    "samples_skipped": self.samples_skipped,
                    "stream_resets": self.resets,
                    "latents_emitted": self.result_count,
                    "packets_sent": self.packets_sent,
                    "first_output_wait_s": (None if self.first_send_time is None else
                                            self.first_send_time - self.first_capture_time),
                    "output_rate_hz": (None if self.latent_packets_sent < 2 else
                                       (self.latent_packets_sent - 1) / (self.last_send_time - self.first_send_time)),
                    "socket_send_failures": self.send_failures,
                    "tcp_packets_sent": network_sender.sent if network_sender is not None else None,
                    "frames_dropped": source.dropped,
                    "feature_p50_s": percentile(self.feature_seconds, 50),
                    "feature_p95_s": percentile(self.feature_seconds, 95),
                    "tcam_p50_s": percentile(self.tcam_seconds, 50),
                    "tcam_p95_s": percentile(self.tcam_seconds, 95),
                    "backend_p50_s": percentile(self.backend_seconds, 50),
                    "backend_p95_s": percentile(self.backend_seconds, 95),
                    "window_inference_p50_s": percentile(self.tcam_seconds, 50),
                    "window_inference_p95_s": percentile(self.tcam_seconds, 95),
                    "capture_to_send_p50_s": percentile(self.capture_to_send, 50),
                    "capture_to_send_p95_s": percentile(self.capture_to_send, 95)}
            Path(metrics_path).write_text(json.dumps(data, indent=2) + "\n")
        return self.result_count
