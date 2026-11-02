"""Heavy RGB-to-z inference; only the resulting packets cross into 50 Hz control."""

import importlib.util
import json
import socket
import sys
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import numpy as np

try:
    from .latent_packet import VARIANTS, encode_packet
    from .latent_transport import LatestTcpSender
    from .vision_source import CameraPreview, camera_frames, replay_video
except ImportError:  # standalone worker process uses this directory without importing RoboJuDo
    from latent_packet import VARIANTS, encode_packet
    from latent_transport import LatestTcpSender
    from vision_source import CameraPreview, camera_frames, replay_video


def _load_bundle(root: Path):
    root = root.resolve()
    if not (root / "package.json").is_file():
        raise FileNotFoundError(f"Missing model package: {root}")
    sys.path.insert(0, str(root / "code"))
    spec = importlib.util.spec_from_file_location("rgb_bfm_bundle_pipeline", root / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisionWorker:
    WINDOW = 50
    STRIDE = 10

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
            from omegaconf import OmegaConf
            from video_frontend import infer_features
            from lib.robot.robot_spec import load_robot_spec
            from lib.util.ufo_bfm import BackwardEncoder, source_adapter
            from postprocess.final_robot_postprocess import final_robot_postprocess

            self.teacher = self.bundle.Teacher(self.frontend, device)
            self.tcam_cfg = OmegaConf.create(self.bundle.load_info()["inference_config"])
            self.infer_features = infer_features
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
            self.head = module.UZHead(**payload["model_config"]).to(device).eval()
            self.head.load_state_dict(payload["state_dict"], strict=True)
        self.frames = deque(maxlen=self.WINDOW)
        self.features = deque(maxlen=self.WINDOW)
        self.last_box = None
        self.detected_box = None
        self.missing_boxes = 0
        self.last_emitted = -1
        self.inference_seconds = []
        self.capture_to_send = []
        self.frame_count = 0
        self.result_count = 0
        self.send_failures = 0
        self.sent_ids = []
        self.sent_z = []
        self.sent_times = []

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
            if self.last_box is None:
                return None
            if self.missing_boxes > 5:
                if allow_missing:
                    self.last_box = None
                    return None
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
        return np.asarray(feature, dtype=np.float32)

    def _window_z(self):
        features = np.stack(self.features).astype(np.float32)
        if self.variant == "original":
            u, raw, probability, _ = self.infer_features(
                features, self.teacher, self.teacher.contact, self.tcam_cfg,
                self.torch, self.device)
            raw["fps"] = np.array([30], dtype=np.int64)
            s = self.final_robot_postprocess(
                raw, self.robot_spec, probability,
                (probability > 0.5).astype(np.int64), fps=30)
            times = np.arange(len(features), dtype=np.float64) / 30.0
            state, privileged = self.adapter.observations(s, times)
            z = self.backward.encode(state, privileged)
        else:
            u, _ = self.bundle.encode_features(features, self.frontend, self.device)
            with self.torch.inference_mode():
                tensor = self.torch.from_numpy(u[None]).to(self.device)
                z = self.head(tensor)[0].detach().cpu().numpy()
        if z.shape != (len(features), 256) or not np.all(np.isfinite(z)):
            raise ValueError("Invalid streaming z")
        return z.astype(np.float32)

    def _emit(self, z, *, final=False):
        # The middle ten frames have twenty future frames for TCAM and filters.
        stop = len(self.frames) if final else len(self.frames) - 20
        first_index = self.frames[0].index
        for local in range(stop):
            frame = self.frames[local]
            if frame.index <= self.last_emitted:
                continue
            self.last_emitted = frame.index
            self.result_count += 1
            yield frame, z[local], final and local == stop - 1

    def process(self, source, *, finalize: bool = True, on_frame=None):
        pending = []
        for frame in source:
            self.frame_count += 1
            feature = self._feature(frame, allow_missing=not finalize)
            if on_frame is not None and not on_frame(frame, self.detected_box):
                break
            if feature is None:
                if not finalize:
                    # A live camera can lose the person; restart the temporal
                    # window when they reappear instead of ending the demo.
                    pending.clear()
                    self.frames.clear()
                    self.features.clear()
                    continue
                pending.append(frame)
                continue
            ready = [(old, self._visual_feature(old, self.last_box)) for old in pending]
            pending.clear()
            ready.append((frame, feature))
            for current, value in ready:
                self.frames.append(current)
                self.features.append(value)
                if len(self.frames) == self.WINDOW and (current.index + 1 - self.WINDOW) % self.STRIDE == 0:
                    began = time.monotonic()
                    z = self._window_z()
                    self.inference_seconds.append(time.monotonic() - began)
                    yield from self._emit(z)
        if not finalize:
            return
        if pending:
            raise RuntimeError("No person was detected in the video")
        if self.frames:
            began = time.monotonic()
            z = self._window_z()
            self.inference_seconds.append(time.monotonic() - began)
            yield from self._emit(z, final=True)

    def run(self, *, video: str | None, camera: int | None, socket_path: str,
            metrics_path: str | None, output_dir: str | None = None,
            camera_backend: str = "opencv", show_camera: bool = False,
            show_detection_box: bool = False, tcp_host: str | None = None,
            tcp_port: int = 5541, healthcheck=None):
        if (video is None) == (camera is None):
            raise ValueError("Choose exactly one video or camera source")
        if show_detection_box and (video is not None or not show_camera):
            raise ValueError("Detection boxes require a camera preview")
        source = (replay_video(video) if video is not None else
                  camera_frames(camera, backend=camera_backend))
        endpoint = Path(socket_path)
        if tcp_host is None:
            deadline = time.monotonic() + 30.0
            while not endpoint.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Control socket did not appear: {endpoint}")
                time.sleep(0.05)
        preview = CameraPreview(show_detection_box=show_detection_box) if show_camera else None

        def on_frame(frame, detected_box):
            if healthcheck is not None:
                healthcheck()
            return preview.show(frame, detected_box) if preview is not None else True

        network_sender = None
        try:
            with ExitStack() as stack:
                if tcp_host is None:
                    sock = stack.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM))
                else:
                    network_sender = LatestTcpSender(tcp_host, tcp_port)
                    stack.callback(network_sender.close)
                for frame, z, eof in self.process(
                    source, finalize=video is not None,
                    on_frame=on_frame if preview is not None or healthcheck is not None else None,
                ):
                    try:
                        packet = encode_packet(self.variant, frame.index, frame.captured_at, z, eof=eof)
                        if network_sender is None:
                            sock.sendto(packet, str(endpoint))
                        else:
                            network_sender.send(packet)
                        self.capture_to_send.append(time.monotonic() - frame.captured_at)
                    except OSError:
                        # Keep the visual comparison after a control-side stale fault.
                        self.send_failures += 1
                    if output_dir:
                        self.sent_ids.append(frame.index)
                        self.sent_z.append(z.copy())
                        self.sent_times.append(frame.captured_at)
        except KeyboardInterrupt:
            if video is not None:
                raise
        finally:
            source.close()
            if preview is not None:
                preview.close()
        if output_dir:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / "frame_id.npy", np.asarray(self.sent_ids, dtype=np.int64))
            np.save(target / "z.npy", np.stack(self.sent_z).astype(np.float32))
            times = np.asarray(self.sent_times, dtype=np.float64)
            np.save(target / "timestamps.npy", times - times[0])
        if metrics_path:
            data = {"variant": self.variant, "frames_read": self.frame_count,
                    "latents_emitted": self.result_count, "socket_send_failures": self.send_failures,
                    "tcp_packets_sent": network_sender.sent if network_sender is not None else None,
                    "frames_dropped": self.frame_count - self.result_count,
                    "window_inference_p50_s": float(np.percentile(self.inference_seconds, 50)) if self.inference_seconds else None,
                    "window_inference_p95_s": float(np.percentile(self.inference_seconds, 95)) if self.inference_seconds else None,
                    "capture_to_send_p50_s": float(np.percentile(self.capture_to_send, 50)) if self.capture_to_send else None,
                    "capture_to_send_p95_s": float(np.percentile(self.capture_to_send, 95)) if self.capture_to_send else None}
            Path(metrics_path).write_text(json.dumps(data, indent=2) + "\n")
        return self.result_count
