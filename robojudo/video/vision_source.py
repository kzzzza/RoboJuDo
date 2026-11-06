"""Timestamped 30 Hz frames for the separate UFO vision process."""

import time
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    index: int
    rgb: object
    captured_at: float  # monotonic clock shared with the local control process
    source_time: float | None = None  # seconds on the source timeline


class LatestFrameSource:
    """Read the camera/video independently of inference, retaining one frame."""

    def __init__(self, source):
        self.source = source
        self.changed = threading.Condition()
        self.latest = None
        self.finished = False
        self.error = None
        self.dropped = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._read, name="VisionCapture", daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for frame in self.source:
                if self.stop.is_set():
                    break
                with self.changed:
                    if self.latest is not None:
                        self.dropped += 1
                    self.latest = frame
                    self.changed.notify()
        except BaseException as exc:
            self.error = exc
        finally:
            try:
                self.source.close()
            finally:
                with self.changed:
                    self.finished = True
                    self.changed.notify_all()

    def __iter__(self):
        while True:
            with self.changed:
                while self.latest is None and not self.finished:
                    self.changed.wait(timeout=0.2)
                frame = self.latest
                self.latest = None
                finished = self.finished
            if frame is not None:
                yield frame
            elif finished:
                if self.error is not None:
                    raise self.error
                return

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.0)


def replay_video(path: str | Path):
    """Replay source PTS; the worker causally maps frames onto a 30 Hz grid."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        if rate <= 0:
            raise ValueError("Video has no valid frame rate")
        start = time.monotonic()
        first_pts = None
        last_pts = -1.0
        for index, decoded in enumerate(container.decode(video=0)):
            if decoded.pts is None:
                raise ValueError("Video frame has no presentation timestamp")
            pts = float(decoded.pts * decoded.time_base)
            if first_pts is None:
                first_pts = pts
            pts -= first_pts
            if pts <= last_pts and index:
                raise ValueError("Video presentation timestamps are not increasing")
            last_pts = pts
            due = start + pts
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            yield Frame(index, decoded.to_ndarray(format="rgb24"), due, pts)


def camera_frames(device: int = 0, *, backend: str = "opencv"):
    """Read RGB frames from V4L2/OpenCV or Azure Kinect."""
    import cv2

    if backend == "k4a":
        try:
            from .azure_kinect import AzureKinectRGB
        except ImportError:  # standalone vision worker import
            from azure_kinect import AzureKinectRGB

        camera = AzureKinectRGB(device)
        camera.__enter__()
        read = camera.read
        close = camera.close
    elif backend == "opencv":
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")
        capture.set(cv2.CAP_PROP_FPS, 30)

        def read():
            ok, bgr = capture.read()
            captured_at = time.monotonic()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), captured_at

        close = capture.release
    else:
        raise ValueError(f"Unknown camera backend: {backend}")

    try:
        index = 0
        first_time = None
        while True:
            rgb, captured_at = read()
            if first_time is None:
                first_time = captured_at
            yield Frame(index, rgb, captured_at, captured_at - first_time)
            index += 1
    finally:
        close()


class CameraPreview:
    """Show the RGB frame after its person detector has run."""

    def __init__(self, *, show_detection_box: bool = False):
        import cv2

        self.cv2 = cv2
        self.name = "Camera RGB"
        self.show_detection_box = show_detection_box
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, 960, 540)
        cv2.moveWindow(self.name, 1260, 80)

    def show(self, frame: Frame, detected_box) -> bool:
        cv2 = self.cv2
        display = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        if self.show_detection_box and detected_box is not None:
            x1, y1, x2, y2 = (int(round(value)) for value in detected_box)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "person", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Camera RGB | q / Esc: quit", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(self.name, display)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            return False
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self):
        try:
            self.cv2.destroyWindow(self.name)
        except self.cv2.error:
            pass
