"""Timestamped 30 Hz frames for the separate UFO vision process."""

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    index: int
    rgb: object
    captured_at: float  # monotonic clock shared with the local control process


def replay_video(path: str | Path):
    """Replay presentation timestamps, preserving lateness for latency measurement."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        if abs(rate - 30.0) > 0.02:
            raise ValueError(f"Stream replay needs a 30 FPS input; got {rate:.3f}")
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
            yield Frame(index, decoded.to_ndarray(format="rgb24"), due)


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
        while True:
            rgb, captured_at = read()
            yield Frame(index, rgb, captured_at)
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
