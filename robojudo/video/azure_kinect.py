"""RGB frames from Azure Kinect DK through the installed libk4a runtime."""

import ctypes
import time
from ctypes import POINTER, byref, c_bool, c_int, c_int32, c_uint32, c_uint8, c_void_p

import numpy as np


class _DeviceConfig(ctypes.Structure):
    _fields_ = [
        ("color_format", c_int),
        ("color_resolution", c_int),
        ("depth_mode", c_int),
        ("camera_fps", c_int),
        ("synchronized_images_only", c_bool),
        ("depth_delay_off_color_usec", c_int32),
        ("wired_sync_mode", c_int),
        ("subordinate_delay_off_master_usec", c_uint32),
        ("disable_streaming_indicator", c_bool),
    ]


class AzureKinectRGB:
    """Read the color stream using the same 720p/30 Hz mode as deploy/camera."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.sdk = ctypes.CDLL("libk4a.so")
        sdk = self.sdk
        sdk.k4a_device_get_installed_count.restype = c_uint32
        sdk.k4a_device_open.argtypes = [c_uint32, POINTER(c_void_p)]
        sdk.k4a_device_open.restype = c_int
        sdk.k4a_device_start_cameras.argtypes = [c_void_p, POINTER(_DeviceConfig)]
        sdk.k4a_device_start_cameras.restype = c_int
        sdk.k4a_device_get_capture.argtypes = [c_void_p, POINTER(c_void_p), c_int32]
        sdk.k4a_device_get_capture.restype = c_int
        sdk.k4a_capture_get_color_image.argtypes = [c_void_p]
        sdk.k4a_capture_get_color_image.restype = c_void_p
        sdk.k4a_image_get_width_pixels.argtypes = [c_void_p]
        sdk.k4a_image_get_width_pixels.restype = c_int
        sdk.k4a_image_get_height_pixels.argtypes = [c_void_p]
        sdk.k4a_image_get_height_pixels.restype = c_int
        sdk.k4a_image_get_stride_bytes.argtypes = [c_void_p]
        sdk.k4a_image_get_stride_bytes.restype = c_int
        sdk.k4a_image_get_buffer.argtypes = [c_void_p]
        sdk.k4a_image_get_buffer.restype = c_void_p
        sdk.k4a_image_release.argtypes = [c_void_p]
        sdk.k4a_capture_release.argtypes = [c_void_p]
        sdk.k4a_device_stop_cameras.argtypes = [c_void_p]
        sdk.k4a_device_close.argtypes = [c_void_p]
        self.device = c_void_p()
        self.started = False

    def __enter__(self):
        sdk = self.sdk
        count = sdk.k4a_device_get_installed_count()
        if self.device_index >= count:
            raise RuntimeError(f"Azure Kinect {self.device_index} unavailable; found {count} device(s)")
        if sdk.k4a_device_open(self.device_index, byref(self.device)) != 0:
            raise RuntimeError(f"Cannot open Azure Kinect {self.device_index}")
        config = _DeviceConfig()
        config.color_format = 3  # K4A_IMAGE_FORMAT_COLOR_BGRA32
        config.color_resolution = 1  # K4A_COLOR_RESOLUTION_720P
        config.depth_mode = 2  # K4A_DEPTH_MODE_NFOV_UNBINNED; proven camera configuration
        config.camera_fps = 2  # K4A_FRAMES_PER_SECOND_30
        try:
            if sdk.k4a_device_start_cameras(self.device, byref(config)) != 0:
                raise RuntimeError("Cannot start Azure Kinect; another process may own the camera")
            self.started = True
        except BaseException:
            self.close()
            raise
        return self

    def read(self) -> tuple[np.ndarray, float]:
        import cv2

        sdk = self.sdk
        for _ in range(5):
            capture = c_void_p()
            result = sdk.k4a_device_get_capture(self.device, byref(capture), 1000)
            if result != 0:
                continue
            captured_at = time.monotonic()
            try:
                color = sdk.k4a_capture_get_color_image(capture)
                if not color:  # The first few captures may have no RGB image.
                    continue
                try:
                    width = sdk.k4a_image_get_width_pixels(color)
                    height = sdk.k4a_image_get_height_pixels(color)
                    stride = sdk.k4a_image_get_stride_bytes(color)
                    buffer = sdk.k4a_image_get_buffer(color)
                    if width <= 0 or height <= 0 or stride < width * 4 or not buffer:
                        raise RuntimeError("Invalid Azure Kinect RGB buffer")
                    pixels = np.ctypeslib.as_array(
                        (c_uint8 * (height * stride)).from_address(buffer)
                    ).reshape(height, stride)[:, :width * 4].reshape(height, width, 4).copy()
                finally:
                    sdk.k4a_image_release(color)
            finally:
                sdk.k4a_capture_release(capture)
            return cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB), captured_at
        raise RuntimeError("Azure Kinect did not produce an RGB frame")

    def close(self):
        if self.device:
            if self.started:
                self.sdk.k4a_device_stop_cameras(self.device)
                self.started = False
            self.sdk.k4a_device_close(self.device)
            self.device = c_void_p()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
