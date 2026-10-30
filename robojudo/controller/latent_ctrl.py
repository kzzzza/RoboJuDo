"""Nonblocking 256-D reference source for UFO and future latent policies."""

import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import LatentCtrlCfg
from robojudo.video.latent_packet import PACKET_BYTES, decode_packet


@ctrl_registry.register
class LatentCtrl(Controller):
    cfg_ctrl: LatentCtrlCfg

    def __init__(self, cfg_ctrl: LatentCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl, env, device)
        self.socket = None
        self.z = None
        self.frame_id = -1
        self.capture_time = None
        self.received_time = None
        self.ended = False
        self.playing = bool(env is not None and env.cfg_env.is_sim)
        self.play_time = 0.0
        if cfg_ctrl.source == "file":
            root = Path(cfg_ctrl.z_dir).expanduser()
            metadata = root / "latent_metadata.json"
            if metadata.exists() and json.loads(metadata.read_text()).get("variant") != cfg_ctrl.variant:
                raise ValueError(f"Latent variant mismatch in {metadata}")
            self.zs = np.load(root / "z.npy", allow_pickle=False)
            self.timestamps = np.load(root / "timestamps.npy", allow_pickle=False)
            if (self.zs.dtype != np.float32 or self.zs.ndim != 2 or self.zs.shape[1] != 256
                    or len(self.zs) == 0 or self.timestamps.shape != (len(self.zs),)
                    or not np.all(np.isfinite(self.zs)) or not np.all(np.isfinite(self.timestamps))
                    or np.any(np.diff(self.timestamps) <= 0)):
                raise ValueError("Invalid z.npy / timestamps.npy")
            if not np.allclose(np.linalg.norm(self.zs, axis=1), 16.0, atol=1e-3):
                raise ValueError("Unexpected UFO latent norm")
        else:
            self.zs = None
            self.timestamps = None
            path = Path(cfg_ctrl.socket_path)
            if path.exists():
                raise FileExistsError(f"Latent socket already exists: {path}")
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.socket.bind(str(path))
            os.chmod(path, 0o600)
            self.socket.setblocking(False)
        self.reset()

    def reset(self):
        self.play_time = 0.0
        self.ended = False
        if self.zs is not None:
            self.frame_id = 0
            self.z = self.zs[0].copy()
        else:
            self.frame_id = -1
            self.z = None
            self.capture_time = None
            self.received_time = None

    def set_playing(self, enabled: bool):
        self.playing = bool(enabled)

    def _poll_socket(self):
        while True:
            try:
                packet = self.socket.recv(PACKET_BYTES + 1)
            except BlockingIOError:
                break
            try:
                variant, sequence, capture_time, z, eof = decode_packet(packet)
            except ValueError:
                continue
            if variant != self.cfg_ctrl.variant or sequence <= self.frame_id:
                continue
            if not np.isclose(np.linalg.norm(z), 16.0, atol=1e-3):
                continue
            self.z = z
            self.frame_id = sequence
            self.capture_time = capture_time
            self.received_time = time.monotonic()
            self.ended = eof

    def get_data(self):
        if self.zs is not None:
            index = int(np.searchsorted(self.timestamps, self.play_time, side="right") - 1)
            self.frame_id = max(0, min(index, len(self.zs) - 1))
            self.z = self.zs[self.frame_id].copy()
            self.ended = self.play_time > float(self.timestamps[-1]) + self.cfg_ctrl.control_dt * 0.5
            return {"z": self.z, "valid": True, "ended": self.ended,
                    "frame_id": self.frame_id, "age": 0.0}
        self._poll_socket()
        now = time.monotonic()
        age = float("inf") if self.capture_time is None else max(0.0, now - self.capture_time)
        idle = float("inf") if self.received_time is None else now - self.received_time
        valid = self.z is not None and age <= self.cfg_ctrl.max_age_seconds and idle <= self.cfg_ctrl.stale_seconds
        return {"z": self.z.copy() if valid else None, "valid": valid, "ended": self.ended,
                "frame_id": self.frame_id, "age": age}

    def post_step_callback(self, commands=None):
        if self.zs is not None and self.playing and not self.ended:
            self.play_time += self.cfg_ctrl.control_dt

    def shutdown(self):
        if self.socket is not None:
            self.socket.close()
            Path(self.cfg_ctrl.socket_path).unlink(missing_ok=True)
