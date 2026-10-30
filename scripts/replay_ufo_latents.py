"""Diagnostic: replay saved z over the same socket without running RGB inference."""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--variant", choices=("original", "tcn", "mlp"), required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--metrics-json")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robojudo/video"))
    from latent_packet import encode_packet

    directory = Path(args.latent_dir)
    z = np.load(directory / "z.npy", allow_pickle=False)
    times = np.load(directory / "timestamps.npy", allow_pickle=False)
    if (z.dtype != np.float32 or z.ndim != 2 or z.shape[1] != 256 or len(z) == 0
            or times.shape != (len(z),) or not np.isfinite(z).all()
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or not np.allclose(np.linalg.norm(z, axis=1), 16, atol=1e-3)):
        raise ValueError("Invalid saved latent sequence")
    delays = []
    start = time.monotonic()
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        for index, (latent, timestamp) in enumerate(zip(z, times)):
            due = start + float(timestamp - times[0])
            time.sleep(max(0.0, due - time.monotonic()))
            sock.sendto(encode_packet(args.variant, index, due, latent, eof=index == len(z) - 1), args.socket)
            delays.append(time.monotonic() - due)
    if args.metrics_json:
        Path(args.metrics_json).write_text(json.dumps({
            "mode": "precomputed_latent_transport", "variant": args.variant,
            "frames_read": len(z), "latents_emitted": len(z), "socket_send_failures": 0,
            "frames_dropped": 0, "capture_to_send_p50_s": float(np.percentile(delays, 50)),
            "capture_to_send_p95_s": float(np.percentile(delays, 95))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
