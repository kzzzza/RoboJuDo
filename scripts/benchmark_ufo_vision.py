"""Replay one clip through the real vision/socket path, without any robot process."""

import argparse
import json
import os
import socket
import sys
import threading
from pathlib import Path

os.environ.setdefault("PYAV_LOGGING", "off")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for variable, relative in (("XDG_CACHE_HOME", ".local/cache"), ("TORCH_HOME", ".local/cache/torch"),
                           ("HF_HOME", ".local/cache/huggingface"), ("CUDA_CACHE_PATH", ".local/cache/cuda"),
                           ("TMPDIR", ".local/setup")):
    (ROOT / relative).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(ROOT / relative))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vision-profile", choices=("legacy", "low-latency"), required=True)
    parser.add_argument("--hmr-backend", choices=("auto", "torch", "tensorrt"), default="auto")
    parser.add_argument("--detector-backend", choices=("auto", "torch", "tensorrt"), default="auto")
    parser.add_argument("--accel-dir", default=str(ROOT / ".local/vision_accel"))
    parser.add_argument("--packages", default=str(ROOT / ".local/rgb_bfm"))
    parser.add_argument("--variant", choices=("tcn", "mlp"), default="tcn")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    directory = Path(args.output_dir).resolve()
    if (directory / "metrics.json").exists() or (directory / "z.npy").exists():
        raise FileExistsError(f"Use a new benchmark output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / ".local/video_runtime"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime))
    from robojudo.video.latent_packet import decode_packet
    from robojudo.video.vision_worker import VisionWorker

    if args.vision_profile == "low-latency":
        from robojudo.video.vision_fast import FastVisionWorker

        worker = FastVisionWorker(args.packages, args.variant, args.device,
                                  cache_dir=args.accel_dir, hmr_backend=args.hmr_backend,
                                  detector_backend=args.detector_backend)
    else:
        worker = VisionWorker(Path(args.packages), args.variant, args.device)
    endpoint = directory / "vision.sock"
    if endpoint.exists():
        raise FileExistsError(f"Refusing to replace existing socket: {endpoint}")
    stop = threading.Event()
    received = []
    errors = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
        receiver.bind(str(endpoint))
        receiver.settimeout(.2)

        def consume():
            while not stop.is_set():
                try:
                    packet = decode_packet(receiver.recv(4096))
                    received.append(dict(sequence=packet.sequence, source_frame_id=packet.source_frame_id,
                                         captured_at=packet.capture_time, produced_at=packet.produced_time,
                                         eof=packet.eof, person_lost=packet.person_lost))
                except socket.timeout:
                    continue
                except Exception as exc:
                    errors.append(str(exc))
                    return

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        try:
            worker.run(video=args.video, camera=None, socket_path=str(endpoint),
                       metrics_path=str(directory / "metrics.json"), output_dir=str(directory),
                       live_replay=True)
        finally:
            stop.set()
            thread.join(timeout=1)
            endpoint.unlink(missing_ok=True)
    if errors:
        raise RuntimeError(f"Socket packet validation failed: {errors}")
    (directory / "packets.json").write_text(json.dumps(received, indent=2) + "\n")
    print((directory / "metrics.json").read_text(), flush=True)
    if not worker.latent_packets_sent:
        raise RuntimeError("No valid z was produced; check person visibility/orientation in the input")


if __name__ == "__main__":
    main()
