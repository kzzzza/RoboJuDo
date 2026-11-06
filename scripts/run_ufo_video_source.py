"""Run RGB-BFM inference and feed LatentCtrl locally or over an SSH tunnel."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="MP4 replayed using presentation timestamps and sampled to 30 Hz")
    source.add_argument("--camera", type=int, help="Camera index")
    parser.add_argument("--camera-backend", choices=("opencv", "k4a"), default="opencv")
    parser.add_argument("--show-camera", action="store_true", help="Open an RGB preview; q/Esc closes it")
    parser.add_argument("--show-detection-box", action="store_true",
                        help="Draw the selected person detector box in the RGB preview")
    parser.add_argument("--variant", choices=("original", "tcn", "mlp"), required=True)
    parser.add_argument("--packages", default=".local/rgb_bfm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--socket", default="/tmp/robojudo_ufo_latent.sock")
    parser.add_argument("--ssh-target", help="Jetson SSH destination, e.g. user@robot; forward z over SSH")
    parser.add_argument("--tcp-port", type=int, default=5541, help="Loopback port on both SSH endpoints")
    parser.add_argument("--metrics-json")
    parser.add_argument("--output-dir", help="Save emitted streaming z and frame indices for comparison")
    args = parser.parse_args()
    if args.show_detection_box and (args.camera is None or not args.show_camera):
        parser.error("--show-detection-box requires --camera and --show-camera")
    if args.ssh_target and args.ssh_target.startswith("-"):
        parser.error("--ssh-target must be a host or user@host")
    if not 1 <= args.tcp_port <= 65535:
        parser.error("--tcp-port must be between 1 and 65535")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    # FFmpeg callbacks into Python can deadlock native camera/decoder threads.
    # Keep FFmpeg's native stderr logger in this mixed OpenCV/PyAV process.
    os.environ.setdefault("PYAV_LOGGING", "off")
    runtime = Path(".local/video_runtime").resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robojudo/video"))
    from vision_worker import VisionWorker

    tunnel = None
    preview = None
    try:
        if args.show_camera:
            from vision_source import CameraPreview

            # timm/torchvision can import PyAV, whose bundled FFmpeg conflicts
            # with Qt's first window initialization on some Linux desktops.
            # Create HighGUI before loading those model dependencies.
            preview = CameraPreview(show_detection_box=args.show_detection_box)
        if args.ssh_target:
            tunnel = subprocess.Popen([
                "ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                "-L", f"127.0.0.1:{args.tcp_port}:127.0.0.1:{args.tcp_port}",
                args.ssh_target,
            ])
            time.sleep(0.5)
            if tunnel.poll() is not None:
                raise RuntimeError(f"SSH tunnel exited with status {tunnel.returncode}")

        def check_tunnel():
            if tunnel is not None and tunnel.poll() is not None:
                raise RuntimeError(f"SSH tunnel exited with status {tunnel.returncode}")

        worker = VisionWorker(Path(args.packages), args.variant, args.device)
        count = worker.run(video=args.video, camera=args.camera, socket_path=args.socket,
                           metrics_path=args.metrics_json, output_dir=args.output_dir,
                           camera_backend=args.camera_backend, show_camera=args.show_camera,
                           show_detection_box=args.show_detection_box,
                           preview=preview,
                           tcp_host="127.0.0.1" if tunnel is not None else None,
                           tcp_port=args.tcp_port, healthcheck=check_tunnel if tunnel is not None else None)
    finally:
        if preview is not None:
            preview.close()
        if tunnel is not None and tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=3)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait()
    print(f"Sent {count} {args.variant} latents")


if __name__ == "__main__":
    main()
