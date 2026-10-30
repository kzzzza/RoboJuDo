"""Run the isolated RGB-BFM worker, feeding one shared LatentCtrl socket."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="30 FPS MP4 replayed using presentation timestamps")
    source.add_argument("--camera", type=int, help="Camera index")
    parser.add_argument("--camera-backend", choices=("opencv", "k4a"), default="opencv")
    parser.add_argument("--show-camera", action="store_true", help="Open an RGB preview; q/Esc closes it")
    parser.add_argument("--show-detection-box", action="store_true",
                        help="Draw the selected person detector box in the RGB preview")
    parser.add_argument("--variant", choices=("original", "tcn", "mlp"), required=True)
    parser.add_argument("--packages", default=".local/rgb_bfm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--socket", default="/tmp/robojudo_ufo_latent.sock")
    parser.add_argument("--metrics-json")
    parser.add_argument("--output-dir", help="Save emitted streaming z and frame indices for comparison")
    args = parser.parse_args()
    if args.show_detection_box and (args.camera is None or not args.show_camera):
        parser.error("--show-detection-box requires --camera and --show-camera")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    runtime = Path(".local/video_runtime").resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robojudo/video"))
    from vision_worker import VisionWorker

    worker = VisionWorker(Path(args.packages), args.variant, args.device)
    count = worker.run(video=args.video, camera=args.camera, socket_path=args.socket,
                       metrics_path=args.metrics_json, output_dir=args.output_dir,
                       camera_backend=args.camera_backend, show_camera=args.show_camera,
                       show_detection_box=args.show_detection_box)
    print(f"Sent {count} {args.variant} latents")


if __name__ == "__main__":
    main()
