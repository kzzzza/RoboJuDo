"""Open Azure Kinect RGB and MuJoCo viewer for a UFO video-control demo."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def _stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _last_lines(path: Path) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-20:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("original", "tcn", "mlp"), default="tcn")
    parser.add_argument("--with-amo", action="store_true", help="Sim keys A/Y/B, space=R2; arrows=walk, J/L=turn")
    parser.add_argument("--camera", type=int, default=0, help="Azure Kinect device index")
    parser.add_argument("--camera-backend", choices=("k4a", "opencv"), default="k4a")
    parser.add_argument("--packages", default=".local/rgb_bfm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-python", default=".venv/bin/python")
    parser.add_argument("--vision-python", default=".venv-video/bin/python")
    parser.add_argument("--output", default=".local/camera_demo")
    parser.add_argument("--max-steps", type=int, help="Optional finite MuJoCo run for diagnosis")
    parser.add_argument("--no-preview", action="store_true", help="Run without the camera window for diagnosis")
    parser.add_argument("--show-detection-box", action="store_true",
                        help="Draw the selected person detector box in Camera RGB")
    args = parser.parse_args()
    if args.show_detection_box and args.no_preview:
        parser.error("--show-detection-box requires the camera preview")

    if not args.no_preview and not os.environ.get("DISPLAY"):
        parser.error("Camera RGB and MuJoCo windows require a desktop DISPLAY")
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve() / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    output.mkdir(parents=True)
    socket_path = Path(f"/tmp/robojudo_ufo_camera_{os.getpid()}.sock")
    control_log = output / "control.log"
    vision_log = output / "vision.log"
    control_metrics = output / "control_metrics.json"
    vision_metrics = output / "vision_metrics.json"
    control_cmd = [str(root / args.control_python), str(root / "scripts/run_pipeline.py"),
                   "-c", "g1_ufo_video_camera_amo" if args.with_amo else "g1_ufo_video_camera",
                   "--latent-variant", args.variant,
                   "--latent-socket", str(socket_path), "--metrics-json", str(control_metrics)]
    if args.max_steps is not None:
        control_cmd += ["--max-steps", str(args.max_steps)]
    vision_cmd = [str(root / args.vision_python), str(root / "scripts/run_ufo_video_source.py"),
                  "--camera", str(args.camera), "--camera-backend", args.camera_backend,
                  "--variant", args.variant, "--packages", str((root / args.packages).resolve()),
                  "--device", args.device, "--socket", str(socket_path),
                  "--metrics-json", str(vision_metrics)]
    if not args.no_preview:
        vision_cmd.append("--show-camera")
    if args.show_detection_box:
        vision_cmd.append("--show-detection-box")

    control = vision = None
    try:
        with control_log.open("w") as control_file, vision_log.open("w") as vision_file:
            control = subprocess.Popen(control_cmd, cwd=root, stdout=control_file,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 30
            while not socket_path.exists():
                if control.poll() is not None:
                    raise RuntimeError(f"MuJoCo exited ({control.returncode}):\n{_last_lines(control_log)}")
                if time.monotonic() > deadline:
                    raise TimeoutError("MuJoCo did not create the latent socket within 30 seconds")
                time.sleep(0.05)
            vision = subprocess.Popen(vision_cmd, cwd=root, stdout=vision_file,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            print(f"Camera RGB + MuJoCo: {args.variant}; press q/Esc in Camera RGB or Ctrl+C to stop", flush=True)
            print(f"Logs: {output}", flush=True)
            while True:
                if vision.poll() is not None:
                    if vision.returncode:
                        raise RuntimeError(f"Vision exited ({vision.returncode}):\n{_last_lines(vision_log)}")
                    print("Camera preview closed", flush=True)
                    break
                if control.poll() is not None:
                    if control.returncode:
                        raise RuntimeError(f"MuJoCo exited ({control.returncode}):\n{_last_lines(control_log)}")
                    print("MuJoCo run completed", flush=True)
                    break
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping camera demo", flush=True)
    finally:
        _stop(vision)
        _stop(control)
        socket_path.unlink(missing_ok=True)
        if control_metrics.exists():
            metrics = json.loads(control_metrics.read_text())
            print(f"MuJoCo policy steps: {metrics['policy_steps']}; "
                  f"unique camera frames used: {metrics['unique_latent_frames']}", flush=True)


if __name__ == "__main__":
    main()
