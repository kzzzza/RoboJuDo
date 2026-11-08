"""Run all UFO video models locally and send guarded HG joint targets directly to G1."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "g1_ufo_video_real_workstation"


def stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def checked(command, log, environment):
    with log.open("a") as output:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Preflight failed ({result.returncode}):\n{log.read_text(errors='replace')[-4000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net-if", required=True, help="Workstation Ethernet/Wi-Fi interface, not Jetson's eth0")
    parser.add_argument("--with-amo", action="store_true", help="A: pose, Y: AMO walking/balance, B: UFO")
    parser.add_argument("--variant", choices=("tcn", "mlp", "original"), default="tcn")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("opencv", "k4a"), default="opencv")
    parser.add_argument("--camera-rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--packages", default=".local/rgb_bfm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vision-profile", choices=("auto", "legacy", "low-latency"), default="auto")
    parser.add_argument("--hmr-backend", choices=("auto", "torch", "tensorrt"), default="auto")
    parser.add_argument("--detector-backend", choices=("auto", "torch", "tensorrt"), default="auto")
    parser.add_argument("--accel-dir", default=".local/vision_accel")
    parser.add_argument("--control-python", default=".venv/bin/python")
    parser.add_argument("--vision-python", default=".venv-video/bin/python")
    parser.add_argument("--output", default=".local/workstation_runs")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--show-detection-box", action="store_true")
    args = parser.parse_args()
    config = CONFIG + "_amo" if args.with_amo else CONFIG
    if os.environ.get("UFO_REAL_ROBOT_OK") != "1":
        parser.error("Real control requires UFO_REAL_ROBOT_OK=1; use check_ufo_m0_real.py for read-only checks")
    if args.no_preview and args.show_detection_box:
        parser.error("--show-detection-box requires the RGB preview")
    if not args.no_preview and not os.environ.get("DISPLAY"):
        parser.error("RGB preview requires DISPLAY; use --no-preview for a headless workstation")
    packages = (ROOT / args.packages).resolve()
    # Keep the venv executable path (resolving its symlink would bypass the venv).
    control_python = os.path.abspath(ROOT / args.control_python)
    vision_python = os.path.abspath(ROOT / args.vision_python)
    for executable in (control_python, vision_python):
        if not Path(executable).is_file():
            parser.error(f"Missing interpreter: {executable}")
    output = (ROOT / args.output).resolve() / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    output.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(ROOT) + os.pathsep + environment.get("PYTHONPATH", ""),
                        "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "XDG_CACHE_HOME": str(ROOT / ".local/cache"),
                        "TORCH_HOME": str(ROOT / ".local/cache/torch"),
                        "CUDA_CACHE_PATH": str(ROOT / ".local/cache/cuda"),
                        "TMPDIR": str(ROOT / ".local/setup"),
                        "HF_HOME": str(ROOT / ".local/cache/huggingface"),
                        "YOLO_CONFIG_DIR": str(ROOT / ".local/video_runtime"),
                        "MPLCONFIGDIR": str(ROOT / ".local/video_runtime"),
                        "PYAV_LOGGING": "off", "YOLO_AUTOINSTALL": "false"})
    socket_path = Path(f"/tmp/robojudo_workstation_{os.getpid()}.sock")
    preflight_log = output / "preflight.log"
    control_log = output / "control.log"
    vision_log = output / "vision.log"
    control = vision = None
    report = {"config": config, "variant": args.variant, "net_if": args.net_if,
              "status": "starting", "robot_models_deployed": False}
    try:
        # Verify all required files/dependencies before releasing built-in motion control.
        for variant in dict.fromkeys(("original", args.variant)):
            checked([vision_python, str(packages / f"{variant}_pipeline/pipeline.py"), "--verify"],
                    preflight_log, environment)
        checked([vision_python, "-c", "import torch, torchvision, cv2, timm, av, ultralytics, lap; "
                 + ("assert torch.cuda.is_available(), 'CUDA unavailable'" if args.device.startswith("cuda") else "")],
                preflight_log, environment)
        checked([vision_python, str(ROOT / "scripts/run_ufo_video_source.py"),
                 "--camera", str(args.camera), "--variant", args.variant, "--packages", str(packages),
                 "--device", args.device, "--vision-profile", args.vision_profile,
                 "--hmr-backend", args.hmr_backend, "--detector-backend", args.detector_backend,
                 "--accel-dir", str(ROOT / args.accel_dir), "--check-runtime"], preflight_log, environment)
        checked([control_python, str(ROOT / "scripts/check_ufo_m0_real.py"), "--config", config,
                 "--net-if", args.net_if, "--latent-variant", args.variant, "--read-only"],
                preflight_log, environment)
        with control_log.open("w") as control_file, vision_log.open("w") as vision_file:
            control = subprocess.Popen([
                control_python, str(ROOT / "scripts/run_pipeline.py"), "-c", config,
                "--net-if", args.net_if, "--latent-variant", args.variant,
                "--latent-socket", str(socket_path), "--metrics-json", str(output / "control_metrics.json")],
                cwd=ROOT, env=environment, stdout=control_file, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 40
            while not socket_path.exists():
                if control.poll() is not None:
                    raise RuntimeError(f"Control startup failed:\n{control_log.read_text(errors='replace')[-4000:]}")
                if time.monotonic() > deadline:
                    raise TimeoutError("G1 control did not become ready within 40 seconds")
                time.sleep(0.05)
            command = [vision_python, str(ROOT / "scripts/run_ufo_video_source.py"),
                       "--camera", str(args.camera), "--camera-backend", args.camera_backend,
                       "--variant", args.variant, "--packages", str(packages), "--device", args.device,
                       "--socket", str(socket_path), "--metrics-json", str(output / "vision_metrics.json")]
            command += ["--camera-rotate", str(args.camera_rotate),
                        "--vision-profile", args.vision_profile, "--hmr-backend", args.hmr_backend,
                        "--detector-backend", args.detector_backend,
                        "--accel-dir", str(ROOT / args.accel_dir)]
            if not args.no_preview:
                command.append("--show-camera")
            if args.show_detection_box:
                command.append("--show-detection-box")
            vision = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=vision_file,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            print(f"Workstation {args.variant} -> G1 DDS on {args.net_if}; A: pose, "
                  + ("Y: AMO, " if args.with_amo else "") + "B: track, R2: stop", flush=True)
            print(f"Logs: {output}", flush=True)
            report["status"] = "running"
            vision_failed = False
            while control.poll() is None:
                vision_code = vision.poll()
                if vision_code is not None:
                    if not args.with_amo or vision_code == 0:
                        break
                    if not vision_failed:
                        vision_failed = True
                        report.update(vision_failed=True, vision_failure_exit=vision_code)
                        print(f"Vision failed ({vision_code}); keeping AMO control alive. "
                              "UFO returns to AMO at the 5s visual-age limit. Ctrl+C exits.\n"
                              + vision_log.read_text(errors="replace")[-2000:], flush=True)
                time.sleep(0.1)
            for name, process, log in (("control", control, control_log), ("vision", vision, vision_log)):
                if process.poll() not in (None, 0) and not (name == "vision" and vision_failed):
                    raise RuntimeError(f"{name} exited ({process.returncode}):\n{log.read_text(errors='replace')[-4000:]}")
            report["status"] = "stopped"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        # Let the control process hold/close while the local visual source is still alive.
        try:
            stop(control)
        finally:
            stop(vision)
            socket_path.unlink(missing_ok=True)
            report.update(control_exit=None if control is None else control.returncode,
                          vision_exit=None if vision is None else vision.returncode)
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
