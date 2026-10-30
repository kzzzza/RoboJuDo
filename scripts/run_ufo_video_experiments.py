"""Run all three paced-video visual workers against the same G1 MuJoCo controller."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(Path.home() / "Videos/dance_bilbil.mp4"))
    parser.add_argument("--offline", default=".local/video_runs/dance_bilbil")
    parser.add_argument("--output", default=".local/video_runs/dance_bilbil/stream")
    parser.add_argument("--packages", default=".local/rgb_bfm")
    parser.add_argument("--control-python", default=".venv/bin/python")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--mode", choices=("vision", "transport"), default="vision",
                        help="transport replays saved z only, for controller diagnostics")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for variant in ("original", "tcn", "mlp"):
        target = output / variant
        target.mkdir()
        socket_path = f"/tmp/robojudo_ufo_video_{os.getpid()}_{variant}.sock"
        control_log = (target / "control.log").open("w")
        worker_log = (target / "worker.log").open("w")
        control_cmd = [str(root / args.control_python), str(root / "scripts/run_pipeline.py"),
                       "-c", "g1_ufo_video_stream", "--latent-variant", variant,
                       "--latent-socket", socket_path, "--metrics-json", str(target / "control.json")]
        if args.record_video:
            control_cmd += ["--record-video", str(target / "simulation.mp4")]
        if args.mode == "vision":
            worker_cmd = [sys.executable, str(root / "scripts/run_ufo_video_source.py"),
                          "--video", str(Path(args.video).resolve()), "--variant", variant,
                          "--packages", str(Path(args.packages).resolve()), "--device", args.device,
                          "--socket", socket_path, "--metrics-json", str(target / "worker.json"),
                          "--output-dir", str(target / "latents")]
        else:
            worker_cmd = [sys.executable, str(root / "scripts/replay_ufo_latents.py"),
                          "--latent-dir", str(Path(args.offline) / variant), "--variant", variant,
                          "--socket", socket_path, "--metrics-json", str(target / "worker.json")]
        started = time.monotonic()
        control = subprocess.Popen(control_cmd, cwd=root, stdout=control_log,
                                   stderr=subprocess.STDOUT)
        worker = None
        try:
            deadline = time.monotonic() + 30
            while not Path(socket_path).exists():
                if control.poll() is not None:
                    raise RuntimeError(f"{variant} control exited before socket bind")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{variant} control socket did not appear")
                time.sleep(0.05)
            worker = subprocess.Popen(worker_cmd, cwd=root, stdout=worker_log,
                                      stderr=subprocess.STDOUT)
            worker_code = worker.wait()
            control_code = control.wait(timeout=20) if worker_code == 0 else None
            report[variant] = {"worker_exit": worker_code, "control_exit": control_code,
                               "wall_seconds": time.monotonic() - started}
            if (target / "control.json").exists():
                control_stats = json.loads((target / "control.json").read_text())
                report[variant]["control"] = {k: v for k, v in control_stats.items() if k != "samples"}
            if (target / "worker.json").exists():
                report[variant]["worker"] = json.loads((target / "worker.json").read_text())
            if "worker" in report[variant] and "control" in report[variant]:
                report[variant]["latents_not_used_by_control"] = max(
                    0, report[variant]["worker"]["latents_emitted"] -
                    report[variant]["control"]["unique_latent_frames"])
            if (target / "latents/z.npy").exists():
                candidate = np.load(target / "latents/z.npy", allow_pickle=False)
                ids = np.load(target / "latents/frame_id.npy", allow_pickle=False)
                reference = np.load(Path(args.offline) / variant / "z.npy", allow_pickle=False)
                aligned = ids < len(reference)
                report[variant]["stream_vs_offline_mse"] = float(np.mean((candidate[aligned] - reference[ids[aligned]]) ** 2))
                report[variant]["stream_vs_offline_cosine"] = float(np.mean(np.sum(candidate[aligned] * reference[ids[aligned]], axis=1) / 256))
        except Exception as exc:
            report[variant] = {"error": str(exc), "wall_seconds": time.monotonic() - started}
        finally:
            if worker is not None and worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=10)
            if control.poll() is None:
                control.terminate()
                control.wait(timeout=10)
            Path(socket_path).unlink(missing_ok=True)
            control_log.close()
            worker_log.close()
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
