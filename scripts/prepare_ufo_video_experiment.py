"""Prepare exact full-video z for the original, TCN and MLP UFO experiments."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(path, variant):
    z = np.load(path / "z.npy", allow_pickle=False)
    timestamps = np.load(path / "timestamps.npy", allow_pickle=False)
    if (z.dtype != np.float32 or z.ndim != 2 or z.shape[1] != 256 or
            timestamps.shape != (len(z),) or not np.isfinite(z).all() or
            not np.allclose(np.linalg.norm(z, axis=1), 16, atol=1e-3)):
        raise ValueError(f"Invalid {variant} output")
    return z, timestamps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(Path.home() / "Videos/dance_bilbil.mp4"))
    parser.add_argument("--packages", default=".local/rgb_bfm")
    parser.add_argument("--output", default=".local/video_runs/dance_bilbil")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    video = Path(args.video).expanduser().resolve()
    packages = Path(args.packages).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    output.mkdir(parents=True, exist_ok=True)
    wall_times = {}
    for variant in ("original", "tcn", "mlp"):
        target = output / variant
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Experiment output must be empty: {target}")
        command = [sys.executable, str(packages / f"{variant}_pipeline/pipeline.py")]
        if variant == "original":
            command += ["--video", str(video)]
        else:
            command += ["--u", str(output / "original/u.npy"),
                        "--timestamps", str(output / "original/timestamps.npy")]
        command += ["--output", str(target), "--device", args.device]
        began = time.monotonic()
        subprocess.run(command, check=True)
        wall_times[variant] = time.monotonic() - began
    reference, times = _check(output / "original", "original")
    report = {"video": str(video), "video_sha256": _sha256(video),
              "frames": len(times), "duration_s": float(times[-1]),
              "device": args.device, "wall_seconds": wall_times, "variants": {}}
    for variant in ("original", "tcn", "mlp"):
        z, candidate_times = _check(output / variant, variant)
        if not np.array_equal(times, candidate_times):
            raise ValueError(f"{variant} timestamps differ")
        cosine = np.sum(reference * z, axis=1) / (np.linalg.norm(reference, axis=1) * np.linalg.norm(z, axis=1))
        report["variants"][variant] = {
            "z_sha256": _sha256(output / variant / "z.npy"),
            "mean_cosine_to_original": float(np.mean(cosine)),
            "mse_to_original": float(np.mean((reference - z) ** 2)),
        }
        (output / variant / "latent_metadata.json").write_text(json.dumps({
            "variant": variant, "video_sha256": report["video_sha256"],
            "z_sha256": report["variants"][variant]["z_sha256"]}, indent=2) + "\n")
    (output / "offline_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
