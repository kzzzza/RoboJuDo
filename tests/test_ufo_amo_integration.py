"""Subprocess isolation prevents accidental reuse of a robot DDS participant."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_helper(name, output, timeout):
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    environment.pop("UFO_REAL_ROBOT_OK", None)
    result = subprocess.run([sys.executable, str(root / "tests/helpers" / name), str(output)],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
    return json.loads(output.read_text())


def test_headless_amo_60s_stand_and_walk(tmp_path):
    report = run_helper("ufo_amo_sim.py", tmp_path / "simulation.json", 60)
    assert report["status"] == "passed" and not report["robot_connected"]
    assert report["zero_velocity_seconds"] == 60
    assert report["drift_m"] < .25 and report["walking_distance_m"] > .1


@pytest.mark.skipif(importlib.util.find_spec("unitree_sdk2py") is None, reason="Install workstation SDK")
def test_isolated_dds_amo_ufo_switches(tmp_path):
    report = run_helper("ufo_amo_dds_loopback.py", tmp_path / "dds.json", 90)
    assert report["status"] == "passed" and not report["robot_connected"]
    assert report["domain"] == 121 and report["interface"] == "lo"
    assert report["peer"]["bad_crc"] == report["peer"]["bad_fields"] == 0
    assert report["dual_policy"]["fault_stopped"]
