"""Two real DDS participants exchange official HG messages over loopback only."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(importlib.util.find_spec("unitree_sdk2py") is None, reason="Install workstation SDK")
def test_official_dds_workstation_closed_loop(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "loopback.json"
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, str(root / "tests/helpers/workstation_dds_loopback.py"), str(output)],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout[-5000:] + result.stderr[-5000:]
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["interface"] == "lo" and report["domain"] == 121
    assert report["peer"]["bad_crc"] == report["peer"]["bad_fields"] == 0
    assert report["healthy_tracking"]["cycles"] == 1000
    assert report["healthy_tracking"]["compute_p95_s"] < 0.02
