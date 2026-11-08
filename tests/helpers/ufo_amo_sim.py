"""Headless MuJoCo physics check with real AMO/UFO models; never loads DDS."""

import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from box import Box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robojudo.config.config_manager import ConfigManager
import robojudo.pipeline.ufo_amo_pipeline as module
from robojudo.utils.util_func import get_gravity_orientation


class Controls:
    def __init__(self):
        self.now = 1000.0
        self.buttons = {}
        self.axes = dict.fromkeys(module.AXES, 0.)
        self.controllers = {"LatentCtrl": SimpleNamespace(inst=SimpleNamespace(set_playing=lambda _: None))}

    def get_ctrl_data(self, data):
        return Box(UnitreeCtrl=dict(buttons=self.buttons, axes=self.axes, button_event=[]),
                   LatentCtrl=dict(valid=False, person_lost=False, ended=False, capture_time=None))

    def post_step_callback(self, data):
        pass

    def shutdown(self):
        pass


def run(output):
    cfg = ConfigManager("g1_ufo_video_camera_amo").get_cfg()
    cfg.env.render = False
    controls = Controls()
    with patch.object(module, "CtrlManager", return_value=controls):
        pipeline = module.UfoAmoPipeline(cfg, clock=lambda: controls.now)
    rows, durations = [], []

    def cycle(buttons=None):
        controls.buttons = buttons or {}
        began = time.monotonic()
        pipeline.step()
        durations.append(time.monotonic() - began)
        controls.now += .02
        gravity = get_gravity_orientation(pipeline.env.base_quat)
        height = float(pipeline.env.data.qpos[2])
        tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
        rows.append(dict(state=pipeline.real_state, height=height, tilt=tilt))
        assert not pipeline._fault_stop, pipeline.stop_reason
        assert height > .55 and tilt < .7, rows[-1]

    try:
        cycle({"A": True})
        for _ in range(152):
            cycle()
        assert pipeline.real_state == "PREPARED"
        cycle({"Y": True})
        for _ in range(28):
            cycle()
        assert pipeline.real_state == "AMO"
        start = pipeline.env.data.qpos[:2].copy()
        for _ in range(3000):
            cycle()
        drift = float(np.linalg.norm(pipeline.env.data.qpos[:2] - start))
        assert drift < .25, drift
        controls.axes["LeftY"] = .2
        for _ in range(250):
            cycle()
        walking_distance = float(np.linalg.norm(pipeline.env.data.qpos[:2] - start))
        assert walking_distance > .1, walking_distance
        controls.axes["LeftY"] = 0.
        for _ in range(250):
            cycle()
        report = dict(status="passed", robot_connected=False, sdk_loaded=False,
                      zero_velocity_seconds=60, drift_m=drift,
                      walking_seconds=5, walking_distance_m=walking_distance,
                      min_height_m=min(r["height"] for r in rows), max_tilt_rad=max(r["tilt"] for r in rows),
                      compute_p95_s=float(np.percentile(durations, 95)), compute_max_s=max(durations),
                      steps=len(rows), dual_policy=pipeline.metrics())
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "dual_policy"}, indent=2), flush=True)
    finally:
        pipeline.ctrl_manager.shutdown()
        pipeline.env.shutdown()


if __name__ == "__main__":
    run(sys.argv[1])
