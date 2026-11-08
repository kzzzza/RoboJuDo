"""Real policies/HG types over domain 121/lo only; no physical dynamics."""

import json
import multiprocessing
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from workstation_dds_loopback import DOMAIN, ROOT, configure_loopback, peer

sys.path.insert(0, str(ROOT))
from robojudo.config.config_manager import ConfigManager
from robojudo.environment.utils.g1_dds import G1DdsError
from robojudo.pipeline.ufo_amo_pipeline import UfoAmoPipeline
from robojudo.video.latent_packet import encode_packet


def run(output):
    context = multiprocessing.get_context("spawn")
    inbox, outbox = context.Queue(), context.Queue()
    process = context.Process(target=peer, args=(inbox, outbox))
    process.start()
    pipeline = None
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        assert outbox.get(timeout=10)["ready"]
        configure_loopback()
        with tempfile.TemporaryDirectory(prefix="amo-dds-") as directory:
            cfg = ConfigManager("g1_ufo_video_real_workstation_amo").get_cfg()
            cfg.env.dds_domain_id = DOMAIN
            cfg.env.unitree.net_if = "lo"
            cfg.auto_debug_mode = False
            cfg.ctrl[-1].variant = "tcn"
            cfg.ctrl[-1].socket_path = str(Path(directory) / "z.sock")
            os.environ["UFO_REAL_ROBOT_OK"] = "1"
            with patch("robojudo.environment.utils.g1_motion_mode.ensure_debug_mode") as release:
                pipeline = UfoAmoPipeline(cfg)
                release.assert_not_called()
            frame = 0
            z = np.zeros(256, np.float32)
            z[0] = 16

            def status():
                inbox.put(dict(op="status"))
                return outbox.get(timeout=5)

            def keys(mask):
                inbox.put(dict(op="keys", value=mask))
                time.sleep(.025)

            def cycle(lost=False, visual=True):
                nonlocal frame
                began = time.monotonic()
                if visual:
                    frame += 1
                    sender.sendto(encode_packet("tcn", frame, time.monotonic(), np.zeros_like(z) if lost else z, session_id=43,
                                                person_lost=lost), cfg.ctrl[-1].socket_path)
                pipeline.step()
                time.sleep(max(0, .02 - (time.monotonic() - began)))

            def wait_state(state, limit=5):
                until = time.monotonic() + limit
                while pipeline.real_state != state:
                    assert time.monotonic() < until, pipeline.real_state
                    cycle()

            assert status()["commands"] == 0
            for _ in range(5):
                cycle()
            assert status()["commands"] == 0
            keys(1 << 8)
            cycle()
            keys(0)
            wait_state("PREPARED")
            keys(1 << 9)
            cycle()
            assert pipeline.real_state == "PREPARED", "B cannot skip Y"
            keys(0)
            cycle()
            keys(1 << 11)
            cycle()
            keys(0)
            wait_state("AMO")
            keys(1 << 9)
            cycle()
            keys(0)
            wait_state("TRACKING")
            began = time.monotonic()
            for _ in range(1000):
                cycle()
            elapsed = time.monotonic() - began
            cycle(lost=True)
            wait_state("AMO")
            for _ in range(5):
                cycle()
            assert pipeline.real_state == "AMO"
            keys(1 << 9)
            cycle()
            keys(0)
            wait_state("TRACKING")
            # Actual visual packet absence (not person_lost) must fall back at 5s.
            for _ in range(260):
                cycle(visual=False)
            deadline = time.monotonic() + 2
            while pipeline.real_state != "AMO":
                assert time.monotonic() < deadline, pipeline.real_state
                cycle(visual=False)
            keys(1 << 4)
            before = pipeline.counts["UFO_inferences"]
            cycle()
            assert pipeline.real_state == "STOPPED" and not pipeline._fault_stop
            assert pipeline.counts["UFO_inferences"] == before
            keys(0)
            cycle()
            keys(1 << 9)
            cycle()
            assert pipeline.real_state == "STOPPED"
            keys(0)
            cycle()
            keys(1 << 8)
            cycle()
            keys(0)
            wait_state("PREPARED")
            keys(1 << 11)
            cycle()
            keys(0)
            wait_state("AMO")
            inbox.put(dict(op="pause", value=True))
            time.sleep(.16)
            before = status()["commands"]
            try:
                cycle()
            except G1DdsError:
                pass
            else:
                raise AssertionError("Stale feedback must stop sending")
            assert pipeline._fault_stop
            assert status()["commands"] == before
            inbox.put(dict(op="pause", value=False))
            time.sleep(.05)
            keys(1 << 8)
            cycle()
            assert pipeline.real_state == "STOPPED"
            final = status()
            assert final["bad_crc"] == final["bad_fields"] == 0
            report = dict(status="passed", domain=121, interface="lo", robot_connected=False,
                          tracking_cycles=1000, tracking_elapsed_s=elapsed, peer=final,
                          dual_policy=pipeline.metrics(), dds=pipeline.env.metrics())
            Path(output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k:v for k,v in report.items() if k not in ("dual_policy", "dds")}), flush=True)
    finally:
        if pipeline is not None:
            pipeline.ctrl_manager.shutdown()
            pipeline.env.shutdown()
        sender.close()
        inbox.put(dict(op="stop"))
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)


if __name__ == "__main__":
    run(sys.argv[1])
