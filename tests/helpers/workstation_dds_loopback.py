"""A two-process G1 DDS contract simulator. Hard restricted to domain 121 / lo.

This emulates message exchange, NOT robot dynamics or physical stability.
"""

import json
import multiprocessing
import os
import socket
import struct
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOMAIN = 121
sys.path.insert(0, str(ROOT))


def configure_loopback():
    import unitree_sdk2py.core.channel as channel

    # No multicast or physical interface can be used, even on a host with G1 connected.
    channel.ChannelConfigHasInterface = """<CycloneDDS><Domain Id="any">
      <General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast></General>
      <Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>20</MaxAutoParticipantIndex>
        <Peers><Peer Address="127.0.0.1"/></Peers></Discovery>
    </Domain></CycloneDDS>"""
    channel.ChannelFactoryInitialize(DOMAIN, "lo")


def peer(inbox, outbox):
    import queue
    import threading

    configure_loopback()
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    crc = CRC()
    lock = threading.Lock()
    seen = {"commands": 0, "bad_crc": 0, "bad_fields": 0}
    positions = [-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0] * 17

    def command(msg):
        with lock:
            seen["commands"] += 1
            seen["bad_crc"] += int(crc.Crc(msg) != msg.crc)
            seen["bad_fields"] += int(
                msg.mode_machine != 5 or msg.mode_pr != 0
                or any(m.mode != 1 or m.dq != 0 or m.tau != 0 for m in msg.motor_cmd[:29])
                or any(m.mode != 0 for m in msg.motor_cmd[29:]))
            positions[:] = [m.q for m in msg.motor_cmd[:29]]

    subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    subscriber.Init(command, 1)
    publisher = ChannelPublisher("rt/lowstate", LowState_)
    publisher.Init()
    outbox.put({"ready": True})
    keys = 0
    paused = False
    tick = 0
    try:
        while True:
            try:
                while True:
                    request = inbox.get_nowait()
                    if request["op"] == "stop":
                        return
                    if request["op"] == "keys":
                        keys = request["value"]
                    elif request["op"] == "pause":
                        paused = request["value"]
                    elif request["op"] == "status":
                        with lock:
                            outbox.put(dict(seen))
            except queue.Empty:
                pass
            if not paused:
                msg = unitree_hg_msg_dds__LowState_()
                msg.tick = tick = tick + 5
                msg.mode_machine = 5
                msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
                msg.wireless_remote[2:4] = struct.pack("<H", keys)
                with lock:
                    for i, q in enumerate(positions):
                        msg.motor_state[i].q = q
                msg.crc = crc.Crc(msg)
                publisher.Write(msg)
            time.sleep(0.005)
    finally:
        publisher.Close()
        subscriber.Close()


def run(output):
    import numpy as np
    from unittest.mock import patch

    from robojudo.config.config_manager import ConfigManager
    from robojudo.environment.utils.g1_dds import G1DdsError
    from robojudo.pipeline.ufo_m0_pipeline import UfoM0Pipeline
    from robojudo.video.latent_packet import encode_packet

    context = multiprocessing.get_context("spawn")
    inbox, outbox = context.Queue(), context.Queue()
    process = context.Process(target=peer, args=(inbox, outbox))
    process.start()
    pipeline = None
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    report = {}
    with tempfile.TemporaryDirectory(prefix="robojudo-dds-") as temp:
        cfg = ConfigManager("g1_ufo_video_real_workstation").get_cfg()
        cfg.env.dds_domain_id = DOMAIN
        cfg.env.unitree.net_if = "lo"
        cfg.env.connect_timeout_s = 8.0
        # Only the test fixture disables mode switching. Production config remains enabled.
        cfg.auto_debug_mode = False
        cfg.ctrl[-1].socket_path = str(Path(temp) / "latent.sock")
        cfg.ctrl[-1].variant = "tcn"
        assert cfg.env.dds_domain_id == 121 and cfg.env.unitree.net_if == "lo"
        os.environ["UFO_REAL_ROBOT_OK"] = "1"
        frame = 0
        z = np.zeros(256, np.float32)
        z[0] = 16

        def status():
            inbox.put({"op": "status"})
            return outbox.get(timeout=5)

        def keys(value):
            inbox.put({"op": "keys", "value": value})
            time.sleep(0.03)

        def cycle(*, lost=False):
            nonlocal frame
            frame += 1
            sender.sendto(encode_packet("tcn", frame, time.monotonic(),
                                       np.zeros(256, np.float32) if lost else z,
                                       session_id=42, person_lost=lost), cfg.ctrl[-1].socket_path)
            pipeline.step()

        try:
            assert outbox.get(timeout=10)["ready"]
            configure_loopback()
            with patch("robojudo.environment.utils.g1_motion_mode.ensure_debug_mode") as release:
                pipeline = UfoM0Pipeline(cfg)
                release.assert_not_called()
            assert status()["commands"] == 0, "Initialization must not send commands"
            for _ in range(5):
                cycle()
            assert status()["commands"] == 0, "DISARMED must not send commands"
            keys(1 << 8)  # A
            cycle()
            keys(0)
            for _ in range(pipeline.INIT_STEPS - 1):
                cycle()
                time.sleep(0.002)
            assert pipeline.real_state == "STANDING"
            keys(1 << 9)  # B
            cycle()
            assert pipeline.real_state == "TRACKING"
            keys(0)
            began = time.monotonic()
            durations = []
            for _ in range(1000):
                start = time.monotonic()
                cycle()
                durations.append(time.monotonic() - start)
                time.sleep(max(0, 0.02 - (time.monotonic() - start)))
            assert pipeline.real_state == "TRACKING"
            report["healthy_tracking"] = {
                "cycles": 1000, "elapsed_s": time.monotonic() - began,
                "compute_p95_s": float(np.percentile(durations, 95)),
                "compute_max_s": max(durations), "driver": pipeline.env.metrics()}

            # Real SDK bytes -> original UnitreeCtrl -> original R2 state transition.
            keys(1 << 4)
            before = len(pipeline.video_metrics)
            cycle()
            assert pipeline.real_state == "STOPPED" and not pipeline._fault_stop
            assert len(pipeline.video_metrics) == before
            keys(0)
            cycle()
            keys(1 << 9)
            cycle()
            assert pipeline.real_state == "STOPPED", "B alone must not rearm after R2"
            keys(0)
            cycle()
            keys(1 << 8)
            cycle()
            keys(0)
            for _ in range(pipeline.INIT_STEPS - 1):
                cycle()
                time.sleep(0.002)
            keys(1 << 9)
            cycle()
            keys(0)
            assert pipeline.real_state == "TRACKING"

            cycle(lost=True)
            assert pipeline.real_state == "RETURNING"
            for _ in range(int(pipeline.RETURN_SECONDS * pipeline.freq)):
                cycle()
                time.sleep(0.002)
            assert pipeline.real_state == "STANDING", "Reappearance must require a new B press"
            keys(1 << 9)
            cycle()
            keys(0)
            assert pipeline.real_state == "TRACKING"

            # A physically absent feedback stream cannot cause replay or stale holds.
            inbox.put({"op": "pause", "value": True})
            time.sleep(0.16)
            before = status()["commands"]
            try:
                cycle()
            except G1DdsError:
                pass
            else:
                raise AssertionError("Missing G1 feedback must terminate the failed cycle")
            time.sleep(0.05)
            assert status()["commands"] == before
            assert pipeline.real_state == "STOPPED" and pipeline._fault_stop
            inbox.put({"op": "pause", "value": False})
            time.sleep(0.05)
            before = len(pipeline.video_metrics)
            keys(1 << 9)
            cycle()
            keys(0)
            cycle()
            assert pipeline.real_state == "STOPPED" and len(pipeline.video_metrics) == before
            pipeline.safe_stop()
            pipeline.ctrl_manager.shutdown()
            pipeline.env.shutdown()
            time.sleep(0.05)
            final = status()
            time.sleep(0.15)
            assert status()["commands"] == final["commands"], "Shutdown must stop the writer"
            assert final["bad_crc"] == final["bad_fields"] == 0
            report.update(status="passed", domain=DOMAIN, interface="lo", peer=final,
                          fault_latched=pipeline._fault_stop, robot_connected=False,
                          scope="HG DDS contracts and state machine; not physical dynamics")
        finally:
            if pipeline is not None:
                pipeline.ctrl_manager.shutdown()
                pipeline.env.shutdown()
            sender.close()
            inbox.put({"op": "stop"})
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            Path(output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run(sys.argv[1])
