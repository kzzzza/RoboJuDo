"""Read-only UFO M0 Real artifact and host preflight; never creates a control environment."""

import argparse
import hashlib
import importlib.util
import json
import socket
from pathlib import Path

import numpy as np
import onnxruntime as ort

from robojudo.config.config_manager import ConfigManager
from robojudo.utils.ufo_m0_motion import UfoM0Motion


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/models/g1/ufo_m0/artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_artifacts(model_dir: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported UFO M0 artifact manifest")
    for name, expected in manifest["sha256"].items():
        actual = sha256(model_dir / name)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {name}: {actual} != {expected}")
    return manifest["sha256"]


def check_host(net_if: str):
    if importlib.util.find_spec("unitree_cpp") is None:
        raise RuntimeError("unitree_cpp is unavailable")
    interfaces = {name for _, name in socket.if_nameindex()}
    if net_if not in interfaces:
        raise RuntimeError(f"Network interface {net_if!r} is unavailable")
    operstate = Path(f"/sys/class/net/{net_if}/operstate").read_text().strip()
    if operstate != "up":
        raise RuntimeError(f"Network interface {net_if!r} is {operstate}, expected up")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-path", help="Override the configured LAFAN G1 motion")
    parser.add_argument("--offline", action="store_true", help="Skip host driver and network checks")
    args = parser.parse_args()

    cfg = ConfigManager("g1_ufo_m0_real").get_cfg()
    if cfg.env.env_type != "UnitreeCppEnv" or cfg.env.dof.num_dofs != 29:
        raise ValueError("Unexpected UFO M0 Real environment configuration")
    model_dir = Path(cfg.policy.model_dir).expanduser()
    digests = check_artifacts(model_dir, MANIFEST)
    metadata = json.loads((model_dir / "FBcprAuxModel.meta.json").read_text())
    if metadata["control_joint_names"] != cfg.env.dof.joint_names:
        raise ValueError("Model and real robot joint order differ")
    if metadata["actor_obs_dim"] != 721 or metadata["z_dim"] != 256:
        raise ValueError("Model metadata dimensions differ from UFO M0")

    motion_path = args.motion_path or cfg.policy.motion_path
    motion = UfoM0Motion(
        motion_path,
        default_pos=np.asarray(cfg.env.dof.default_pos, dtype=np.float32),
        start_frame=cfg.policy.start_frame,
        max_frames=cfg.policy.max_frames,
    )
    motion_arrays = (motion.dof_pos, motion.dof_vel, motion.body_pos, motion.body_quat,
                     motion.body_vel, motion.body_ang_vel)
    if motion.num_frames < 2 or any(not np.all(np.isfinite(values)) for values in motion_arrays):
        raise ValueError("Motion needs at least two finite frames throughout")
    state, privileged = motion.backward_inputs(1, 2)
    if state.shape != (1, 64) or privileged.shape != (1, 463):
        raise ValueError("Unexpected B network motion input shape")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(privileged)):
        raise ValueError("Non-finite B network motion input")

    providers = ["CPUExecutionProvider"]
    backward = ort.InferenceSession(str(model_dir / "backward_encoder.onnx"), providers=providers)
    actor = ort.InferenceSession(str(model_dir / "FBcprAuxModel.onnx"), providers=providers)
    if {item.name: item.shape[-1] for item in backward.get_inputs()} != {
        "state": 64, "privileged_state": 463,
    }:
        raise ValueError("Unexpected B network input contract")
    if [(item.name, item.shape[-1]) for item in actor.get_inputs()] != [("actor_obs", 721)]:
        raise ValueError("Unexpected actor input contract")
    z = backward.run(["z"], {"state": state, "privileged_state": privileged})[0]
    if z.shape != (1, 256) or not np.all(np.isfinite(z)):
        raise ValueError("Invalid B network output")
    actor_obs = np.zeros((1, 721), dtype=np.float32)
    actor_obs[:, -256:] = z
    action = actor.run(["action"], {"actor_obs": actor_obs})[0]
    if action.shape != (1, 29) or not np.all(np.isfinite(action)):
        raise ValueError("Invalid actor output")

    if not args.offline:
        check_host(cfg.env.unitree.net_if)
    print(json.dumps({
        "status": "passed", "control_environment_created": False,
        "model_sha256": digests, "motion": str(motion.path),
        "motion_frames": motion.num_frames, "network_interface": cfg.env.unitree.net_if,
        "host_checks": not args.offline,
    }, indent=2))


if __name__ == "__main__":
    main()
