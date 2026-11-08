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


def check_artifacts(model_dir: Path, manifest_path: Path, names=None):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported UFO M0 artifact manifest")
    expected_files = manifest["sha256"] if names is None else {name: manifest["sha256"][name] for name in names}
    for name, expected in expected_files.items():
        actual = sha256(model_dir / name)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {name}: {actual} != {expected}")
    return expected_files


def check_host(net_if: str, driver="unitree_cpp"):
    if importlib.util.find_spec(driver) is None:
        raise RuntimeError(f"{driver} is unavailable")
    interfaces = {name for _, name in socket.if_nameindex()}
    if net_if not in interfaces:
        raise RuntimeError(f"Network interface {net_if!r} is unavailable")
    operstate = Path(f"/sys/class/net/{net_if}/operstate").read_text().strip()
    if operstate != "up" and not (net_if == "lo" and operstate == "unknown"):
        raise RuntimeError(f"Network interface {net_if!r} is {operstate}, expected up")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("g1_ufo_m0_real", "g1_ufo_video_real_file",
                                             "g1_ufo_video_real_stream", "g1_ufo_video_real_workstation",
                                             "g1_ufo_video_real_workstation_amo"),
                        default="g1_ufo_m0_real")
    parser.add_argument("--net-if", help="Explicit workstation DDS interface")
    parser.add_argument("--motion-path", help="Override the configured LAFAN G1 motion")
    parser.add_argument("--latent-dir", help="Directory containing z.npy and timestamps.npy for video-file control")
    parser.add_argument("--latent-variant", choices=("original", "tcn", "mlp"), default="original")
    parser.add_argument("--offline", action="store_true", help="Skip host driver and network checks")
    parser.add_argument("--read-only", action="store_true", help="Also receive G1 lowstate; never publish commands")
    args = parser.parse_args()

    cfg = ConfigManager(args.config).get_cfg()
    workstation = args.config.startswith("g1_ufo_video_real_workstation")
    if args.read_only and (args.offline or not workstation):
        parser.error("--read-only requires the workstation config without --offline")
    if args.net_if:
        cfg.env.unitree.net_if = args.net_if
    if workstation and not args.offline and not cfg.env.unitree.net_if:
        parser.error("Workstation preflight requires --net-if")
    expected_env = "UnitreeG1DdsEnv" if workstation else "UnitreeCppEnv"
    if cfg.env.env_type != expected_env or cfg.env.dof.num_dofs != 29:
        raise ValueError("Unexpected UFO M0 Real environment configuration")
    model_dir = Path(cfg.policy.model_dir).expanduser()
    video_stream = args.config == "g1_ufo_video_real_stream" or workstation
    actor_files = ("FBcprAuxModel.onnx", "FBcprAuxModel.meta.json")
    digests = check_artifacts(model_dir, MANIFEST, actor_files if video_stream else None)
    if video_stream:
        latent = next(ctrl for ctrl in cfg.ctrl if ctrl.ctrl_type == "LatentCtrl")
        expected_source = "socket" if workstation else "tcp"
        if not cfg.video_real_enabled or not cfg.auto_debug_mode or latent.source != expected_source:
            raise ValueError("Video Real stream has an unexpected transport or debug-mode configuration")
    metadata = json.loads((model_dir / "FBcprAuxModel.meta.json").read_text())
    if metadata["control_joint_names"] != cfg.env.dof.joint_names:
        raise ValueError("Model and real robot joint order differ")
    if metadata["actor_obs_dim"] != 721 or metadata["z_dim"] != 256:
        raise ValueError("Model metadata dimensions differ from UFO M0")

    providers = ["CPUExecutionProvider"]
    actor = ort.InferenceSession(str(model_dir / "FBcprAuxModel.onnx"), providers=providers)
    if [(item.name, item.shape[-1]) for item in actor.get_inputs()] != [("actor_obs", 721)]:
        raise ValueError("Unexpected actor input contract")
    motion = None
    if cfg.policy.latent_source == "motion":
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
        backward = ort.InferenceSession(str(model_dir / "backward_encoder.onnx"), providers=providers)
        if {item.name: item.shape[-1] for item in backward.get_inputs()} != {
            "state": 64, "privileged_state": 463,
        }:
            raise ValueError("Unexpected B network input contract")
        z = backward.run(["z"], {"state": state, "privileged_state": privileged})[0]
    elif args.config.endswith("_file"):
        if not args.latent_dir:
            parser.error("--latent-dir is required for g1_ufo_video_real_file")
        directory = Path(args.latent_dir).expanduser()
        z_values = np.load(directory / "z.npy", allow_pickle=False)
        times = np.load(directory / "timestamps.npy", allow_pickle=False)
        metadata_path = directory / "latent_metadata.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()).get("variant") != args.latent_variant:
            raise ValueError("Video latent variant does not match metadata")
        if (z_values.dtype != np.float32 or z_values.ndim != 2 or z_values.shape[1] != 256
                or not len(z_values) or times.shape != (len(z_values),)
                or not np.isfinite(z_values).all() or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)
                or not np.allclose(np.linalg.norm(z_values, axis=1), 16, atol=1e-3)):
            raise ValueError("Invalid video z.npy/timestamps.npy")
        z = z_values[:1]
    else:
        z = np.zeros((1, 256), dtype=np.float32)
        z[0, 0] = 16.0
    if z.shape != (1, 256) or not np.all(np.isfinite(z)):
        raise ValueError("Invalid B network output")
    actor_obs = np.zeros((1, 721), dtype=np.float32)
    actor_obs[:, -256:] = z
    action = actor.run(["action"], {"actor_obs": actor_obs})[0]
    if action.shape != (1, 29) or not np.all(np.isfinite(action)):
        raise ValueError("Invalid actor output")

    amo = None
    if args.config.endswith("_amo"):
        from box import Box
        from robojudo.policy.amo_policy import AMOPolicy

        directory = Path(cfg.amo_policy.policy_file).parent
        amo_digests = check_artifacts(directory, directory / "artifact_manifest.json")
        policy = AMOPolicy(cfg.amo_policy, "cpu")
        sample = Box(dof_pos=np.asarray(cfg.amo_policy.obs_dof.default_pos, dtype=np.float32),
                     dof_vel=np.zeros(23, dtype=np.float32), base_quat=np.array([0., 0., 0., 1.]),
                     base_ang_vel=np.zeros(3, dtype=np.float32))
        obs, _ = policy.get_observation(sample, {})
        result = policy.get_action(obs)
        if obs.shape != (1043,) or result.shape != (15,) or not np.isfinite(result).all():
            raise ValueError("Unexpected AMO observation/action contract")
        amo = {"sha256": amo_digests, "obs_dim": 1043, "action_dim": 15, "device": "cpu"}

    if not args.offline:
        check_host(cfg.env.unitree.net_if, "unitree_sdk2py" if workstation else "unitree_cpp")
        if video_stream and importlib.util.find_spec("unitree_sdk2py") is None:
            raise RuntimeError("unitree_sdk2py is required for automatic G1 debug mode")
    feedback = None
    if args.read_only:
        from robojudo.environment.utils.g1_dds import G1StateReader

        reader = G1StateReader(cfg.env.unitree.net_if, feedback_timeout_s=cfg.env.feedback_timeout_s)
        try:
            state = reader.wait(cfg.env.connect_timeout_s)
            feedback = {"tick": state.tick, "mode_machine": state.mode_machine,
                        "joints": len(state.q), "remote_bytes": len(state.remote), **reader.metrics()}
        finally:
            reader.close()
    print(json.dumps({
        "status": "passed", "control_environment_created": False,
        "model_sha256": digests, "motion": str(motion.path) if motion is not None else None,
        "motion_frames": motion.num_frames if motion is not None else None,
        "latent_source": cfg.policy.latent_source, "latent_variant": args.latent_variant if motion is None else None,
        "network_interface": cfg.env.unitree.net_if,
        "host_checks": not args.offline,
        "feedback": feedback,
        "amo": amo,
    }, indent=2))


if __name__ == "__main__":
    main()
