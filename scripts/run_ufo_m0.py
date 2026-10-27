"""Finite UFO M0 sim2sim rollout with a selectable G1 reference motion."""

import argparse
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-path", default=None, help="LAFAN retargeted G1 .npz")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--render", action="store_true", help="Open a MuJoCo viewer")
    parser.add_argument("--show-reference", action="store_true", help="Show a translucent reference in the viewer")
    parser.add_argument("--record-video", default=None, metavar="PATH", help="Record a headless MP4")
    parser.add_argument("--video-content", choices=("actual", "comparison"), default="actual")
    parser.add_argument(
        "--stop-on-fall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when root height falls below 0.35 m (default: enabled)",
    )
    args = parser.parse_args()
    if args.record_video and (args.render or args.show_reference):
        parser.error("--record-video is headless; use --video-content comparison for the reference")
    if args.show_reference and not args.render:
        parser.error("--show-reference requires --render")
    if args.record_video:
        os.environ["MUJOCO_GL"] = "egl"

    import robojudo.pipeline
    from robojudo.config.config_manager import ConfigManager

    cfg = ConfigManager("g1_ufo_m0").get_cfg()
    if args.motion_path is not None:
        cfg.policy.motion_path = args.motion_path
    cfg.policy.start_frame = args.start_frame
    cfg.policy.max_frames = args.max_frames
    cfg.env.render = args.render
    cfg.env.show_reference = args.show_reference
    cfg.env.video_path = args.record_video
    cfg.env.video_content = args.video_content
    pipeline_class = getattr(robojudo.pipeline, cfg.pipeline_type)
    pipeline = pipeline_class(cfg=cfg)
    motion = pipeline.policy.policy.motion
    joint_error = []
    height_error = []
    body_error = []
    fallen = False
    start = time.monotonic()
    try:
        for step in range(motion.num_frames - 1):
            tick = time.monotonic()
            pipeline.step()
            reference_frame = step + 1
            joint_error.append(float(np.mean(np.abs(pipeline.env.dof_pos - motion.dof_pos[reference_frame]))))
            height_error.append(float(abs(pipeline.env.base_pos[2] - motion.body_pos[reference_frame, 0, 2])))
            actual_bodies = pipeline.env.data.xpos[1:31]
            reference_bodies = motion.body_pos[reference_frame, :30]
            actual_local = Rotation.from_quat(pipeline.env.base_quat).inv().apply(
                actual_bodies - actual_bodies[:1]
            )
            reference_local = Rotation.from_quat(motion.body_quat[reference_frame, 0]).inv().apply(
                reference_bodies - reference_bodies[:1]
            )
            body_error.append(float(np.mean(np.linalg.norm(actual_local - reference_local, axis=1))))
            if pipeline.env.base_pos[2] < 0.35:
                fallen = True
                if args.stop_on_fall:
                    break
            if args.render:
                time.sleep(max(0.0, pipeline.dt - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.env.shutdown()
    print(json.dumps({
        "motion": str(motion.path), "frames_run": len(joint_error),
        "reference_frames": motion.num_frames, "fallen": fallen,
        "mean_joint_abs_error_rad": float(np.mean(joint_error)) if joint_error else None,
        "max_frame_mean_joint_abs_error_rad": float(np.max(joint_error)) if joint_error else None,
        "mean_root_height_abs_error_m": float(np.mean(height_error)) if height_error else None,
        "mean_body_local_pos_error_m": float(np.mean(body_error)) if body_error else None,
        "elapsed_wall_seconds": round(time.monotonic() - start, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
