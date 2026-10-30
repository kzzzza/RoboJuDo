# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import logging
import time

import numpy as np

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument("--show-reference", action="store_true", help="Show a translucent reference in the viewer")
    parser.add_argument("--record-video", default=None, metavar="PATH", help="Record a headless MP4")
    parser.add_argument("--video-content", choices=("actual", "comparison"), default="actual")
    parser.add_argument("--latent-dir", help="Directory with z.npy and timestamps.npy for a video-file config")
    parser.add_argument("--latent-variant", choices=("original", "tcn", "mlp"), default="original")
    parser.add_argument("--latent-socket", default="/tmp/robojudo_ufo_latent.sock")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many control cycles")
    parser.add_argument("--metrics-json", help="Write video experiment metrics to this JSON path")
    args = parser.parse_args()
    if args.record_video and args.show_reference:
        parser.error("--record-video is headless; use --video-content comparison for the reference")
    return args


def main():
    args = parse_args()
    if args.record_video:
        os.environ["MUJOCO_GL"] = "egl"

    import robojudo.pipeline
    from robojudo.config.config_manager import ConfigManager

    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg = config_manager.get_cfg()
    for controller in cfg.ctrl:
        if controller.ctrl_type == "LatentCtrl":
            controller.variant = args.latent_variant
            if controller.source == "file":
                if not args.latent_dir:
                    raise ValueError("--latent-dir is required for a video-file config")
                controller.z_dir = args.latent_dir
            else:
                controller.socket_path = args.latent_socket
            if args.video_content == "comparison":
                raise ValueError("Video z has no reference qpos; use --video-content actual")
    if args.record_video or args.show_reference:
        if cfg.env.env_type != "MujocoEnv":
            raise ValueError("Reference visualization and video recording require MujocoEnv")
        cfg.env.video_path = args.record_video
        cfg.env.video_content = args.video_content
        cfg.env.render = args.show_reference or (cfg.env.render and not args.record_video)
        cfg.env.show_reference = args.show_reference

    pipeline_type = cfg.pipeline_type

    pipeline_class = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    try:
        if not cfg.env.is_sim:
            pipeline.prepare()
        elif getattr(pipeline, "_has_default_pose_mode", False):
            pipeline._set_default_pose_mode(True)
            logger.warning("Sim mode — holding default pose, press R to start motion")

        while args.max_steps is None or pipeline.timestep < args.max_steps:
            time_start = time.time()
            pipeline.step()
            time_end = time.time()
            time_diff = time_end - time_start

            # keep the pipeline running at the desired frequency
            if not cfg.run_fullspeed:
                time_diff = pipeline.dt - time_diff
                if time_diff > 0:
                    time.sleep(time_diff)
                elif not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        break
            if getattr(pipeline, "video_sim_done", False):
                break
    except KeyboardInterrupt:
        logger.info("Interrupted; finalizing simulation outputs")
    finally:
        if args.metrics_json and getattr(pipeline, "_video_mode", False) and cfg.env.is_sim:
            metrics = pipeline.video_metrics
            ages = np.asarray([row["capture_to_pd_s"] for row in metrics
                               if row["capture_to_pd_s"] is not None], dtype=np.float64)
            heights = np.asarray([row["base_height_m"] for row in metrics], dtype=np.float64)
            first_low = next((row["step"] for row in metrics if row["base_height_m"] < 0.45), None)
            first_fall = next((row["step"] for row in metrics
                               if row["base_height_m"] < 0.45 and row["base_tilt_rad"] > 1.0), None)
            report = {"config": args.config, "variant": args.latent_variant,
                      "steps": pipeline.timestep, "policy_steps": len(metrics),
                      "unique_latent_frames": len({row["latent_frame"] for row in metrics}),
                      "low_height_step": first_low, "fall_step": first_fall,
                      "min_base_height_m": float(heights.min()) if len(heights) else None,
                      "capture_to_pd_p50_s": float(np.percentile(ages, 50)) if len(ages) else None,
                      "capture_to_pd_p95_s": float(np.percentile(ages, 95)) if len(ages) else None,
                      "samples": metrics}
            from pathlib import Path
            target = Path(args.metrics_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2) + "\n")
        try:
            safe_stop = getattr(pipeline, "safe_stop", None)
            if safe_stop is not None:
                safe_stop()
        finally:
            try:
                pipeline.ctrl_manager.shutdown()
            finally:
                pipeline.env.shutdown()


if __name__ == "__main__":
    main()
