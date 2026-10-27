# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

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

        while True:
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
    except KeyboardInterrupt:
        logger.info("Interrupted; finalizing simulation outputs")
    finally:
        try:
            safe_stop = getattr(pipeline, "safe_stop", None)
            if safe_stop is not None:
                safe_stop()
        finally:
            pipeline.env.shutdown()


if __name__ == "__main__":
    main()
