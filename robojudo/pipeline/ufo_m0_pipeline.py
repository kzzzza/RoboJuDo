"""UFO M0 reference rollout and guarded real G1 control flow."""

import logging
import os
import time
from collections import deque

import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


@pipeline_registry.register
class UfoM0Pipeline(RlPipeline):
    _video_mode = False
    INIT_STEPS = 500
    RETURN_SECONDS = 5.0
    BUTTONS = ("A", "B", "X", "R1", "R2")

    def __init__(self, cfg):
        self._real_mode = not cfg.env.is_sim
        self._video_mode = cfg.policy.latent_source == "controller"
        if self._real_mode and not getattr(cfg, "do_safety_check", True):
            logger.warning("UFO M0 Real fall check disabled; tilted/lying poses will not stop the policy")
        if self._real_mode and self._video_mode and not cfg.video_real_enabled:
            raise RuntimeError("Video Real is disabled until sim and latency qualification passes")
        if self._real_mode and os.environ.get("UFO_REAL_ROBOT_OK") != "1":
            raise RuntimeError("Real UFO M0 control requires UFO_REAL_ROBOT_OK=1")
        if self._real_mode and self._video_mode and getattr(cfg, "auto_debug_mode", False):
            from robojudo.environment.utils.g1_motion_mode import ensure_debug_mode

            ensure_debug_mode(cfg.env.unitree.net_if)
        self._feedback_valid_cycle = False
        super().__init__(cfg)
        if self._real_mode:
            self.policy.policy.set_tracking_enabled(False)

    def reset(self):
        super().reset()
        self.video_metrics = deque(maxlen=4096)
        if self._video_mode:
            self._latent_ctrl().set_playing(not self._real_mode)
        if self._real_mode:
            self.real_state = "DISARMED"
            self.stop_reason = None
            self._fault_stop = False
            self._released_after_stop = False
            self._has_actuated = False
            self._standing_ready = False
            self._previous_buttons = {name: False for name in self.BUTTONS}
            self._init_start = None
            self._init_step = 0
            self._return_start = None
            self._return_step = 0
            self.policy.policy.set_tracking_enabled(False)
        else:
            if self._video_mode:
                qpos = self.env.model.qpos0.copy()
                qpos[-29:] = self.env.dof_cfg.default_pos
                self.env.reborn(qpos, np.zeros(self.env.model.nv))
                self._video_seen = False
                self._video_sim_return_step = 0
                self._video_sim_return_start = None
                self._video_sim_return_reason = None
                self._video_sim_waiting_person = False
                self.video_sim_done = False
            else:
                motion = self.policy.policy.motion
                self.env.reborn(motion.first_qpos(), motion.first_qvel())
            self.env.update()

    def _latent_ctrl(self):
        return self.ctrl_manager.controllers["LatentCtrl"].inst

    def _source_reset(self):
        self.policy.policy.reset_reference()
        if self._video_mode:
            controller = self._latent_ctrl()
            if controller.cfg_ctrl.source == "file":
                controller.reset()

    def _source_play(self, enabled: bool):
        self.policy.policy.set_tracking_enabled(enabled)
        if self._video_mode:
            self._latent_ctrl().set_playing(enabled)

    def _source_done(self) -> bool:
        return self._latent_ctrl().ended if self._video_mode else self.policy.policy.motion_done

    def _video_sim_step(self, ctrl_data, dry_run):
        if dry_run:
            return True
        if self.video_sim_done:
            self.env.step(np.asarray(self.env.dof_cfg.default_pos, dtype=np.float32))
            self.timestep += 1
            return True
        source = ctrl_data["LatentCtrl"]
        if self._video_sim_waiting_person:
            if source["valid"] and not source.get("person_lost", False) and not source["ended"]:
                self.policy.reset()
                self._source_play(True)
                self._video_sim_waiting_person = False
            self.env.step(np.asarray(self.env.dof_cfg.default_pos, dtype=np.float32))
            self.timestep += 1
            return True
        should_return = (source["ended"] or
                         (self._video_seen and source.get("person_lost", False)))
        if should_return and self._video_sim_return_start is None:
            self._video_sim_return_start = np.asarray(self.env.dof_pos, dtype=np.float32).copy()
            self._video_sim_return_reason = "person_lost" if source.get("person_lost", False) else "ended"
            self._source_play(False)
        if self._video_sim_return_start is not None:
            self._video_sim_return_step += 1
            total = max(1, int(self.RETURN_SECONDS * self.freq))
            alpha = min(self._video_sim_return_step / total, 1.0)
            standing = np.asarray(self.env.dof_cfg.default_pos, dtype=np.float32)
            self.env.step((1 - alpha) * self._video_sim_return_start + alpha * standing)
            if self._video_sim_return_step >= total:
                if self._video_sim_return_reason == "person_lost":
                    self._video_sim_waiting_person = True
                else:
                    self.video_sim_done = True
                self._video_sim_return_start = None
            self.timestep += 1
            return True
        if not source["valid"]:
            if self._video_seen:
                raise RuntimeError("Video latent source became stale")
            # Show the initial standing pose without letting it fall while
            # the camera and visual models warm up.
            if self.env.cfg_env.render:
                self.env.render_frame()
                self.timestep += 1
            return True
        self._video_seen = True
        return False

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        if self._video_mode and self.env.cfg_env.is_sim and "latent_frame" in extras:
            source = ctrl_data["LatentCtrl"]
            captured_at = self._latent_ctrl().capture_time
            age = None if captured_at is None else max(0.0, time.monotonic() - captured_at)
            gravity = get_gravity_orientation(self.env.base_quat)
            tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
            self.video_metrics.append({
                "step": self.timestep, "latent_frame": int(source["frame_id"]),
                "capture_to_pd_s": age,
                "produce_to_pd_s": (None if source.get("produced_time") is None else
                                    max(0.0, time.monotonic() - source["produced_time"])),
                "sample_time_s": source.get("sample_time"),
                "source_frame_id": source.get("source_frame_id"),
                "session_id": source.get("session_id"),
                "base_height_m": float(self.env.data.qpos[2]),
                "base_tilt_rad": tilt,
            })
        super().post_step_callback(env_data, ctrl_data, extras, pd_target)

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        if self._real_mode:
            logger.warning("UFO M0 Real is disarmed; press A to begin standing initialization")
            return
        return super().prepare(init_motor_angle, prepare_seconds)

    def _button_snapshot(self, ctrl_data) -> dict[str, bool]:
        raw = ctrl_data.get("UnitreeCtrl", {}).get("buttons", {})
        return {name: bool(raw.get(name, False)) for name in self.BUTTONS}

    def _pressed(self, buttons: dict[str, bool], name: str) -> bool:
        return buttons[name] and not self._previous_buttons[name]

    def _finish_handled(self, ctrl_data, buttons: dict[str, bool]) -> bool:
        self._previous_buttons = buttons
        self.ctrl_manager.post_step_callback(ctrl_data)
        self.timestep += 1
        return True

    def _send_hold(self):
        sender = getattr(self.env, "send_hold", None)
        return bool(sender()) if sender is not None else False

    def _enter_stop(self, reason: str, *, fault: bool):
        if self.real_state != "STOPPED":
            logger.warning("UFO M0 Real stopped: %s", reason)
            self.stop_reason = reason
        self.real_state = "STOPPED"
        self._fault_stop = self._fault_stop or fault
        self._released_after_stop = False
        self._source_play(False)

    def _check_feedback(self, env_data):
        for key, size in (("dof_pos", 29), ("dof_vel", 29), ("base_quat", 4), ("base_ang_vel", 3)):
            values = np.asarray(env_data[key], dtype=np.float64)
            if values.shape != (size,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Invalid real robot feedback: {key}")
        self._feedback_valid_cycle = True

    def _attitude_is_valid(self) -> bool:
        quat = np.asarray(self.env.base_quat, dtype=np.float64)
        if quat.shape != (4,) or not np.all(np.isfinite(quat)):
            return False
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or abs(norm - 1.0) > 0.05:
            return False
        gravity = np.asarray(get_gravity_orientation(quat / norm), dtype=np.float64)
        if gravity.shape != (3,) or not np.all(np.isfinite(gravity)):
            return False
        return bool(np.arccos(np.clip(-gravity[2], -1.0, 1.0)) <= 1.0)

    def _begin_return(self):
        guard = getattr(self.env, "target_guard", None)
        last_command = getattr(guard, "last_command", None)
        self._return_start = (
            np.asarray(last_command, dtype=np.float64).copy()
            if last_command is not None
            else np.asarray(self.env.dof_pos, dtype=np.float64)
        )
        self._return_step = 0
        self.real_state = "RETURNING"
        self._source_play(False)
        logger.info("UFO M0 motion completed; returning to standing pose")

    def _video_real_step(self, env_data, ctrl_data, dry_run=False) -> bool:
        """Camera control: A stands, B tracks z, R2 stops before inference."""
        if dry_run:
            return True
        buttons = self._button_snapshot(ctrl_data)
        source = ctrl_data["LatentCtrl"]
        if buttons["R2"]:
            self._enter_stop("R2", fault=False)
            if self._has_actuated:
                self._send_hold()
            return self._finish_handled(ctrl_data, buttons)

        self._check_feedback(env_data)
        if self.cfg.do_safety_check and not self._attitude_is_valid():
            self._enter_stop("invalid or fallen base attitude", fault=True)
            if self._has_actuated:
                self._send_hold()
            return self._finish_handled(ctrl_data, buttons)

        if self.real_state == "STOPPED":
            # A held together with R2 cannot arm the robot after R2 is released.
            if not buttons["A"]:
                self._released_after_stop = True
            if self._fault_stop or not self._released_after_stop or not self._pressed(buttons, "A"):
                if self._has_actuated:
                    self._send_hold()
                return self._finish_handled(ctrl_data, buttons)

        if self._pressed(buttons, "A") and self.real_state in ("DISARMED", "STOPPED", "STANDING"):
            self._init_start = np.asarray(env_data.dof_pos, dtype=np.float64).copy()
            self._init_step = 0
            self._standing_ready = False
            self.policy.reset()
            self._source_play(False)
            self.real_state = "INITIALIZING"
            logger.info("UFO video Real standing initialization started")

        if self.real_state == "DISARMED":
            return self._finish_handled(ctrl_data, buttons)

        standing = np.asarray(self.env.dof_cfg.default_pos, dtype=np.float64)
        if self.real_state == "INITIALIZING":
            self._init_step += 1
            progress = min(self._init_step / self.INIT_STEPS, 1.0)
            self._has_actuated = True
            self.env.step((1.0 - progress) * self._init_start + progress * standing)
            if self._init_step >= self.INIT_STEPS:
                self.real_state = "STANDING"
                self._standing_ready = True
                logger.info("UFO video Real standing initialization completed")
            return self._finish_handled(ctrl_data, buttons)

        if self.real_state == "RETURNING":
            self._return_step += 1
            total = max(1, int(self.RETURN_SECONDS * self.freq))
            progress = min(self._return_step / total, 1.0)
            self.env.step((1.0 - progress) * self._return_start + progress * standing)
            if self._return_step >= total:
                self.real_state = "STANDING"
            return self._finish_handled(ctrl_data, buttons)

        if self.real_state == "STANDING":
            if self._pressed(buttons, "B") and source["valid"]:
                self.policy.reset()
                self._source_reset()
                self._source_play(True)
                self.real_state = "TRACKING"
                self._previous_buttons = buttons
                return False
            self.env.step(standing)
            return self._finish_handled(ctrl_data, buttons)

        if self.real_state == "TRACKING":
            if self._pressed(buttons, "X") or source["ended"]:
                self._begin_return()
                return self._finish_handled(ctrl_data, buttons)
            if source.get("person_lost", False):
                logger.warning("UFO video Real person lost; returning to standing pose")
                self._begin_return()
                return self._finish_handled(ctrl_data, buttons)
            if not source["valid"]:
                self._enter_stop("video latent source stale or invalid", fault=True)
                if self._has_actuated:
                    self._send_hold()
                return self._finish_handled(ctrl_data, buttons)
            self._previous_buttons = buttons
            return False

        raise RuntimeError(f"Unexpected UFO video Real state: {self.real_state}")

    def pre_policy_step(self, env_data, ctrl_data, dry_run=False) -> bool:
        if not self._real_mode:
            return self._video_sim_step(ctrl_data, dry_run) if self._video_mode else False
        if self._video_mode:
            return self._video_real_step(env_data, ctrl_data, dry_run)
        if dry_run:
            return False

        buttons = self._button_snapshot(ctrl_data)
        if buttons["R2"]:
            self._enter_stop("R2", fault=False)
            if self._has_actuated:
                self._send_hold()
            return self._finish_handled(ctrl_data, buttons)

        self._check_feedback(env_data)
        if self.cfg.do_safety_check and not self._attitude_is_valid():
            self._enter_stop("invalid or fallen base attitude", fault=True)
            if self._has_actuated:
                self._send_hold()
            return self._finish_handled(ctrl_data, buttons)

        if self.real_state == "STOPPED":
            if not buttons["R1"] and not buttons["B"]:
                self._released_after_stop = True
            combo = buttons["R1"] and buttons["B"]
            previous_combo = self._previous_buttons["R1"] and self._previous_buttons["B"]
            if (
                not self._fault_stop and self._standing_ready
                and self._released_after_stop and combo and not previous_combo
            ):
                self.policy.reset()
                self._source_reset()
                self._source_play(True)
                self.real_state = "TRACKING"
                logger.info("UFO M0 Real rearmed with R1+B")
                self._previous_buttons = buttons
                return False
            if not self._fault_stop and not self._standing_ready and self._pressed(buttons, "A"):
                self.real_state = "DISARMED"
            else:
                if self._has_actuated:
                    self._send_hold()
                return self._finish_handled(ctrl_data, buttons)

        if self._pressed(buttons, "A"):
            self._init_start = np.asarray(env_data.dof_pos, dtype=np.float64).copy()
            self._init_step = 0
            self._standing_ready = False
            self.policy.reset()
            self._source_play(False)
            self.real_state = "INITIALIZING"
            logger.info("UFO M0 Real standing initialization started")

        if self.real_state == "DISARMED":
            return self._finish_handled(ctrl_data, buttons)

        standing = np.asarray(self.env.dof_cfg.default_pos, dtype=np.float64)
        if self.real_state == "INITIALIZING":
            self._init_step += 1
            progress = self._init_step / self.INIT_STEPS
            target = (1.0 - progress) * self._init_start + progress * standing
            self._has_actuated = True
            self.env.step(target)
            if self._init_step >= self.INIT_STEPS:
                self.real_state = "STANDING"
                self._standing_ready = True
                logger.info("UFO M0 Real standing initialization completed")
            return self._finish_handled(ctrl_data, buttons)

        if self.real_state == "STANDING":
            if self._pressed(buttons, "R1"):
                self._source_reset()
                self._source_play(False)
                self.real_state = "ENABLED"
                self._previous_buttons = buttons
                return False
            self.env.step(standing)
            return self._finish_handled(ctrl_data, buttons)

        if self._pressed(buttons, "X"):
            self._source_reset()
            self._source_play(False)
            self.real_state = "ENABLED"
            self._previous_buttons = buttons
            return False

        if self.real_state == "ENABLED" and self._pressed(buttons, "B"):
            self._source_reset()
            self._source_play(True)
            self.real_state = "TRACKING"

        if self.real_state == "RETURNING":
            self._return_step += 1
            total = max(1, int(self.RETURN_SECONDS * self.freq))
            progress = min(self._return_step / total, 1.0)
            target = (1.0 - progress) * self._return_start + progress * standing
            self.env.step(target)
            if self._return_step >= total:
                self.real_state = "STANDING"
                logger.info("UFO M0 Real returned to standing pose")
            return self._finish_handled(ctrl_data, buttons)

        self._previous_buttons = buttons
        return False

    def step(self, dry_run=False):
        self._feedback_valid_cycle = False
        self._dry_run_cycle = dry_run
        try:
            super().step(dry_run=dry_run)
            if self._real_mode and not dry_run and self.real_state == "TRACKING":
                if self._source_done():
                    self._begin_return()
        except Exception as exc:
            if not self._real_mode or dry_run:
                raise
            logger.exception("UFO M0 Real control step failed: %s", exc)
            self._enter_stop(str(exc), fault=True)
            if self._feedback_valid_cycle:
                if self._has_actuated:
                    self._send_hold()
            else:
                # Feedback was not refreshed: do not send a target from stale state.
                raise
        finally:
            self._dry_run_cycle = False

    def before_env_step(self, env_data, ctrl_data, extras, pd_target):
        if not (self._real_mode and self._video_mode and self.real_state == "TRACKING"):
            return
        captured = ctrl_data["LatentCtrl"].get("capture_time")
        if captured is None or time.monotonic() - captured > 5.0:
            raise ValueError("Video latent exceeded 5 seconds before actuation")
        source = ctrl_data["LatentCtrl"]
        produced = source.get("produced_time")
        self.video_metrics.append({
            "step": self.timestep, "latent_frame": int(source.get("frame_id", -1)),
            "capture_to_pd_s": max(0.0, time.monotonic() - captured),
            "produce_to_pd_s": None if produced is None else max(0.0, time.monotonic() - produced),
            "sample_time_s": source.get("sample_time"),
            "source_frame_id": source.get("source_frame_id"),
            "session_id": source.get("session_id"),
        })

    def safety_check(self):
        if not self._real_mode:
            return super().safety_check()
        if getattr(self, "_dry_run_cycle", False):
            return
        if self.cfg.do_safety_check and not self._attitude_is_valid():
            self._enter_stop("fallen base attitude", fault=True)
            if self._has_actuated:
                self._send_hold()

    def safe_stop(self):
        """Best-effort measured-position hold before the common runner exits."""
        if not self._real_mode:
            return
        self._enter_stop("runner exit", fault=True)
        if not self._has_actuated:
            return
        try:
            self.env.update()
            data = self.env.get_data()
            self._check_feedback(data)
            self._send_hold()
        except Exception as exc:
            logger.warning("No fresh valid feedback for exit hold: %s", exc)
