"""UFO M0 reference rollout and guarded real G1 control flow."""

import logging
import os

import numpy as np

from robojudo.pipeline import pipeline_registry
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


@pipeline_registry.register
class UfoM0Pipeline(RlPipeline):
    INIT_STEPS = 500
    RETURN_SECONDS = 5.0
    BUTTONS = ("A", "B", "X", "R1", "R2")

    def __init__(self, cfg):
        self._real_mode = not cfg.env.is_sim
        if self._real_mode and os.environ.get("UFO_REAL_ROBOT_OK") != "1":
            raise RuntimeError("Real UFO M0 control requires UFO_REAL_ROBOT_OK=1")
        self._feedback_valid_cycle = False
        super().__init__(cfg)
        if self._real_mode:
            self.policy.policy.set_tracking_enabled(False)

    def reset(self):
        super().reset()
        if self._real_mode:
            self.real_state = "DISARMED"
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
            motion = self.policy.policy.motion
            self.env.reborn(motion.first_qpos(), motion.first_qvel())
            self.env.update()

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
        self.real_state = "STOPPED"
        self._fault_stop = self._fault_stop or fault
        self._released_after_stop = False
        self.policy.policy.set_tracking_enabled(False)

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
        self.policy.policy.set_tracking_enabled(False)
        logger.info("UFO M0 motion completed; returning to standing pose")

    def pre_policy_step(self, env_data, ctrl_data, dry_run=False) -> bool:
        if not self._real_mode or dry_run:
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
                self.policy.policy.set_tracking_enabled(True)
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
            self.policy.policy.set_tracking_enabled(False)
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
                self.policy.policy.reset_reference()
                self.policy.policy.set_tracking_enabled(False)
                self.real_state = "ENABLED"
                self._previous_buttons = buttons
                return False
            self.env.step(standing)
            return self._finish_handled(ctrl_data, buttons)

        if self._pressed(buttons, "X"):
            self.policy.policy.reset_reference()
            self.policy.policy.set_tracking_enabled(False)
            self.real_state = "ENABLED"
            self._previous_buttons = buttons
            return False

        if self.real_state == "ENABLED" and self._pressed(buttons, "B"):
            self.policy.policy.reset_reference()
            self.policy.policy.set_tracking_enabled(True)
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
                if self.policy.policy.motion_done:
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
