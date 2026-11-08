"""Explicit A/Y/B dual-policy control; one environment and one guarded writer."""

import logging
import os
import time
from collections import Counter, deque

import numpy as np

import robojudo.environment
from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.config.g1.env.g1_ufo_m0_env_cfg import G1UfoM0RealEnvCfg
from robojudo.controller import CtrlManager
from robojudo.environment.utils.real_target_guard import RealTargetGuard
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.pipeline.ufo_m0_pipeline import UfoM0Pipeline
from robojudo.tools.dof import DoFAdapter, merge_dof_cfgs


logger = logging.getLogger(__name__)
AXES = ("LeftX", "LeftY", "RightX", "RightY")


def amo_profile(env_dof, policy_cfg):
    """23 AMO joints plus the six existing G1 wrist defaults/gains."""
    return merge_dof_cfgs(merge_dof_cfgs(env_dof, G1_29DoF()), policy_cfg.obs_dof)


def reset_amo(wrapper, env_data, last_target=None):
    """Reset all adaptation/gait/history state without changing AMO parameters."""
    policy = wrapper.policy
    policy.reset()
    policy.last_action = np.zeros(policy.num_actions, dtype=np.float32)
    if last_target is not None:
        action_names = policy.cfg_action_dof.joint_names
        previous = DoFAdapter(wrapper.env_dof_cfg.joint_names, action_names).fit(last_target)
        policy.last_action = np.clip((previous - policy.default_pos) / policy.action_scale, -40, 40)
    policy.cmd = np.zeros(8, dtype=np.float32)
    policy.cmd[:4] = [mapping[1] for mapping in policy.commands_map]
    policy.gait_cycle = np.array([0.25, 0.25])
    policy.target_yaw = 0.0
    policy._in_place_stand_flag = True
    policy.proprio_history_buf.clear()
    policy.extra_history_buf.clear()
    for _ in range(policy.history_len):
        policy.proprio_history_buf.append(np.zeros(policy.n_proprio))
    for _ in range(policy.extra_history_len):
        policy.extra_history_buf.append(np.zeros(policy.n_proprio))
    observation, _ = wrapper.get_observation(env_data, {})
    current = observation[:policy.n_proprio].copy()
    for history in (policy.proprio_history_buf, policy.extra_history_buf):
        for i in range(len(history)):
            history[i] = current.copy()
    policy.timestep = 0
    policy.gait_cycle = np.array([0.25, 0.25])


@pipeline_registry.register
class UfoAmoPipeline(UfoM0Pipeline):
    BUTTONS = ("A", "Y", "B", "X", "R2")

    def __init__(self, cfg, *, clock=None):
        # Deliberately bypass the old single-policy reset/state machine.
        Pipeline.__init__(self, cfg)
        self._clock = clock or time.monotonic
        self._real_mode = not cfg.env.is_sim
        self._video_mode = True
        if self.device != "cpu" or cfg.policy.freq != 50 or cfg.amo_policy.freq != 50:
            raise ValueError("AMO/UFO control requires CPU policies at 50Hz")
        if self._real_mode and os.environ.get("UFO_REAL_ROBOT_OK") != "1":
            raise RuntimeError("Real UFO/AMO control requires UFO_REAL_ROBOT_OK=1")
        if self._real_mode and not cfg.video_real_enabled:
            raise RuntimeError("Video Real is disabled")
        self.freq, self.dt = 50, 0.02
        self.profiles = {
            "AMO": amo_profile(cfg.env.dof, cfg.amo_policy),
            "UFO": merge_dof_cfgs(cfg.env.dof, cfg.policy.action_dof),
        }
        # Load both models BEFORE releasing the robot's built-in motion control.
        self.policy = PolicyWrapper(cfg.policy, self.profiles["UFO"], "cpu")
        self.amo = PolicyWrapper(cfg.amo_policy, self.profiles["AMO"], "cpu")
        self.env = None
        self.ctrl_manager = None
        try:
            if self._real_mode and cfg.auto_debug_mode:
                from robojudo.environment.utils.g1_motion_mode import ensure_debug_mode

                ensure_debug_mode(cfg.env.unitree.net_if)
            env_class = getattr(robojudo.environment, cfg.env.env_type)
            self.env = env_class(cfg_env=cfg.env, device="cpu")
            self.ctrl_manager = CtrlManager(cfg.ctrl, env=self.env, device="cpu")
            self.visualizer = self.env.visualizer
            self.guard = getattr(self.env, "target_guard", None)
            if self.guard is None:
                velocity = DoFAdapter(G1_29DoF().joint_names, self.env.joint_names).fit(
                    G1UfoM0RealEnvCfg().target_velocity_limits)
                self.guard = RealTargetGuard(self.env.position_limits, velocity, self.dt, 0.5)
            if not self._real_mode:
                qpos = self.env.model.qpos0.copy()
                qpos[-29:] = self.profiles["AMO"].default_pos
                self.env.reborn(qpos, np.zeros(self.env.model.nv))
                self._sim_anchor = self.env.data.qpos[:7].copy()
                self._sim_supported = getattr(cfg, "sim_initial_support", False)
            self.env.update()
            self._check_feedback(self.env.get_data())
            self.real_state = "DISARMED"
            self.stop_reason = None
            self._fault_stop = False
            self._has_actuated = False
            self._released_after_stop = False
            self._previous_buttons = dict.fromkeys(self.BUTTONS, False)
            self._keys = set()
            self._last_q = np.asarray(self.env.dof_pos).copy()
            self._last_kp = np.asarray(self.profiles["AMO"].stiffness).copy()
            self._last_kd = np.asarray(self.profiles["AMO"].damping).copy()
            self._last_axes = dict.fromkeys(AXES, 0.0)
            self._neutral_since = None
            self._walking_unlocked = False
            self._transition = None
            self.video_metrics = deque(maxlen=4096)
            self.mode_samples = deque(maxlen=4096)
            self.events = deque(maxlen=4096)
            self.counts = Counter()
            self.model_max_s = {"AMO": 0.0, "UFO": 0.0}
            self.model_times = {name: deque(maxlen=4096) for name in self.model_max_s}
            self.policy.policy.set_tracking_enabled(True)
            for _ in range(10):
                self.policy.get_action(np.zeros(721, dtype=np.float32))
            self.policy.reset()
            reset_amo(self.amo, self.env.get_data())
            for _ in range(10):
                obs, _ = self.amo.get_observation(self.env.get_data(), {})
                self.amo.get_pd_target(obs)
            reset_amo(self.amo, self.env.get_data())
            self._source_play(False)
        except BaseException:
            if self.ctrl_manager is not None:
                self.ctrl_manager.shutdown()
            if self.env is not None:
                self.env.shutdown()
            raise

    def prepare(self, **kwargs):
        logger.warning("UFO/AMO disarmed: A=3s pose, Y=AMO, B=UFO, R2=stop")

    def reset(self):
        """Reset policy history only; never clear a fault or change arming state."""
        self.env.update()
        self._check_feedback(self.env.get_data())
        self.policy.reset()
        self.policy.policy.set_tracking_enabled(self.real_state in ("TO_UFO", "TRACKING"))
        reset_amo(self.amo, self.env.get_data(), self._last_q)

    def _state(self, state, reason):
        old = self.real_state
        self.real_state = state
        row = dict(at=self._clock(), step=self.timestep, previous=old, state=state, reason=reason)
        self.events.append(row)
        self.counts["transitions"] += 1
        logger.info("UFO/AMO %s -> %s: %s", old, state, reason)

    @staticmethod
    def _smooth(value):
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _begin(self, state, reason, data, duration):
        self._transition = dict(start=self._clock(), duration=duration,
                                q=self._last_q.copy(), kp=self._last_kp.copy(), kd=self._last_kd.copy())
        if state == "INITIALIZING":
            self._transition["q"] = np.asarray(data.dof_pos).copy()
        self._state(state, reason)

    def _begin_amo(self, reason, data):
        self._source_play(False)
        reset_amo(self.amo, data, self._last_q)
        self._walking_unlocked = False
        self._neutral_since = None
        self._begin("TO_AMO", reason, data, self.cfg.switch_seconds)

    def _controls(self, ctrl):
        if self._real_mode or "UnitreeCtrl" in ctrl:
            remote = ctrl.get("UnitreeCtrl", {})
            buttons = {name: bool(remote.get("buttons", {}).get(name, False)) for name in self.BUTTONS}
            axes = {name: float(remote.get("axes", {}).get(name, 0.0)) for name in AXES}
        else:
            for event in ctrl.get("KeyboardCtrl", {}).get("keyboard_event", []):
                key = event["name"].lower()
                if event["pressed"]:
                    self._keys.add(key)
                else:
                    self._keys.discard(key)
            mapping = {"A": "a", "Y": "y", "B": "b", "X": "x", "R2": "key.space"}
            buttons = {name: key in self._keys for name, key in mapping.items()}
            axes = dict(LeftX=float(("key.right" in self._keys) - ("key.left" in self._keys)),
                        LeftY=float(("key.up" in self._keys) - ("key.down" in self._keys)),
                        RightX=float(("l" in self._keys) - ("j" in self._keys)), RightY=0.0)
        if not np.isfinite(list(axes.values())).all():
            raise ValueError("Nonfinite remote axes")
        return buttons, {key: 0.0 if abs(value) <= self.cfg.stick_deadzone else float(np.clip(value, -1, 1))
                         for key, value in axes.items()}

    def _valid_source(self, source):
        captured = source.get("capture_time")
        return (source.get("valid", False) and not source.get("person_lost", False)
                and not source.get("ended", False) and captured is not None
                and 0 <= self._clock() - captured <= 5.0)

    def _amo_target(self, data, axes):
        started = time.monotonic()
        # Never pass FSM button events into AMO's own Y toggle.
        ctrl = {"UnitreeCtrl": {"axes": axes, "button_event": []}}
        obs, extras = self.amo.get_observation(data, ctrl)
        if obs.shape != (1043,) or not np.isfinite(obs).all():
            raise ValueError("Invalid AMO observation")
        target = self.amo.get_pd_target(obs)
        self.amo.post_step_callback([])
        self._timing("AMO", started)
        self._last_axes = axes.copy()
        self._amo_commands = extras["commands"].tolist()
        return target

    def _ufo_target(self, data, ctrl):
        started = time.monotonic()
        obs, _ = self.policy.get_observation(data, ctrl)
        target = self.policy.get_pd_target(obs)
        self.policy.post_step_callback([])
        self._timing("UFO", started)
        return target

    def _timing(self, name, started):
        duration = time.monotonic() - started
        self.model_times[name].append(duration)
        self.model_max_s[name] = max(self.model_max_s[name], duration)
        self.counts[name + "_inferences"] += 1

    def _send(self, target, mode, ctrl=None, blend=False):
        target = np.asarray(target, dtype=np.float64)
        profile = self.profiles[mode]
        kp, kd = np.asarray(profile.stiffness), np.asarray(profile.damping)
        if blend:
            transition = self._transition
            alpha = self._smooth((self._clock() - transition["start"]) / transition["duration"])
            target = (1 - alpha) * transition["q"] + alpha * target
            kp = (1 - alpha) * transition["kp"] + alpha * kp
            kd = (1 - alpha) * transition["kd"] + alpha * kd
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError(f"Invalid {mode} joint target")
        video = None
        if mode == "UFO":
            source = ctrl["LatentCtrl"]
            if not self._valid_source(source):
                # This exception is handled as a visual fallback, not a hardware fault.
                raise VisualExpired()
            now = self._clock()
            video = dict(step=self.timestep, mode=self.real_state, latent_frame=int(source["frame_id"]),
                         capture_to_pd_s=now - source["capture_time"],
                         produce_to_pd_s=None if source.get("produced_time") is None else now - source["produced_time"],
                         source_frame_id=source.get("source_frame_id"), session_id=source.get("session_id"),
                         sample_time_s=source.get("sample_time"))
        self.env.set_gains(kp, kd)
        if not self._real_mode:
            target = self.guard.filter(target, self.env.dof_pos)
        self.env.step(target)
        self._has_actuated = True
        self._last_q = self.guard.last_command.copy()
        self._last_kp, self._last_kd = kp.copy(), kd.copy()
        self.counts["commands_sent"] += 1
        self.mode_samples.append(dict(step=self.timestep, at=self._clock(), mode=self.real_state,
                                      axes=self._last_axes.copy(), commands=getattr(self, "_amo_commands", None)))
        if video is not None:
            self.video_metrics.append(video)
            self.counts["video_policy_steps"] += 1

    def _done(self):
        return self._clock() - self._transition["start"] >= self._transition["duration"]

    def _hold(self):
        if not self._has_actuated:
            return
        self.env.set_gains(self._last_kp, self._last_kd)
        if self._real_mode:
            sent = self.env.send_hold()
        else:
            target = self.guard.hold(self.env.dof_pos)
            self.env.step(target)
            sent = True
        if sent:
            self._last_q = self.guard.last_command.copy()
            self.counts["holds_sent"] += 1

    def _stop(self, reason, fault):
        if self.real_state != "STOPPED" or (fault and not self._fault_stop):
            self.stop_reason = reason
            self._state("STOPPED", reason)
            self.counts["faults" if fault else "operator_stops"] += 1
        self._fault_stop |= fault
        self._released_after_stop = False
        self._walking_unlocked = False
        self._source_play(False)
        if not self._real_mode and not fault and getattr(self.cfg, "sim_initial_support", False):
            self._sim_anchor = self.env.data.qpos[:7].copy()
            self._sim_supported = True

    def _advance(self, data, ctrl, buttons, axes):
        edge = lambda name: buttons[name] and not self._previous_buttons[name]
        source = ctrl["LatentCtrl"]
        zero = dict.fromkeys(AXES, 0.0)
        if self.real_state == "STOPPED":
            if not any(buttons.values()):
                self._released_after_stop = True
            if self._fault_stop or not self._released_after_stop or not edge("A"):
                self._hold()
                return
        # A blocked Y/B press must not accidentally fall through to a lower priority A.
        if edge("Y") or edge("X"):
            if self.real_state in ("PREPARED", "SETTLING", "TO_UFO", "TRACKING"):
                self._begin_amo("Y/X", data)
        elif edge("B"):
            if self.real_state == "AMO":
                if self._valid_source(source):
                    self._settle_axes = self._last_axes.copy()
                    self._begin("SETTLING", "B", data, self.cfg.settle_seconds)
                else:
                    self.counts["B_rejected"] += 1
                    logger.warning("UFO/AMO B rejected: no fresh visual latent")
        elif edge("A") and self.real_state in ("DISARMED", "STOPPED", "PREPARED"):
            self._source_play(False)
            self.policy.reset()
            reset_amo(self.amo, data)
            self._begin("INITIALIZING", "A", data, self.cfg.init_seconds)
        if self.real_state in ("TO_UFO", "TRACKING") and not self._valid_source(source):
            self._begin_amo("person lost / visual stale / ended", data)
        state = self.real_state
        if state == "DISARMED":
            if not self._real_mode and self.env.cfg_env.render:
                self.env.render_frame()
        elif state in ("INITIALIZING", "PREPARED"):
            self._send(self.profiles["AMO"].default_pos, "AMO", blend=state == "INITIALIZING")
            if state == "INITIALIZING" and self._done():
                self._state("PREPARED", "3s pose complete; awaiting Y")
        elif state in ("AMO", "TO_AMO"):
            if state == "AMO" and not self._walking_unlocked:
                if max(abs(value) for value in axes.values()) <= self.cfg.stick_deadzone:
                    if self._neutral_since is None:
                        self._neutral_since = self._clock()
                    self._walking_unlocked = self._clock() - self._neutral_since >= self.cfg.neutral_seconds
                else:
                    self._neutral_since = None
            command_axes = axes if state == "AMO" and self._walking_unlocked else zero
            self._send(self._amo_target(data, command_axes), "AMO", blend=state == "TO_AMO")
            if state == "TO_AMO" and self._done():
                self._state("AMO", "AMO blend complete; center sticks to walk")
                if not self._real_mode:
                    self._sim_supported = False
        elif state == "SETTLING":
            alpha = self._smooth((self._clock() - self._transition["start"]) / self.cfg.settle_seconds)
            self._send(self._amo_target(data, {key: value * (1 - alpha) for key, value in self._settle_axes.items()}), "AMO")
            if self._done():
                if self._valid_source(source):
                    self.policy.reset()
                    self._source_play(True)
                    self._begin("TO_UFO", "zero-speed settle complete", data, self.cfg.switch_seconds)
                else:
                    self._walking_unlocked = False
                    self._neutral_since = None
                    self._state("AMO", "visual became invalid during settling")
        elif state in ("TO_UFO", "TRACKING"):
            try:
                self._send(self._ufo_target(data, ctrl), "UFO", ctrl, blend=state == "TO_UFO")
            except VisualExpired:
                self._begin_amo("visual expired during inference", data)
                self._send(self._amo_target(data, zero), "AMO", blend=True)
                return
            if state == "TO_UFO" and self._done():
                self._state("TRACKING", "UFO blend complete")

    def step(self, dry_run=False):
        buttons = self._previous_buttons.copy()
        self._feedback_valid_cycle = False
        try:
            if not self._real_mode and self._sim_supported:
                # Models the external suspension required for the static A phase.
                # Release once AMO has completed its first active handoff.
                self.env.data.qpos[:7] = self._sim_anchor
                self.env.data.qvel[:6] = 0
            self.env.update()
            data = self.env.get_data()
            self._check_feedback(data)
            ctrl = self.ctrl_manager.get_ctrl_data(data)
            buttons, axes = self._controls(ctrl)
            if dry_run:
                return
            if buttons["R2"]:
                self._stop("R2", False)
                self._hold()
                return
            if self.cfg.do_safety_check and not self._attitude_is_valid():
                raise ValueError("Invalid or fallen base attitude")
            self._advance(data, ctrl, buttons, axes)
        except Exception as exc:
            logger.exception("UFO/AMO control fault")
            self._stop(str(exc), True)
            if self._feedback_valid_cycle:
                self._hold()
            else:
                raise
        finally:
            self._previous_buttons = buttons
            if not dry_run:
                self.timestep += 1
                self.counts["cycles"] += 1
                if self.ctrl_manager is not None:
                    self.ctrl_manager.post_step_callback({"COMMANDS": []})

    def safe_stop(self):
        if not self._has_actuated:
            return
        self._stop("runner exit", False)
        try:
            self.env.update()
            self._check_feedback(self.env.get_data())
            self._hold()
        except Exception:
            logger.warning("No fresh feedback for exit hold", exc_info=True)

    def metrics(self):
        return dict(state=self.real_state, fault_stopped=self._fault_stop, stop_reason=self.stop_reason,
                    counts=dict(self.counts), events=list(self.events), samples=list(self.mode_samples),
                    models={name: dict(max_s=self.model_max_s[name], retained_samples=len(values),
                                       p95_s=float(np.percentile(values, 95)) if values else None)
                            for name, values in self.model_times.items()})


class VisualExpired(RuntimeError):
    pass
