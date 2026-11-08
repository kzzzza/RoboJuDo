"""Real CPU policies with deterministic state-machine time and no robot transport."""

from types import SimpleNamespace

from box import Box
import numpy as np
import pytest

import robojudo.environment
from robojudo.config.config_manager import ConfigManager
from robojudo.environment.utils.g1_dds import G1DdsError
from robojudo.environment.utils.real_target_guard import RealTargetGuard
import robojudo.pipeline.ufo_amo_pipeline as module


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeEnv:
    def __init__(self, cfg_env, device):
        self.cfg_env = cfg_env
        self.dof_cfg = cfg_env.dof
        self.joint_names = self.dof_cfg.joint_names
        self.position_limits = np.asarray(self.dof_cfg.position_limits)
        self.dof_pos = np.asarray(self.dof_cfg.default_pos, dtype=float)
        self.base_quat = np.array([0., 0., 0., 1.])
        self.target_guard = RealTargetGuard(self.position_limits, cfg_env.target_velocity_limits, .02, .5)
        self.visualizer = None
        self.commands = []
        self.holds = []
        self.stale = self.fail = False
        self.closed = False

    def update(self):
        if self.stale:
            raise G1DdsError("stale feedback")

    def get_data(self):
        return Box(dof_pos=self.dof_pos.copy(), dof_vel=np.zeros(29), base_quat=self.base_quat.copy(),
                   base_ang_vel=np.zeros(3))

    def set_gains(self, kp, kd):
        self.kp, self.kd = np.asarray(kp).copy(), np.asarray(kd).copy()

    def step(self, target):
        if self.fail:
            raise G1DdsError("write failed")
        target = self.target_guard.filter(target, self.dof_pos)
        self.commands.append((target.copy(), self.kp.copy(), self.kd.copy()))
        self.dof_pos = target.copy()

    def send_hold(self):
        if self.stale or not self.commands:
            return False
        self.target_guard.hold(self.dof_pos)
        self.holds.append(self.dof_pos.copy())
        return True

    def shutdown(self):
        self.closed = True


class Controls:
    def __init__(self, clock):
        self.clock = clock
        self.buttons = {}
        self.axes = dict.fromkeys(module.AXES, 0.0)
        self.valid = True
        self.lost = False
        self.frame = 0
        self.controllers = {"LatentCtrl": SimpleNamespace(inst=SimpleNamespace(set_playing=lambda _: None))}

    def get_ctrl_data(self, data):
        self.frame += 1
        z = np.zeros(256, dtype=np.float32)
        z[0] = 16
        return Box(UnitreeCtrl=dict(buttons=self.buttons, axes=self.axes, button_event=[]),
                   LatentCtrl=dict(valid=self.valid, person_lost=self.lost, ended=False, z=z,
                                   capture_time=self.clock(), produced_time=self.clock(), age=0,
                                   frame_id=self.frame, session_id=1))

    def post_step_callback(self, data):
        pass

    def shutdown(self):
        pass


@pytest.fixture
def pipeline(monkeypatch):
    clock = Clock()
    controls = Controls(clock)
    cfg = ConfigManager("g1_ufo_video_real_workstation_amo").get_cfg()
    cfg.env.unitree.net_if = "lo"
    cfg.auto_debug_mode = False
    monkeypatch.setenv("UFO_REAL_ROBOT_OK", "1")
    monkeypatch.setattr(robojudo.environment, "UnitreeG1DdsEnv", FakeEnv)
    monkeypatch.setattr(module, "CtrlManager", lambda *args, **kwargs: controls)
    p = module.UfoAmoPipeline(cfg, clock=clock)
    p.test_clock, p.test_controls = clock, controls
    yield p
    p.ctrl_manager.shutdown()
    p.env.shutdown()


def cycle(p, buttons=None, seconds=.02):
    p.test_controls.buttons = buttons or {}
    p.step()
    p.test_clock.now += seconds


def run_for(p, seconds):
    for _ in range(round(seconds / .02) + 1):
        cycle(p)


def prepared(p):
    cycle(p, {"A": True})
    run_for(p, 3.02)
    assert p.real_state == "PREPARED"


def amo(p):
    prepared(p)
    cycle(p, {"Y": True})
    run_for(p, .52)
    assert p.real_state == "AMO"


def ufo(p):
    amo(p)
    cycle(p, {"B": True})
    run_for(p, 1.06)
    assert p.real_state == "TRACKING"


def test_startup_dry_run_and_prepare_never_send(pipeline):
    p = pipeline
    p.prepare()
    p.step(dry_run=True)
    run_for(p, 1)
    p.safe_stop()
    assert not p.env.commands and not p.env.holds
    assert not p.counts["AMO_inferences"] and not p.counts["UFO_inferences"]


def test_3s_pose_profiles_and_no_queued_y(pipeline):
    p = pipeline
    cycle(p, {"A": True})
    for _ in range(150):
        cycle(p, {"Y": True})
    assert p.real_state == "INITIALIZING" or p.real_state == "PREPARED"
    cycle(p, {"Y": True})
    assert p.real_state == "PREPARED"
    assert not p.counts["AMO_inferences"]
    np.testing.assert_allclose(p.env.commands[-1][0], p.profiles["AMO"].default_pos)
    for name in ["left_shoulder_pitch_joint", "right_elbow_joint"]:
        i = p.env.joint_names.index(name)
        j = p.cfg.amo_policy.obs_dof.joint_names.index(name)
        assert p.env.commands[-1][1][i] == p.cfg.amo_policy.obs_dof.stiffness[j]
    cycle(p)
    cycle(p, {"Y": True})
    assert p.real_state == "TO_AMO"


def test_full_sequence_single_writer_and_gain_continuity(pipeline):
    p = pipeline
    ufo(p)
    assert p.counts["UFO_inferences"] > 0
    assert len(p.env.commands) == p.counts["commands_sent"]
    before = p.env.commands[-1]
    cycle(p, {"Y": True})
    np.testing.assert_allclose(p.env.commands[-1][0], before[0])
    np.testing.assert_allclose(p.env.commands[-1][1], before[1])
    run_for(p, .52)
    assert p.real_state == "AMO"
    np.testing.assert_allclose(p.env.commands[-1][1], p.profiles["AMO"].stiffness)
    q = np.stack([r[0] for r in p.env.commands])
    assert np.all(np.abs(np.diff(q, axis=0)) <= p.guard.max_delta + 1e-7)


def test_walking_requires_center_then_b_settles(pipeline):
    p = pipeline
    amo(p)
    p.test_controls.axes["LeftY"] = .5
    run_for(p, .6)
    assert np.allclose(p.amo.policy.cmd[:3], 0)
    p.test_controls.axes["LeftY"] = 0
    run_for(p, .52)
    assert p._walking_unlocked
    p.test_controls.axes["LeftY"] = .5
    cycle(p)
    assert p.amo.policy.cmd[0] > 0
    cycle(p, {"B": True})
    assert p.real_state == "SETTLING"
    run_for(p, .52)
    assert p.real_state == "TO_UFO"
    np.testing.assert_allclose(p.amo.policy.cmd[:3], 0)


@pytest.mark.parametrize("reason", ["person", "stale"])
def test_visual_failure_returns_to_amo_without_fault_or_auto_resume(pipeline, reason):
    p = pipeline
    ufo(p)
    previous = p.counts["UFO_inferences"]
    p.test_controls.lost = reason == "person"
    p.test_controls.valid = False
    cycle(p)
    assert p.real_state == "TO_AMO" and not p._fault_stop
    assert p.counts["UFO_inferences"] == previous
    run_for(p, .52)
    p.test_controls.lost = False
    p.test_controls.valid = True
    run_for(p, .6)
    assert p.real_state == "AMO"


def test_r2_preempts_all_buttons_and_requires_full_rearm(pipeline):
    p = pipeline
    ufo(p)
    before = dict(p.counts)
    cycle(p, {"R2": True, "Y": True, "B": True, "A": True})
    assert p.real_state == "STOPPED" and not p._fault_stop
    assert p.counts["UFO_inferences"] == before["UFO_inferences"]
    assert p.counts["AMO_inferences"] == before["AMO_inferences"]
    cycle(p, {"A": True})
    assert p.real_state == "STOPPED"
    cycle(p)
    cycle(p, {"B": True})
    assert p.real_state == "STOPPED"
    cycle(p)
    prepared(p)
    cycle(p, {"B": True})
    assert p.real_state == "PREPARED"
    cycle(p)
    cycle(p, {"Y": True})
    run_for(p, .52)
    assert p.real_state == "AMO"


@pytest.mark.parametrize("failure", ["feedback", "write", "nan", "attitude"])
def test_faults_latch_and_never_rearm(pipeline, failure, monkeypatch):
    p = pipeline
    amo(p)
    if failure == "feedback":
        p.env.stale = True
        with pytest.raises(G1DdsError):
            cycle(p)
    else:
        if failure == "write":
            p.env.fail = True
        elif failure == "nan":
            monkeypatch.setattr(p.amo, "get_pd_target", lambda _: np.full(29, np.nan))
        else:
            p.env.base_quat = np.array([0., np.sqrt(.5), 0., np.sqrt(.5)])
        cycle(p)
    assert p._fault_stop and p.real_state == "STOPPED"
    before = len(p.env.commands)
    p.env.stale = p.env.fail = False
    p.env.base_quat = np.array([0., 0., 0., 1.])
    p.reset()
    assert p._fault_stop and p.real_state == "STOPPED"
    cycle(p)
    cycle(p, {"A": True, "Y": True, "B": True})
    assert len(p.env.commands) == before
    assert p.real_state == "STOPPED"


def test_reset_clears_amo_history_and_reserved_y_never_toggles(pipeline):
    p = pipeline
    amo(p)
    p.amo.policy.cmd[7] = 1
    p.amo.policy.proprio_history_buf.append(np.full(p.amo.policy.n_proprio, 999))
    module.reset_amo(p.amo, p.env.get_data(), p._last_q)
    assert p.amo.policy.cmd[7] == 0
    assert not np.any(np.asarray(p.amo.policy.proprio_history_buf) == 999)
    np.testing.assert_allclose(p.amo.policy.gait_cycle, [.25, .25])
    cycle(p, {"Y": True})
    assert p.amo.policy.cmd[7] == 0


def test_b_without_z_is_rejected_but_amo_keeps_running(pipeline):
    p = pipeline
    amo(p)
    p.test_controls.valid = False
    cycle(p, {"B": True})
    assert p.real_state == "AMO" and p.counts["B_rejected"] == 1
    assert not p._fault_stop


def test_amo_fills_all_arms_and_wrists_by_name(pipeline):
    p = pipeline
    amo(p)
    target = p._amo_target(p.env.get_data(), dict.fromkeys(module.AXES, 0.))
    for i, name in enumerate(p.env.joint_names):
        if name in p.cfg.amo_policy.action_dof.joint_names:
            j = p.cfg.amo_policy.action_dof.joint_names.index(name)
            expected = (p.amo.policy.last_action[j] * p.amo.policy.action_scale
                        + p.cfg.amo_policy.action_dof.default_pos[j])
            assert target[i] == pytest.approx(expected)
        else:
            assert target[i] == p.profiles["AMO"].default_pos[i]


def test_raw_nonfinite_amo_output_is_rejected_before_clipping(pipeline, monkeypatch):
    import torch

    p = pipeline
    amo(p)
    monkeypatch.setattr(p.amo.policy, "model", lambda *args: torch.full((1, 15), float("inf")))
    cycle(p)
    assert p._fault_stop and "Invalid AMO model action" in p.stop_reason


def test_transition_rechecks_visual_age_before_sending(pipeline, monkeypatch):
    p = pipeline
    ufo(p)
    original = p._ufo_target

    def late(data, ctrl):
        target = original(data, ctrl)
        ctrl["LatentCtrl"]["capture_time"] -= 6
        return target

    monkeypatch.setattr(p, "_ufo_target", late)
    cycle(p)
    assert p.real_state == "TO_AMO" and not p._fault_stop


def test_stick_deadzone_is_zero_velocity(pipeline):
    p = pipeline
    amo(p)
    run_for(p, .52)
    p.test_controls.axes["LeftY"] = .08
    cycle(p)
    assert p._walking_unlocked
    np.testing.assert_allclose(p.amo.policy.cmd[:3], 0)
