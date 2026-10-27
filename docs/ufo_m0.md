# UFO M0 G1 sim2sim

This deployment uses the original UFO G1 29-DoF M0 checkpoint's two exported
ONNX graphs: `backward_encoder.onnx` (B) and `FBcprAuxModel.onnx` (actor).
Place both ONNX files locally in `assets/models/g1/ufo_m0/`; model binaries are
excluded from Git. The small `FBcprAuxModel.meta.json` stays in the repository.
The training checkpoint is not needed. The source robot XML is copied to
`assets/robots/g1/g1_ufo_m0.xml`.
The sim XML also supplies the ground plane and the G1 mode-15 joint armature
and dry friction that UFO's training simulator creates separately.

## Setup and run

From the repository root:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
git submodule update --init third_party/mujoco_viewer
(cd third_party/mujoco_viewer && \
  git apply ../patches/mujoco_viewer.patch && \
  git apply ../patches/mujoco_viewer_mujoco3.patch && \
  git apply ../patches/mujoco_viewer_scene_hook.patch)
uv pip install --python .venv/bin/python -e third_party/mujoco_viewer
# Copy FBcprAuxModel.onnx and backward_encoder.onnx into assets/models/g1/ufo_m0/
.venv/bin/python scripts/run_ufo_m0.py \
  --motion-path ~/data/lafan/lafan_g1_npz/walk1_subject1.npz \
  --max-frames 1500 --render
```

The viewer opens in a desktop session and follows the robot during playback.
Add `--show-reference` to display the motion as a translucent blue robot at
the same world position as the simulated robot. The reference follows the
policy's next-frame target, so the two poses are compared after each control
step. The scene uses the reusable light-background preset.

To save a video without a desktop display, omit `--render` and run:

```bash
.venv/bin/python scripts/run_ufo_m0.py \
  --motion-path ~/data/lafan/lafan_g1_npz/walk1_subject1.npz \
  --max-frames 1500 --record-video recordings/ufo_walk.mp4 \
  --video-content comparison
```

Use `--video-content actual` to record only the simulated robot. Recording
requires the `ffmpeg` executable and an EGL-capable MuJoCo installation. The
CLI sets `MUJOCO_GL=egl` before importing MuJoCo. An existing output file is
never overwritten. See [MuJoCo rendering](mujoco_rendering.md) for the
interface used by other policies.

The example shows 30 seconds at 50 Hz. The runner defaults to the first 500
frames of that walk when `--max-frames` is omitted. `--start-frame`
selects another segment; `--motion-path` selects another retargeted G1 NPZ.
For example, `short_dance1_subject1.npz` has 497 frames. Each run reports
fall status, mean joint error, root-height error, and local body-position error.
By default, the runner stops if the simulated root drops below 0.35 m. Pass
`--no-stop-on-fall` to keep rendering until the selected frame limit; the
reported `fallen` value still records whether the threshold was crossed.
Omit `--render` for headless checks. The viewer needs an available desktop
display; closing its window ends the visual display while the finite check
continues to its frame limit.

## Data and model conventions

- The LAFAN G1 NPZ uses a kinematic-depth joint order and a different body
  order from the UFO XML. `UfoM0Motion` explicitly maps both to the UFO order.
  Source body quaternions are `wxyz`; RoboJuDo observations use `xyzw`.
- The NPZ provides 30 bodies. The B reference input adds UFO's virtual
  `head_link` from the torso position, orientation, linear velocity, and
  angular velocity. B consumes reference `state[64]` and
  `privileged_state[463]`, producing `z[256]` for the next reference frame.
- The actor consumes `actor_obs[721]`: current state, previous normalized
  action, four prior observation frames grouped by feature in newest-first
  order, and `z`. History is read before the current frame is inserted. Its
  output is multiplied by 5 and clipped to `[-5, 5]`; the PD target is
  `default_pos + action * 0.25 * effort_limit / kp`.
- Simulation runs at 200 Hz physics and 50 Hz policy control. The policy
  default pose, stiffness, damping, effort limits, and output scaling come
  from the UFO M0 G1 configuration. The package's ONNX graphs already contain
  their own observation normalization.

The finite runner is intended for repeatable motion checks. The registered
`g1_ufo_m0` config also works with `scripts/run_pipeline.py` for a continuous
headless run; after the selected reference ends, it keeps the last reference.

## G1 Real deployment

`g1_ufo_m0_real` uses RoboJuDo's existing `UnitreeCppEnv` driver, the same M0
Actor/B pair and the selected LAFAN G1 motion. The real configuration runs at
50 Hz, uses `eth0`, and enables position, velocity-slew and base-attitude
checks. The checked-in `artifact_manifest.json` records SHA256 hashes of the
local model pair and metadata; ONNX binaries remain ignored by Git. UFO's
`deploy` branch uses different ONNX files, so its real-robot result does not
validate these particular M0 binaries.

Run the non-actuating preflight before launching control:

```bash
python scripts/check_ufo_m0_real.py
UFO_REAL_ROBOT_OK=1 python scripts/run_pipeline.py -c g1_ufo_m0_real
```

The preflight checks model hashes, ONNX inference, the selected motion,
`unitree_cpp` availability, and the configured network interface. It does not
construct a robot control environment or publish commands. On a development
machine without the robot interface, use `--offline` to check models and
motion only. Install the existing RoboJuDo real-robot dependencies following
[Unitree setup](unitree_setup.md); no UFO `g1_interface` installation is needed.
The default motion path is
`~/data/lafan/lafan_g1_npz/walk1_subject1.npz`; set the policy configuration
to the copied G1 motion path if the robot uses a different location.

After launch, the pipeline is disarmed and sends no joint target until the
operator presses **A**. A ramps from measured joints to the default standing
pose for 500 cycles (10 seconds). **R1** enables policy output with the first
reference frame fixed. **B** starts reference playback. **X** stops playback
and resets it to the first reference frame. At the end of the selected motion,
the robot returns to the standing pose over five seconds.

**R2** takes priority over all other buttons: it stops inference and sends a
measured-position hold when feedback is valid and control has already started.
If R2 is pressed before standing initialization completes, release it and
press A again; no policy action can start before the standing phase completes.
To rearm after an R2 stop from the ready state,
release R2, R1 and B, then press **R1+B** together. An observation, inference,
target or base-attitude fault cannot be rearmed in the same process; inspect
the fault and restart. Keyboard interruption or a severe frame drop also
attempts a fresh measured-position hold before driver shutdown.

The software path can be checked with hardware-free tests and sim2sim. Real
tracking quality and stop behavior still require a supported, supervised
robot test; this document does not claim that the M0 Real adaptation has
already passed one.
