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
  git apply ../patches/mujoco_viewer_mujoco3.patch)
uv pip install --python .venv/bin/python -e third_party/mujoco_viewer
# Copy FBcprAuxModel.onnx and backward_encoder.onnx into assets/models/g1/ufo_m0/
.venv/bin/python scripts/run_ufo_m0.py \
  --motion-path ~/data/lafan/lafan_g1_npz/walk1_subject1.npz \
  --max-frames 1500 --render
```

The viewer opens in a desktop session and follows the robot during playback.
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
  action, four prior observation frames grouped by feature, and `z`. Its
  output is multiplied by 5 and clipped to `[-5, 5]`; the PD target is
  `default_pos + action * 0.25 * effort_limit / kp`.
- Simulation runs at 200 Hz physics and 50 Hz policy control. The policy
  default pose, stiffness, damping, effort limits, and output scaling come
  from the UFO M0 G1 configuration. The package's ONNX graphs already contain
  their own observation normalization.

The finite runner is intended for repeatable motion checks. The registered
`g1_ufo_m0` config also works with `scripts/run_pipeline.py` for a continuous
headless run; after the selected reference ends, it keeps the last reference.
