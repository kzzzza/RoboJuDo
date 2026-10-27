# MuJoCo reference visualization and recording

The shared renderer is part of `MujocoEnv`. It works with any policy that can
provide a complete reference pose for the same MuJoCo model used by the
environment. The policy puts `reference_qpos` in its observation `extras`:

```python
return obs, {"reference_qpos": pose_for_next_control_frame}
```

The value must be a finite, one-dimensional array of length `model.nq` in
MuJoCo `qpos` order, including the floating-base position and `wxyz` quaternion
when the model has a free root joint. The Pipeline passes it to the environment
before stepping the physics. The reference is then drawn against the resulting
simulated pose. A missing reference raises an error when comparison rendering
is requested; it is optional for actual-only recording.

`MujocoEnvCfg` provides `scene_style` (`model` or `light`), `show_reference`,
`reference_root_body`, `video_path`, `video_content` (`actual` or `comparison`),
`video_width`, and `video_height`. The default is the model's original visual
style, with reference and recording disabled. `light` creates an in-memory
near-white skybox and bright floor material, leaving the source XML, contact,
and friction unchanged. UFO M0 selects `light` by default. If a scene has
several moving objects, set `reference_root_body` to the robot's root body so
only that robot receives the translucent reference appearance.

For policies registered with the standard runner:

```bash
.venv/bin/python scripts/run_pipeline.py -c CONFIG --show-reference
.venv/bin/python scripts/run_pipeline.py -c CONFIG \
  --record-video recordings/run.mp4 --video-content comparison
```

Use `--video-content actual` when the policy does not expose a reference pose.
Recording is headless, runs through MuJoCo EGL, and uses FFmpeg to write an
H.264 MP4 at one frame per control step. Width and height default to 1280x720;
the encoded frame rate is `1 / (sim_dt * sim_decimation)`. Stop the continuous
runner with Ctrl+C; its cleanup finalizes the MP4. The recorder refuses to
overwrite an existing file. Window viewing and headless recording are separate
run modes.
