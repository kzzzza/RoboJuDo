"""Shared MuJoCo scene styling, reference geometry, and headless recording."""

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np


def load_model(xml_path: str, scene_style: str) -> mujoco.MjModel:
    if scene_style == "model":
        return mujoco.MjModel.from_xml_path(xml_path)
    if scene_style != "light":
        raise ValueError(f"Unknown MuJoCo scene style: {scene_style}")

    # Compile an in-memory visual variant; the source XML and physics settings
    # (including floor contact and friction) remain untouched.
    spec = mujoco.MjSpec.from_file(xml_path)
    skyboxes = [tex for tex in spec.textures if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX]
    if skyboxes:
        skybox = skyboxes[0]
        skybox.file = ""
        skybox.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
        skybox.rgb1[:] = [0.96, 0.97, 0.99]
        skybox.rgb2[:] = [0.96, 0.97, 0.99]
    else:
        spec.add_texture(
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
            rgb1=[0.96, 0.97, 0.99],
            rgb2=[0.96, 0.97, 0.99],
            width=256,
            height=1536,
        )
    floor_material = "robojudo_light_floor"
    if any(mat.name == floor_material for mat in spec.materials):
        raise ValueError(f"Scene already defines reserved material {floor_material}")
    spec.add_material(name=floor_material, rgba=[0.96, 0.96, 0.96, 1.0], emission=0.65)
    for geom in spec.worldbody.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.material = floor_material
    return spec.compile()


class ReferenceGhost:
    """Adds a translucent robot pose to a MuJoCo scene without simulating it."""

    def __init__(self, model: mujoco.MjModel, root_body: str | None = None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.option = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        self.has_pose = False

        if root_body is None:
            free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            if not len(free_joints) and model.nbody < 2:
                raise ValueError("Reference visualization requires a robot body")
            root_id = int(model.jnt_bodyid[free_joints[0]]) if len(free_joints) else 1
        else:
            root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
            if root_id < 1:
                raise ValueError(f"Unknown reference robot root body: {root_body}")
        self.body_ids = {root_id}
        for body_id in range(root_id + 1, model.nbody):
            if int(model.body_parentid[body_id]) in self.body_ids:
                self.body_ids.add(body_id)

    def set_pose(self, qpos):
        if qpos is None:
            self.has_pose = False
            return
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (self.model.nq,) or not np.all(np.isfinite(values)):
            raise ValueError(f"reference_qpos must be finite and have shape ({self.model.nq},)")
        self.data.qpos[:] = values
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.has_pose = True

    def append(self, scene: mujoco.MjvScene):
        if not self.has_pose:
            return
        first = scene.ngeom
        mujoco.mjv_addGeoms(
            self.model, self.data, self.option, self.perturb,
            mujoco.mjtCatBit.mjCAT_ALL, scene,
        )
        if scene.ngeom == scene.maxgeom:
            raise RuntimeError("MuJoCo scene geometry capacity exhausted by reference pose")
        for index in range(first, scene.ngeom):
            geom = scene.geoms[index]
            if (geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or geom.objid < 0
                    or int(self.model.geom_bodyid[geom.objid]) not in self.body_ids):
                geom.matid = -1
                geom.texid = -1
                geom.rgba[3] = 0.0
                geom.transparent = True
                continue
            geom.matid = -1
            geom.texid = -1
            geom.rgba[:] = [0.57, 0.79, 1.0, 0.35]
            geom.transparent = True


class MujocoVideoRecorder:
    """Encode one offscreen RGB frame per control step to an H.264 MP4."""

    def __init__(self, model, path, fps, width, height, ghost=None):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Headless video recording requires the ffmpeg executable")
        self.path = Path(path).expanduser()
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("Video output path must end in .mp4")
        if self.path.exists():
            raise FileExistsError(self.path)
        if width < 2 or height < 2 or width % 2 or height % 2 or fps <= 0:
            raise ValueError("Video width and height must be positive even numbers; fps must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ghost = ghost
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 3.0
        self.camera.elevation = -10.0
        self.camera.azimuth = 180.0
        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
        ]
        # Ctrl+C belongs to the runner; FFmpeg must stay alive until close()
        # closes its input and lets it finalize the MP4 container.
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
        self.frames = 0

    def write_frame(self, data):
        self.camera.lookat[:] = data.qpos[:3]
        self.renderer.update_scene(data, camera=self.camera)
        if self.ghost is not None:
            if not self.ghost.has_pose:
                raise ValueError("comparison video requires reference_qpos on every frame")
            self.ghost.append(self.renderer.scene)
        frame = self.renderer.render()
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self.process.stderr.read().decode(errors="replace")) from exc
        self.frames += 1

    def close(self):
        if self.renderer is None:
            return
        self.renderer.close()
        self.renderer = None
        self.process.stdin.close()
        error = self.process.stderr.read().decode(errors="replace")
        code = self.process.wait(timeout=30)
        self.process.stderr.close()
        if code:
            raise RuntimeError(f"ffmpeg exited with code {code}: {error}")
