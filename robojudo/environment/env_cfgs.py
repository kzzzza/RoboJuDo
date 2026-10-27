from typing import Literal

import numpy as np
from pydantic import model_validator

from robojudo.config import Config
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg, ZedOdometryCfg


class EnvCfg(Config):
    env_type: str  # name of the environment class
    is_sim: bool = False

    urdf: str | None = None
    xml: str
    body_names: list[str] | None = None

    dof: DoFConfig

    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False
    """Whether to update info from fk"""
    torso_name: str = "torso_link"
    """Name of the torso link, used in fk info extraction"""

    born_place_align: bool = True
    """Whether to align the born place to zero position and heading"""


class MujocoEnvCfg(EnvCfg):
    env_type: str = "MujocoEnv"
    is_sim: bool = True
    # ====== ENV CONFIGURATION ======
    sim_duration: float = 60.0
    sim_dt: float = 0.001
    sim_decimation: int = 20

    visualize_extras: bool = True  # TODO: remove
    render: bool = True
    show_reference: bool = False
    reference_root_body: str | None = None
    scene_style: Literal["model", "light"] = "model"
    video_path: str | None = None
    video_content: Literal["actual", "comparison"] = "actual"
    video_width: int = 1280
    video_height: int = 720

    random_heading: bool = False
    """Randomize the robot's yaw heading on each spawn/reborn (useful for testing heading alignment)."""


class RobotEnvCfg(EnvCfg):
    env_type: str = "DummyEnv"
    is_sim: bool = False
    # ====== ENV CONFIGURATION ======
    act: bool = True

    odometry_type: Literal["NONE", "DUMMY", "ZED"] = "NONE"
    zed_cfg: ZedOdometryCfg | None = None
    """ZED odometry config, if odometry_type is "ZED", this must be set"""

    @model_validator(mode="after")
    def check_zed_config(self):
        if self.odometry_type == "ZED" and self.zed_cfg is None:
            raise ValueError("zed_cfg must be set if odometry_type is 'ZED'")
        return self


class UnitreeEnvCfg(RobotEnvCfg):
    """
    Configuration for Unitree Robot environment.
    """

    class UnitreeCfg(Config):
        """Unitree SDK configuration"""

        net_if: str = "eth0"
        """network interface to communicate with the robot"""

        robot: Literal["h1", "g1"]
        msg_type: Literal["hg", "go"]
        control_mode: str = "position"
        hand_type: Literal["Dex-3", "Inspire", "NONE"] = "NONE"

        lowcmd_topic: str = "rt/lowcmd"
        lowstate_topic: str = "rt/lowstate"

        enable_odometry: bool = False
        sport_state_topic: str = "rt/odommodestate"

        control_dt: float = 0.02
        """control command dt"""

    env_type: str = "UnitreeEnv"  # For unitree_sdk2py
    # env_type: str = "UnitreeCppEnv" # For unitree_cpp
    """UnitreeEnv for unitree_sdk2py, UnitreeCppEnv for unitree_cpp, check README for more details"""

    unitree: UnitreeCfg

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "DUMMY"  # pyright: ignore[reportIncompatibleVariableOverride]

    joint2motor_idx: list[int] | None = None
    """Mapping from env dof to motor index, None for direct mapping"""
    target_velocity_limits: list[float] | None = None
    """Enable final real target guarding when set; values follow env joint order."""
    target_slew_safety_factor: float = 0.5
    weak_motor: list[int] = []

    hand_retarget: None = None  # TODO

    @model_validator(mode="after")
    def check_joint2motor_idx(self):
        if self.joint2motor_idx is not None and len(self.joint2motor_idx) != self.dof.num_dofs:
            raise ValueError("joint2motor_idx length must match dof.num_dofs")
        if self.target_velocity_limits is not None:
            if len(self.target_velocity_limits) != self.dof.num_dofs:
                raise ValueError("target_velocity_limits length must match dof.num_dofs")
            if any(not np.isfinite(value) or value <= 0 for value in self.target_velocity_limits):
                raise ValueError("target_velocity_limits must be finite and positive")
            if not np.isfinite(self.target_slew_safety_factor) or not 0 < self.target_slew_safety_factor <= 1:
                raise ValueError("target_slew_safety_factor must be in (0, 1]")
        return self
