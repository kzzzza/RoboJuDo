"""Optional final position and slew guard for real robot PD targets."""

import numpy as np


class RealTargetGuard:
    def __init__(self, position_limits, velocity_limits, control_dt: float, slew_factor: float):
        self.position_limits = np.asarray(position_limits, dtype=np.float64)
        self.velocity_limits = np.asarray(velocity_limits, dtype=np.float64)
        self.num_dofs = self.velocity_limits.size
        if self.position_limits.shape != (self.num_dofs, 2):
            raise ValueError("position limits must have shape (num_dofs, 2)")
        if not np.all(np.isfinite(self.position_limits)) or np.any(
            self.position_limits[:, 0] >= self.position_limits[:, 1]
        ):
            raise ValueError("position limits must be finite and ordered")
        if not np.all(np.isfinite(self.velocity_limits)) or np.any(self.velocity_limits <= 0):
            raise ValueError("velocity limits must be finite and positive")
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if not np.isfinite(slew_factor) or not 0 < slew_factor <= 1:
            raise ValueError("slew_factor must be in (0, 1]")
        self.max_delta = self.velocity_limits * control_dt * slew_factor
        self.last_command: np.ndarray | None = None

    def _checked(self, values, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.num_dofs,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite {self.num_dofs}-joint vector")
        return result

    def filter(self, target, measured) -> np.ndarray:
        target = self._checked(target, "target")
        measured = self._checked(measured, "measured joint positions")
        baseline = self.last_command
        if baseline is None:
            baseline = measured
        target = np.clip(target, self.position_limits[:, 0], self.position_limits[:, 1])
        limited = baseline + np.clip(target - baseline, -self.max_delta, self.max_delta)
        limited = np.clip(limited, self.position_limits[:, 0], self.position_limits[:, 1])
        self.last_command = limited.copy()
        return limited

    def hold(self, measured) -> np.ndarray:
        """Use measured joints directly for a stop, bypassing the slew ramp."""
        measured = self._checked(measured, "measured joint positions")
        target = np.clip(measured, self.position_limits[:, 0], self.position_limits[:, 1])
        self.last_command = target.copy()
        return target
