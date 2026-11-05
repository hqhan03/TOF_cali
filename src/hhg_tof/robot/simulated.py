"""Perfect-kinematics stand-in for the FR5. Knows nothing about reachability.

`pose_noise_mm` models repeatability error between command and arrival.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import RobotStatus


class SimulatedRobot:
    def __init__(self, start_pose: Sequence[float], pose_noise_mm: float = 0.0, seed: int = 1):
        self.pose = [float(v) for v in start_pose]
        self.pose_noise_mm = pose_noise_mm
        self.commands: list[list[float]] = []
        self._rng = np.random.default_rng(seed)

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...

    def status(self) -> RobotStatus:
        return RobotStatus(list(self.pose), [0.0] * 6, robot_mode=1, tool_id=0, wobj_id=0, simulated=True)

    def solve_ik(self, pose, ref_joints=None) -> list[float]:
        return [0.0] * 6

    def prepare_motion(self, speed_percent: int) -> None: ...

    def move_linear(self, pose, joints, vel, ovl) -> None:
        self.commands.append(list(pose))
        self.pose = list(pose)
        self.pose[:3] = list(np.array(pose[:3]) + self._rng.normal(size=3) * self.pose_noise_mm)

    def wait_settled(self, timeout_s, tol_mm, hold_s) -> None: ...
    def stop(self) -> None: ...
    def restore(self) -> None: ...
