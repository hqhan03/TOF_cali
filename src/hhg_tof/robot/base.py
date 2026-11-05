"""Robot interface shared by the real FR5 and the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


class RobotError(RuntimeError):
    pass


class MotionRefused(RobotError):
    """A guard refused a motion before anything was sent."""


class SafetyStop(RobotError):
    """The arm was stopped during a move."""


class IkFailed(RobotError):
    pass


@dataclass
class RobotStatus:
    tcp_pose: list[float]
    joints: list[float]
    robot_mode: int = 0               # 0 auto, 1 manual
    main_code: int = 0
    sub_code: int = 0
    motion_done: Optional[int] = 1
    emergency_stop: Optional[int] = 0
    collision_state: Optional[int] = 0
    tool_id: Optional[int] = None
    wobj_id: Optional[int] = None
    simulated: bool = False

    @property
    def fault(self) -> Optional[str]:
        if self.emergency_stop:
            return "emergency stop is pressed"
        if self.collision_state:
            return "collision detected"
        if self.main_code or self.sub_code:
            return f"controller error main={self.main_code} sub={self.sub_code}"
        return None


class Robot(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def status(self) -> RobotStatus: ...
    def solve_ik(self, pose: Sequence[float], ref_joints: Sequence[float] | None = None) -> list[float]: ...
    def prepare_motion(self, speed_percent: int) -> None: ...
    def move_linear(self, pose: Sequence[float], joints: Sequence[float], vel: float, ovl: float) -> None: ...
    def wait_settled(self, timeout_s: float, tol_mm: float, hold_s: float) -> None: ...
    def stop(self) -> None: ...
    def restore(self) -> None: ...
