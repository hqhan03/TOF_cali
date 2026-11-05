"""Live frame processing for the viewer. No drawing and no hardware access here.

Each frame goes through `processing.view_to_cloud`, so a live point passes the same
filters as a point in a saved session. Two honest limits follow from what is known:

- Without a mount, points stay in the sensor frame. A guessed mount would draw a
  plausible cloud in the wrong place.
- A frame takes about 190 ms, so one taken while the arm moves is smeared and its
  pose is ambiguous (see capture.py). It is shown, but never added to the map.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import pose_to_T, transform_points
from .processing import mount_T, view_to_cloud
from .sensor_model import LutCorrection
from .sensors.base import Frame
from .session import View

FRAME_WINDOW_S = 0.35     # one frame plus transport, the span that must be motionless
MAX_POSE_AGE_S = 0.5
STILL_TOL_MM = 0.5
STILL_TOL_DEG = 0.2

# fairino5_v6.urdf (frcobot_ros2), mm and deg: (joint origin xyz, rpy), axis always +Z.
# Checked on this robot: FK orientation matches the controller to 0.01 deg, and the
# flange -> TCP difference is the pure 136.4 mm Z offset of tool 1.
FR5_CHAIN = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 152.0), (90.0, 0.0, 0.0)),
    ((-425.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-395.01, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 102.1), (90.0, 0.0, 0.0)),
    ((0.0, 0.0, 102.0), (-90.0, 0.0, 0.0)),
)


def arm_transforms(joints_deg) -> list[np.ndarray]:
    """Base-frame transform of each link frame, ending at the flange."""
    T, out = np.eye(4), []
    for (xyz, rpy), angle in zip(FR5_CHAIN, joints_deg):
        origin = np.eye(4)
        origin[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        origin[:3, 3] = xyz
        spin = np.eye(4)
        spin[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
        T = T @ origin @ spin
        out.append(T)
    return out


def arm_polyline(joints_deg) -> np.ndarray:
    """(7, 3) points from the base to the flange."""
    return np.vstack([np.zeros(3)] + [T[:3, 3] for T in arm_transforms(joints_deg)])


class PoseHistory:
    def __init__(self, keep_s: float = 3.0):
        self.keep_s = keep_s
        self.samples: deque[tuple[float, np.ndarray]] = deque()

    def add(self, t: float, pose) -> None:
        self.samples.append((t, np.asarray(pose, float)))
        while self.samples and t - self.samples[0][0] > self.keep_s:
            self.samples.popleft()

    def latest_before(self, t: float) -> tuple[float, np.ndarray] | None:
        for sample in reversed(self.samples):
            if sample[0] <= t:
                return sample
        return None

    def still(self, t0: float, t1: float) -> bool:
        """True only if samples span [t0, t1] and the pose held. Missing data is not still."""
        inside = [p for t, p in self.samples if t0 <= t <= t1]
        if len(inside) < 2 or self.samples[0][0] > t0:
            return False
        poses = np.array(inside)
        spread = poses.max(axis=0) - poses.min(axis=0)
        return float(spread[:3].max()) <= STILL_TOL_MM and float(spread[3:].max()) <= STILL_TOL_DEG


class VoxelMap:
    """Bounded point map: newest point per voxel, oldest voxels dropped first."""

    def __init__(self, voxel_mm: float, max_points: int = 200_000):
        self.voxel_mm, self.max_points = voxel_mm, max_points
        self._voxels: dict[tuple[int, int, int], np.ndarray] = {}

    def add(self, points: np.ndarray) -> None:
        for key, point in zip(map(tuple, np.floor(points / self.voxel_mm).astype(int).tolist()), points):
            self._voxels.pop(key, None)
            self._voxels[key] = point
        while len(self._voxels) > self.max_points:
            del self._voxels[next(iter(self._voxels))]

    def clear(self) -> None:
        self._voxels.clear()

    @property
    def points(self) -> np.ndarray:
        return np.array(list(self._voxels.values())).reshape(-1, 3)


@dataclass
class LiveFrame:
    frame: Frame
    sensor_points: np.ndarray             # (N, 3) sensor frame, filtered
    T_base_sensor: np.ndarray | None      # None when the frame cannot be placed
    base_points: np.ndarray | None
    still: bool
    placement: str                        # why it was or was not placed, shown to the user
    valid_percent: float = field(default=0.0)


class LiveProcessor:
    def __init__(self, cfg: dict, correction: LutCorrection | None = None):
        self.cfg, self.correction = cfg, correction
        self.T_mount = mount_T(cfg) if cfg["mount"]["xyz_mm"] and cfg["mount"]["rpy_deg"] else None
        self.expected_tool = cfg["robot"]["expected_tool"]
        self.poses = PoseHistory()
        self.tool: int | None = None
        self.map = VoxelMap(cfg["processing"]["voxel_mm"])

    def add_robot(self, message: dict) -> None:
        self.poses.add(message["t"], message["tcp_pose"])
        self.tool = message.get("tool")

    def _sensor_points(self, frame: Frame) -> np.ndarray:
        snr = None if frame.snr is None else frame.snr[None]
        zero = np.zeros(6)
        view = View(0, frame.distance_mm[None], zero, zero, zero, snr)
        return view_to_cloud(view, self.cfg, np.eye(4), self.correction).points

    def _pose_for(self, t: float) -> tuple[np.ndarray | None, str]:
        if self.T_mount is None:
            return None, "no mount: sensor frame only"
        sample = self.poses.latest_before(t)
        if sample is None or t - sample[0] > MAX_POSE_AGE_S:
            return None, "no fresh robot pose"
        if self.tool is not None and self.tool != self.expected_tool:
            return None, f"active tool {self.tool}, mount is for tool {self.expected_tool}"
        return sample[1], "placed"

    def add_frame(self, frame: Frame) -> LiveFrame:
        t = frame.host_time
        points = self._sensor_points(frame)
        valid = 100.0 * float((frame.distance_mm > 0).mean())
        pose, placement = self._pose_for(t)
        still = self.poses.still(t - FRAME_WINDOW_S, t)
        if pose is None:
            return LiveFrame(frame, points, None, None, still, placement, valid)
        T = pose_to_T(pose) @ self.T_mount
        base = transform_points(T, points)
        if still:
            self.map.add(base)
        else:
            placement = "moving: shown, not mapped"
        return LiveFrame(frame, points, T, base, still, placement, valid)
