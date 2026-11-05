"""Is the sensor's range scale right? Move straight up and down over a flat desk.

The arm knows how far it moved, so the change in reported range can be compared with a
true displacement, with no target of known size. A slope of 1 means the range error is a
constant. Anything else is an error that grows with distance. The pass goes up, down
and back to the start, so each height is measured twice and repeatability comes free.
"""

from __future__ import annotations

import numpy as np

from hhg_tof.geometry import pose_to_T
from hhg_tof.planning import Viewpoint


def heights(start_z: float, top_z: float, bottom_z: float, step_mm: float) -> list[float]:
    """start -> top -> bottom -> start, in steps of at most `step_mm`."""
    def leg(a: float, b: float) -> list[float]:
        count = max(int(np.ceil(abs(b - a) / step_mm)), 1)
        return np.linspace(a, b, count + 1)[1:].tolist()

    return [start_z, *leg(start_z, top_z), *leg(top_z, bottom_z), *leg(bottom_z, start_z)]


def vertical_views(pose, zs: list[float]) -> list[Viewpoint]:
    views = []
    for index, z in enumerate(zs):
        target = list(pose)
        target[2] = float(z)
        views.append(Viewpoint(index, 0.0, 0.0, pose_to_T(target), 0.0, target))
    return views


def central_range(frames: np.ndarray, rows: int = 8, cols: int = 12) -> float:
    """Median range of the central zones, over all frames of one view. Ignores dropouts."""
    r, c = frames.shape[1] // 2, frames.shape[2] // 2
    block = frames[:, r - rows // 2:r + rows // 2, c - cols // 2:c + cols // 2]
    return float(np.median(block[block > 0]))


def fit_scale(tcp_z: np.ndarray, measured_mm: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(slope, offset, residuals) of measured range against true height change."""
    slope, offset = np.polyfit(tcp_z - tcp_z.mean(), measured_mm, 1)
    return float(slope), float(offset), measured_mm - (slope * (tcp_z - tcp_z.mean()) + offset)


def tilt_plan(tilts_deg=(15.0, 30.0, 45.0), azimuths_deg=(0.0, 90.0, 180.0, 270.0)) -> list[tuple[float, float]]:
    """(tilt, azimuth) sequence: out and back along each direction, level in between."""
    plan = [(0.0, 0.0)]
    for azimuth in azimuths_deg:
        plan += [(t, azimuth) for t in tilts_deg]
        plan += [(t, azimuth) for t in reversed(tilts_deg[:-1])]
        plan.append((0.0, azimuth))
    return plan


def tilt_views(level_sensor: np.ndarray, mount: np.ndarray, plan) -> list[Viewpoint]:
    """Sensor stays at one point and turns; the TCP goes wherever that needs."""
    from scipy.spatial.transform import Rotation

    from hhg_tof.geometry import T_to_pose, inv_T

    views = []
    for index, (tilt, azimuth) in enumerate(plan):
        turn = Rotation.from_euler("z", azimuth, degrees=True) * Rotation.from_euler("x", tilt, degrees=True) \
            * Rotation.from_euler("z", -azimuth, degrees=True)
        T = level_sensor.copy()
        T[:3, :3] = turn.as_matrix() @ level_sensor[:3, :3]
        views.append(Viewpoint(index, tilt, azimuth, T, 0.0, T_to_pose(T @ inv_T(mount))))
    return views


def plane_range(T_sensor: np.ndarray, rays: np.ndarray, plane_z: float) -> np.ndarray:
    """Radial range to the plane z = plane_z along sensor-frame rays (..., 3), any length."""
    directions = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    world = directions @ T_sensor[:3, :3].T
    return (plane_z - T_sensor[2, 3]) / world[..., 2]
