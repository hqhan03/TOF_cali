"""Rigid transforms. Units are mm and degrees, matching the FR5 controller.

FR5 poses are [x, y, z, rx, ry, rz] with fixed-axis (extrinsic) RPY, so
R = Rz @ Ry @ Rx. In scipy that is the lowercase "xyz" sequence.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_T(pose: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:6], degrees=True).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T: np.ndarray) -> list[float]:
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return [*map(float, T[:3, 3]), *map(float, rpy)]


def inv_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """Sensor pose at `eye` looking at `target`.

    The sensor frame is +Z along the optical axis, +X along increasing column,
    +Y along increasing row, so +Y points away from `up`.
    """
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    y = -np.asarray(up, float)
    y = y - (y @ z) * z
    norm = np.linalg.norm(y)
    if norm < 1e-9:
        raise ValueError("`up` is parallel to the viewing direction")
    y /= norm
    T = np.eye(4)
    T[:3, 0] = np.cross(y, z)
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = eye
    return T


def rot_z(T: np.ndarray, angle_deg: float) -> np.ndarray:
    """Roll a pose about its own Z axis."""
    R = np.eye(4)
    R[:3, :3] = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    return T @ R


def segment_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Shortest distance from point p to the segment a-b."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))
