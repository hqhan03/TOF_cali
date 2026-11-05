"""Ray-cast TMF8829 simulator, for testing the pipeline against known geometry.

Each zone is sampled by a small grid of sub-rays and their ranges are averaged.
That reproduces the mixed-pixel artefact of a real zone straddling an edge,
where the reported distance belongs to neither surface, which is what the edge
filter in processing.py has to remove.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..geometry import pose_to_T
from ..sensor_model import zone_tangents
from .base import Frame


class Plane:
    def __init__(self, z: float = 0.0, normal=(0.0, 0.0, 1.0)):
        self.n = np.asarray(normal, float) / np.linalg.norm(normal)
        self.point = self.n * z

    def intersect(self, o, d):
        denom = d @ self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.point - o) @ self.n) / denom
        return np.where(t > 0, t, np.inf)

    def sdf(self, p):
        return np.abs((p - self.point) @ self.n)


class Sphere:
    def __init__(self, center, radius: float):
        self.c = np.asarray(center, float)
        self.r = float(radius)

    def intersect(self, o, d):
        oc = o - self.c
        b = d @ oc
        disc = b * b - (oc @ oc - self.r**2)
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        return np.where((disc >= 0) & (t > 0), t, np.inf)

    def sdf(self, p):
        return np.abs(np.linalg.norm(p - self.c, axis=-1) - self.r)

    def surface_samples(self, n: int, rng) -> np.ndarray:
        v = rng.normal(size=(n, 3))
        return self.c + self.r * v / np.linalg.norm(v, axis=1, keepdims=True)


class Box:
    def __init__(self, center, size):
        self.c = np.asarray(center, float)
        self.h = np.asarray(size, float) / 2

    def intersect(self, o, d):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (self.c - self.h - o) / d
            t2 = (self.c + self.h - o) / d
        near = np.minimum(t1, t2).max(axis=1)
        far = np.maximum(t1, t2).min(axis=1)
        return np.where((near <= far) & (near > 0), near, np.inf)

    def sdf(self, p):
        q = np.abs(p - self.c) - self.h
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return outside + np.abs(np.minimum(q.max(axis=-1), 0.0))

    def surface_samples(self, n: int, rng) -> np.ndarray:
        p = rng.uniform(-1, 1, size=(n, 3)) * self.h
        axis = rng.integers(0, 3, n)
        p[np.arange(n), axis] = rng.choice([-1, 1], n) * self.h[axis]
        return self.c + p


@dataclass
class SimNoise:
    sigma_mm: float = 1.0
    sigma_rel: float = 0.004
    dropout: float = 0.02
    range_bias_mm: float = 0.0


class SimulatedSensor:
    def __init__(self, robot, scene: list, T_tcp_sensor: np.ndarray, rows: int = 32, cols: int = 48,
                 noise: SimNoise | None = None, mirror_columns: bool = False, subrays: int = 3,
                 max_range_mm: float = 2000.0, seed: int = 0):
        self.robot, self.scene, self.T_tcp_sensor = robot, scene, T_tcp_sensor
        self.rows, self.cols = rows, cols
        self.noise = noise or SimNoise()
        self.max_range_mm = max_range_mm
        self.serial = "SIM"
        self._rng = np.random.default_rng(seed)
        self._frame_number = 0
        xn, yn = zone_tangents(rows, cols, mirror_columns)
        # Sub-ray offsets span one zone; zone pitch is constant in tangent space.
        pitch_x = 1.0 / (cols * 0.75)
        pitch_y = 1.0 / rows
        offsets = (np.arange(subrays) + 0.5) / subrays - 0.5
        ox, oy = np.meshgrid(offsets * pitch_x, offsets * pitch_y)
        dirs = np.stack([
            xn[..., None] + ox.ravel(), yn[..., None] + oy.ravel(),
            np.ones((rows, cols, subrays * subrays)),
        ], axis=-1)
        self._rays = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    def open(self) -> None: ...
    def close(self) -> None: ...

    def _range_image(self) -> np.ndarray:
        T = pose_to_T(self.robot.status().tcp_pose) @ self.T_tcp_sensor
        d = self._rays.reshape(-1, 3) @ T[:3, :3].T
        t = np.full(len(d), np.inf)
        for shape in self.scene:
            t = np.minimum(t, shape.intersect(T[:3, 3], d))
        t = t.reshape(self.rows, self.cols, -1)
        t[t > self.max_range_mm] = np.nan
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(t, axis=-1)

    def read_frames(self, count: int, discard: int = 0) -> list[Frame]:
        base = self._range_image()
        n = self.noise
        frames = []
        for _ in range(count):
            d = base + n.range_bias_mm
            d = d + self._rng.normal(size=d.shape) * (n.sigma_mm + n.sigma_rel * np.nan_to_num(d))
            d = np.where(self._rng.random(d.shape) < n.dropout, 0.0, d)
            self._frame_number += 1
            frames.append(Frame(np.nan_to_num(d).astype(np.float32), self._frame_number, time.time()))
        return frames
