"""Common frame type and sensor interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Frame:
    distance_mm: np.ndarray               # (rows, cols) float32 radial range, 0 = no target
    frame_number: int
    host_time: float
    snr: Optional[np.ndarray] = None      # (rows, cols), None if the source has no SNR
    temperature_c: Optional[float] = None


class Sensor(Protocol):
    serial: str

    def open(self) -> None: ...
    def close(self) -> None: ...
    def read_frames(self, count: int, discard: int = 0) -> list[Frame]: ...


def expected_grid_or_raise(frame: Frame, grid: tuple[int, int]) -> None:
    if frame.distance_mm.shape != tuple(grid):
        raise RuntimeError(
            f"sensor returned {frame.distance_mm.shape}, config expects {tuple(grid)}. "
            "Fix the firmware profile or sensor.grid; a wrong grid gives wrong rays."
        )
