"""Saving histograms next to a normal hhg_tof session, without changing how one is taken.

`HistogramFrames` looks like any `hhg_tof` sensor, so the scan runs through the normal
capture path with all its checks, and each view additionally leaves a hist_NNN.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hhg_tof.sensors.base import Frame

from .evm_hist import HistogramSensor


class HistogramFrames:
    def __init__(self, sensor: HistogramSensor, session: Path):
        self.sensor, self.session = sensor, Path(session)
        self.view = 0

    @property
    def serial(self) -> str:
        return self.sensor.serial

    def open(self) -> None:
        self.sensor.open()

    def close(self) -> None:
        self.sensor.close()

    def read_frames(self, count: int, discard: int = 0) -> list[Frame]:
        measurements = self.sensor.read(count, max(discard, 1))
        write_view(self.session, self.view, measurements)
        self.view += 1
        return [Frame(m.distance_mm[..., 0], m.frame_number, m.host_time, m.snr[..., 0]) for m in measurements]


def write_view(session: Path, index: int, measurements) -> None:
    np.savez_compressed(
        Path(session) / f"hist_{index:03d}.npz",
        histogram=np.stack([m.histogram for m in measurements]),   # (M, 32, 48, 64)
        reference=np.stack([m.reference for m in measurements]),   # (M, 12, 4, 64)
        distance_mm=np.stack([m.distance_mm for m in measurements]),
        snr=np.stack([m.snr for m in measurements]),
    )


def read_view(session: Path, index: int) -> dict:
    with np.load(Path(session) / f"hist_{index:03d}.npz") as data:
        return {key: data[key] for key in data.files}
