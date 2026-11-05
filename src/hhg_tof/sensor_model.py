"""TMF8829 zone geometry and distance correction.

TWO KINDS OF DISTANCE -- do not mix them up
-------------------------------------------
radial   What the chip reports: range along the zone's ray. A flat wall reads
         further at the corners than at the centre.
plane_z  Perpendicular depth along the optical axis.

The per-zone LUTs from ~/Desktop/TMF8829_cali map raw -> plane_z, because they
were fitted against a flat board at a known perpendicular distance. Measured in
the slot 7 48x32 calibration, d(raw)/d(true) is 1.02 at the centre and 1.28 at
the corner, against a vendor cosine factor of 1.26 there. So the LUT already
contains the cosine term, and dividing its output by that factor again would
shrink the edges of every view by about 20 %.

Ray geometry is ams' Tmf8829AppCommon.zCorrection, the same one the vendor uses
for its own XYZ output. Its half-angles (33.7 x 26.6 deg) match the datasheet
FOV (67.9 x 52.8 deg).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TICK_MM = 0.25  # TMF8829 result distances are UQ2: 1 count = 0.25 mm

# fp_mode -> (rows, cols) of the full focal plane
FP_MODE_GRID = {0: (8, 8), 1: (8, 8), 2: (16, 16), 3: (32, 32), 4: (32, 32), 5: (32, 48)}


def zone_tangents(rows: int, cols: int, mirror_columns: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Per-zone (x/z, y/z) slopes, each shaped (rows, cols).

    `mirror_columns` flips X. A mirror cannot be absorbed by a rigid extrinsic,
    and datasheet Figure 16 ("concept drawing only") is ambiguous about it, so
    it is settled by measurement: see `calibration.calibrate_mount`.
    """
    x = (np.arange(cols) - cols / 2 + 0.5) / (cols * 0.75)
    y = (np.arange(rows) - rows / 2 + 0.5) / rows
    xn, yn = np.meshgrid(-x if mirror_columns else x, y)
    return xn, yn


def distance_to_xyz(distance: np.ndarray, kind: str, xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    """Convert a (rows, cols) distance grid to (rows, cols, 3) sensor-frame points."""
    if kind == "radial":
        z = distance / np.sqrt(1.0 + xn**2 + yn**2)
    elif kind == "plane_z":
        z = distance
    else:
        raise ValueError(f"unknown distance kind {kind!r}")
    return np.stack([z * xn, z * yn, z], axis=-1)


class LutCorrection:
    """Per-zone raw -> plane_z correction from a TMF8829_cali calibration JSON.

    Raw values outside a zone's LUT come back as NaN. The calibration project's
    runtime falls back to a constant offset there, but that offset is only right
    at its 200 mm reference distance, and a scan is better off with a hole than
    with a point that is wrong by an unknown amount.
    """

    kind = "plane_z"

    def __init__(self, path: str | Path, rows: int, cols: int, expected_serial: str | None = None):
        data = json.loads(Path(path).read_text())
        # Filenames cannot be trusted: the 16x16 campaign wrote *_48x32_ha_*.json.
        grid = data["actual_grid"]
        if (grid["rows"], grid["cols"]) != (rows, cols):
            raise ValueError(
                f"{path}: calibration grid is {grid['rows']}x{grid['cols']}, "
                f"frames are {rows}x{cols}"
            )
        serial = str(data.get("device_serial") or "")
        if expected_serial and serial and serial != str(expected_serial):
            raise ValueError(f"{path}: calibration is for sensor {serial}, not {expected_serial}")
        self.path = str(path)
        self.serial = serial
        self._tables = []
        for r in range(rows):
            for c in range(cols):
                pairs = sorted((p["raw_mm"], p["true_mm"]) for p in data["lut"][r][c])
                self._tables.append(np.array(pairs, float).reshape(-1, 2))
        self.shape = (rows, cols)

    def apply(self, raw_mm: np.ndarray) -> np.ndarray:
        out = np.full(self.shape, np.nan)
        flat = raw_mm.ravel()
        for i, table in enumerate(self._tables):
            if len(table) >= 2 and table[0, 0] <= flat[i] <= table[-1, 0]:
                out.flat[i] = np.interp(flat[i], table[:, 0], table[:, 1])
        return out
