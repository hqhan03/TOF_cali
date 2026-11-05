"""Binary PLY read/write, so the core pipeline needs no 3D library."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(path: str | Path, points: np.ndarray, normals: np.ndarray | None = None,
              views: np.ndarray | None = None) -> None:
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if normals is not None:
        fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")]
    if views is not None:
        fields += [("view_count", "<u1")]
    data = np.zeros(len(points), dtype=fields)
    data["x"], data["y"], data["z"] = points.T
    if normals is not None:
        data["nx"], data["ny"], data["nz"] = normals.T
    if views is not None:
        data["view_count"] = np.minimum(views, 255)
    names = {"<f4": "float", "<u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    header += [f"property {names[t]} {n}" for n, t in fields]
    header.append("end_header\n")
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        f.write(data.tobytes())


def read_ply(path: str | Path) -> np.ndarray:
    """Read a PLY from write_ply() as a structured array."""
    with open(path, "rb") as f:
        fields, count = [], 0
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            words = line.split()
            if words[:2] == ["element", "vertex"]:
                count = int(words[2])
            elif words[:1] == ["property"]:
                fields.append((words[2], {"float": "<f4", "uchar": "<u1"}[words[1]]))
        return np.frombuffer(f.read(), dtype=fields, count=count)
