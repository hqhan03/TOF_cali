"""On-disk scan session: raw data only, so processing can be re-run freely.

<session>/
    session.json      config, sensor identity, sequence of views
    view_000.npz      frames (F, rows, cols) raw radial mm, snr, poses
    ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class View:
    index: int
    frames: np.ndarray                 # (F, rows, cols) float32, 0 = no target
    tcp_pose: np.ndarray               # (6,) mean of before/after
    joints: np.ndarray
    planned_pose: np.ndarray
    snr: Optional[np.ndarray] = None   # (F, rows, cols)


def new_session_dir(root: str | Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{stamp}_{name}"
    path.mkdir(parents=True)
    return path


def write_view(session: Path, view: View) -> None:
    arrays = dict(frames=view.frames, tcp_pose=view.tcp_pose, joints=view.joints,
                  planned_pose=view.planned_pose)
    if view.snr is not None:
        arrays["snr"] = view.snr
    np.savez_compressed(session / f"view_{view.index:03d}.npz", **arrays)


def write_meta(session: Path, meta: dict) -> None:
    tmp = session / "session.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(session / "session.json")  # an abort must not leave half a file


def read_meta(session: str | Path) -> dict:
    return json.loads((Path(session) / "session.json").read_text())


def read_views(session: str | Path) -> list[View]:
    views = []
    for path in sorted(Path(session).glob("view_*.npz")):
        with np.load(path) as z:
            views.append(View(
                index=int(path.stem.split("_")[1]), frames=z["frames"], tcp_pose=z["tcp_pose"],
                joints=z["joints"], planned_pose=z["planned_pose"],
                snr=z["snr"] if "snr" in z.files else None,
            ))
    if not views:
        raise FileNotFoundError(f"no views in {session}")
    return views
