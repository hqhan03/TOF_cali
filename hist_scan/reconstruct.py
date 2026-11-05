"""Histograms -> point cloud with every surface a zone saw, not just the strongest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from hhg_tof import session as session_io
from hhg_tof.geometry import pose_to_T, transform_points
from hhg_tof.processing import mount_T
from hhg_tof.sensor_model import distance_to_xyz, zone_tangents

from .peaks import BinScale, PeakRules, fit_bin_scale, remove_stray, zone_peaks
from .session import read_view

SINGLE, NEAR, FAR = 0, 1, 2
KIND_NAMES = {SINGLE: "single peak", NEAR: "nearer of two", FAR: "farther of two"}


@dataclass
class Cloud:
    points: np.ndarray    # (N, 3) base frame
    kind: np.ndarray      # (N,) SINGLE, NEAR or FAR
    view: np.ndarray      # (N,)
    counts: np.ndarray    # (N,) counts per measurement
    scale: BinScale
    stats: dict = field(default_factory=dict)


def session_scale(session: Path, indices: list[int]) -> BinScale:
    histograms, chip = [], []
    for index in indices:
        data = read_view(session, index)
        histograms.append(data["histogram"].sum(axis=0).reshape(-1, data["histogram"].shape[-1]))
        chip.append(np.median(data["distance_mm"][..., 0], axis=0).ravel())
    return fit_bin_scale(np.vstack(histograms), np.concatenate(chip))


def view_points(data: dict, T: np.ndarray, cfg: dict, scale: BinScale, rules: PeakRules):
    """(points, kinds, counts) of one view."""
    summed = data["histogram"].sum(axis=0).astype(float)
    cleaned = remove_stray(summed)
    measurements = data["histogram"].shape[0]
    xn, yn = zone_tangents(*summed.shape[:2], cfg["sensor"]["mirror_columns"])
    bias = cfg["sensor"]["range_bias_mm"]
    points, kinds, counts = [], [], []
    for r in range(summed.shape[0]):
        for c in range(summed.shape[1]):
            found = zone_peaks(cleaned[r, c], summed[r, c], scale, rules, measurements)
            for rank, (mm, height) in enumerate(found):
                xyz = distance_to_xyz(np.array(mm - bias), "radial", xn[r, c], yn[r, c])
                points.append(xyz)
                kinds.append(SINGLE if len(found) == 1 else NEAR if rank == 0 else FAR)
                counts.append(height)
    if not points:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    return transform_points(T, np.array(points)), np.array(kinds), np.array(counts)


def seen_from_other_views(points: np.ndarray, views: np.ndarray, radius_mm: float, need: int) -> np.ndarray:
    """True where at least `need` other views have a point within `radius_mm`.

    A real surface is there whichever way it is seen from. A multipath ghost moves with
    the viewpoint, so it finds no support.
    """
    tree = cKDTree(points)
    keep = np.zeros(len(points), bool)
    for i, near in enumerate(tree.query_ball_point(points, radius_mm)):
        keep[i] = len(set(views[near].tolist()) - {int(views[i])}) >= need
    return keep


def build(session: str | Path, rules: PeakRules | None = None, radius_mm: float = 12.0,
          need_views: int = 2) -> Cloud:
    session = Path(session)
    rules = rules or PeakRules()
    cfg = session_io.read_meta(session)["config"]
    views = session_io.read_views(session)
    indices = [v.index for v in views]
    scale = session_scale(session, indices)
    mount = mount_T(cfg)
    all_points, all_kinds, all_views, all_counts = [], [], [], []
    for view in views:
        p, k, c = view_points(read_view(session, view.index), pose_to_T(view.tcp_pose) @ mount, cfg, scale, rules)
        all_points.append(p), all_kinds.append(k), all_counts.append(c)
        all_views.append(np.full(len(p), view.index))
    points, kinds = np.vstack(all_points), np.concatenate(all_kinds)
    which, counts = np.concatenate(all_views), np.concatenate(all_counts)
    stats = {"peaks_found": len(points), "double_peak_points": int((kinds != SINGLE).sum())}

    proc, scan = cfg["processing"], cfg["scan"]
    keep = np.ones(len(points), bool)
    if proc["floor_z_mm"] is not None:
        keep &= points[:, 2] > proc["floor_z_mm"] + proc["floor_margin_mm"]
    if scan["center_mm"] is not None and proc["roi_radius_mm"]:
        keep &= np.linalg.norm(points - np.asarray(scan["center_mm"]), axis=1) <= proc["roi_radius_mm"]
    points, kinds, which, counts = points[keep], kinds[keep], which[keep], counts[keep]
    stats["inside_roi"] = len(points)
    if len(views) > need_views and len(points):
        keep = seen_from_other_views(points, which, radius_mm, need_views)
        points, kinds, which, counts = points[keep], kinds[keep], which[keep], counts[keep]
    stats["confirmed_by_other_views"] = len(points)
    return Cloud(points, kinds, which, counts, scale, stats)


def write_ply(path: Path, cloud: Cloud) -> None:
    header = ["ply", "format ascii 1.0", f"element vertex {len(cloud.points)}",
              "property float x", "property float y", "property float z",
              "property uchar kind", "property uchar view", "property float counts", "end_header"]
    rows = [f"{x:.2f} {y:.2f} {z:.2f} {k} {v} {c:.1f}" for (x, y, z), k, v, c
            in zip(cloud.points, cloud.kind, cloud.view, cloud.counts)]
    Path(path).write_text("\n".join(header + rows) + "\n")
