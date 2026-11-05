"""Stricter multi-peak reconstruction from saved histograms. No hardware.

`reconstruct` takes every peak that clears a threshold. This adds what makes the default
pipeline clean, in histogram terms, and models the tail a strong target leaves behind:

- Response peeling. The shape one surface leaves in a histogram is learned from the data,
  the strongest peak is subtracted with that shape, and only what remains can be a
  second surface. That removes tail artefacts, and two surfaces need no dip between them.
- Temporal: a peak must be present in most of the measurements of the view.
- Spatial: a neighbouring zone must see a surface at about the same range. A zone
  straddling an edge reports a blend that belongs to neither side.
- Cylindrical region of interest, so its boundary cannot draw a false curved surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hhg_tof import session as session_io
from hhg_tof.geometry import pose_to_T, transform_points
from hhg_tof.processing import mount_T
from hhg_tof.sensor_model import distance_to_xyz, zone_tangents

from .peaks import BinScale, centroid
from .reconstruct import FAR, NEAR, SINGLE, Cloud, seen_from_other_views, session_scale
from .session import read_view

BEFORE, AFTER = 2, 12  # template span around the peak, in bins


@dataclass
class RefineRules:
    min_range_mm: float = 60.0
    stray_percentile: float = 25.0     # the median removes a plate that fills half the view
    min_counts: float = 8.0            # per measurement
    noise_sigmas: float = 5.0
    min_ratio: float = 0.05
    min_separation_bins: int = 3       # 2 splits oblique surfaces in two: lid scatter 6.5 -> 9.6 mm
    model_margin: float = 0.3          # residual must beat this share of the subtracted model
    max_peaks: int = 2
    min_present: float = 0.6           # share of measurements showing the peak
    neighbour_mm: float = 20.0
    neighbour_rel: float = 0.06
    min_neighbours: int = 2


def remove_stray(histograms: np.ndarray, percentile: float) -> np.ndarray:
    shared = np.percentile(histograms.reshape(-1, histograms.shape[-1]), percentile, axis=0)
    return np.clip(histograms - shared, 0.0, None)


def learn_response(cleaned_views: list[np.ndarray], first_bin: int, min_height: float) -> np.ndarray:
    """Median peak-normalised shape around the strongest bin, bins -BEFORE..+AFTER.

    A median over many zones ignores the minority that also see a second surface.
    """
    shapes = []
    for cleaned in cleaned_views:
        flat = cleaned.reshape(-1, cleaned.shape[-1])
        peaks = flat[:, first_bin:].argmax(axis=1) + first_bin
        for h, p in zip(flat, peaks):
            if h[p] >= min_height and BEFORE <= p < len(h) - AFTER:
                shapes.append(h[p - BEFORE:p + AFTER + 1] / h[p])
    if len(shapes) < 50:
        raise ValueError(f"only {len(shapes)} zones to learn the response from")
    shape = np.median(shapes, axis=0)
    shape[BEFORE] = 1.0
    return shape


def peel(cleaned: np.ndarray, raw: np.ndarray, response: np.ndarray, first_bin: int,
         rules: RefineRules, measurements: int) -> list[tuple[float, float, int]]:
    """Up to max_peaks (bin position, counts per measurement, integer bin), strongest first."""
    residual = cleaned.copy()
    model = np.zeros_like(cleaned)
    noise = float(np.median(raw[first_bin:]))
    height = max(rules.min_counts * measurements, noise + rules.noise_sigmas * np.sqrt(max(noise, 1.0)))
    found: list[tuple[float, float, int]] = []
    for _ in range(rules.max_peaks):
        limit = height + rules.model_margin * model
        excess = np.where(residual >= limit, residual, 0.0)
        excess[:first_bin] = 0.0
        for _, _, b in found:
            excess[max(b - rules.min_separation_bins + 1, 0):b + rules.min_separation_bins] = 0.0
        if excess.max() <= 0:
            break
        p = int(excess.argmax())
        value = float(residual[p])
        if found and value < rules.min_ratio * found[0][1] * measurements:
            break
        found.append((centroid(residual, p), value / measurements, p))
        lo, hi = p - BEFORE, min(p + AFTER + 1, len(residual))
        at = max(lo, 0)
        part = value * response[at - lo:hi - lo]
        residual[at:hi] = np.clip(residual[at:hi] - part, 0.0, None)
        model[at:hi] += part
    return found


def present_enough(stack: np.ndarray, r: int, c: int, b: int, rules: RefineRules) -> bool:
    """Whether the peak shows in most measurements, not just in the sum."""
    window = stack[:, r, c, max(b - 1, 0):b + 2].sum(axis=1).astype(float)
    return float((window >= 0.4 * window.mean()).mean()) >= rules.min_present and window.mean() > 0


def neighbour_support(ranges: dict, r: int, c: int, mm: float, rules: RefineRules) -> int:
    tolerance = max(rules.neighbour_mm, rules.neighbour_rel * mm)
    support = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and any(abs(other - mm) <= tolerance for other in ranges.get((r + dr, c + dc), ())):
                support += 1
    return support


def view_points(data: dict, T: np.ndarray, cfg: dict, scale: BinScale, response: np.ndarray,
                first_bin: int, rules: RefineRules):
    stack = data["histogram"]
    summed = stack.sum(axis=0).astype(float)
    cleaned = remove_stray(summed, rules.stray_percentile)
    measurements = stack.shape[0]
    zones: dict = {}
    for r in range(summed.shape[0]):
        for c in range(summed.shape[1]):
            found = peel(cleaned[r, c], summed[r, c], response, first_bin, rules, measurements)
            kept = [(float(scale.to_mm(pos)), height) for pos, height, b in found
                    if present_enough(stack, r, c, b, rules)]
            if kept:
                zones[r, c] = sorted(kept)
    ranges = {key: [mm for mm, _ in value] for key, value in zones.items()}
    xn, yn = zone_tangents(*summed.shape[:2], cfg["sensor"]["mirror_columns"])
    bias = cfg["sensor"]["range_bias_mm"]
    points, kinds, counts = [], [], []
    for (r, c), peaks in zones.items():
        good = [(mm, h) for mm, h in peaks if neighbour_support(ranges, r, c, mm, rules) >= rules.min_neighbours]
        for rank, (mm, height) in enumerate(good):
            points.append(distance_to_xyz(np.array(mm - bias), "radial", xn[r, c], yn[r, c]))
            kinds.append(SINGLE if len(good) == 1 else NEAR if rank == 0 else FAR)
            counts.append(height)
    if not points:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    return transform_points(T, np.array(points)), np.array(kinds), np.array(counts)


def build(session: str | Path, rules: RefineRules | None = None, roi_radius_mm: float = 100.0,
          radius_mm: float = 12.0, need_views: int = 2) -> Cloud:
    session = Path(session)
    rules = rules or RefineRules()
    cfg = session_io.read_meta(session)["config"]
    views = session_io.read_views(session)
    scale = session_scale(session, [v.index for v in views])
    first_bin = int(np.ceil(scale.zero_bin + rules.min_range_mm / scale.mm_per_bin))
    data = [read_view(session, v.index) for v in views]
    cleaned = [remove_stray(d["histogram"].sum(axis=0).astype(float), rules.stray_percentile) for d in data]
    response = learn_response(cleaned, first_bin, 30 * rules.min_counts * data[0]["histogram"].shape[0] / 8)
    mount = mount_T(cfg)
    P, K, V, N = [], [], [], []
    for view, d in zip(views, data):
        p, k, n = view_points(d, pose_to_T(view.tcp_pose) @ mount, cfg, scale, response, first_bin, rules)
        P.append(p), K.append(k), N.append(n), V.append(np.full(len(p), view.index))
    points, kinds, which, counts = np.vstack(P), np.concatenate(K), np.concatenate(V), np.concatenate(N)
    stats = {"peaks_kept": len(points), "double_peak_points": int((kinds != SINGLE).sum()),
             "response": np.round(response, 3).tolist()}
    proc, scan = cfg["processing"], cfg["scan"]
    center = np.asarray(scan["center_mm"], float)
    keep = np.linalg.norm(points[:, :2] - center[:2], axis=1) <= roi_radius_mm
    if proc["floor_z_mm"] is not None:
        keep &= points[:, 2] > proc["floor_z_mm"] + proc["floor_margin_mm"]
    points, kinds, which, counts = points[keep], kinds[keep], which[keep], counts[keep]
    stats["inside_roi"] = len(points)
    keep = seen_from_other_views(points, which, radius_mm, need_views)
    points, kinds, which, counts = points[keep], kinds[keep], which[keep], counts[keep]
    stats["confirmed_by_other_views"] = len(points)
    return Cloud(points, kinds, which, counts, scale, stats)
