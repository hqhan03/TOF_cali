"""Sensor mount calibration from scans of one flat surface.

Every zone of every view must land on the same plane in the base frame, so the
mount (6 DOF), the plane (3), and optionally a constant range bias are solved
together from the residual to that plane. The target needs no known pose: a
table or a board. What it does need is views that differ in tilt AND roll: with
one fixed roll on a look-at orbit, the plane normal stays in the sensor's Y-Z
plane, so the X offset and the mirror are mathematically invisible. The solver
then still returns numbers, which is why `unobserved` exists.

The same residual settles the column mirror. A wrong mirror cannot be absorbed
by a rigid mount across differently tilted views, so both hypotheses are solved
and compared. That needs no reliance on the datasheet's concept drawing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import T_to_pose, pose_to_T
from .processing import edge_mask, temporal_filter
from .sensor_model import LutCorrection, distance_to_xyz, zone_tangents
from .session import View

# Range bias is meaningless once a LUT has already corrected the distances.
PARAM_NAMES = ["rx", "ry", "rz", "x", "y", "z", "plane_a", "plane_b", "plane_d", "range_bias"]
MIN_ZONES = 200
MIN_VIEWS = 6
SEARCH_ZONES_PER_VIEW = 40
SEARCH_RANDOM_ROTATIONS = 300
SEARCH_REFINED = 16
PLANE_INLIER_MM = 8.0
PLANE_MIN_FRACTION = 0.5
PLANE_RANSAC_TRIES = 200
MAX_LAYERS = 2
MIN_LAYER_ZONES = 40
LAYER_STARTS = 40
LAYER_GAP_MM = 30.0
MAX_SIGMA = {"x": 2.0, "y": 2.0, "z": 2.0, "rx": 0.01, "ry": 0.01, "rz": 0.01}  # mm, rad


@dataclass
class CalibrationResult:
    mirror_columns: bool
    xyz_mm: list[float]
    rpy_deg: list[float]
    range_bias_mm: float
    rms_mm: float
    rms_before_mm: float
    sigma: dict[str, float]
    plane_normal: list[float]
    plane_offset_mm: float
    zones_used: int


def dominant_plane(points: np.ndarray, rng, min_fraction: float = PLANE_MIN_FRACTION) -> np.ndarray:
    """Mask of the points on the largest plane, or all False if no plane dominates.

    Needs no mount: a plane is a plane in the sensor frame too. This lets the target
    be a real desk, where objects would otherwise pull every view off the plane.
    """
    best = np.zeros(len(points), bool)
    for _ in range(PLANE_RANSAC_TRIES):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-6:
            continue
        inliers = np.abs((points - a) @ normal) / length <= PLANE_INLIER_MM
        if inliers.sum() > best.sum():
            best = inliers
    return best if best.mean() >= min_fraction else np.zeros(len(points), bool)


def _candidate_planes(points: np.ndarray, rng) -> list[np.ndarray]:
    """Index arrays of up to two planes, largest first."""
    remaining = np.arange(len(points))
    found: list[np.ndarray] = []
    for _ in range(MAX_LAYERS):
        if len(remaining) < MIN_LAYER_ZONES:
            break
        mask = dominant_plane(points[remaining], rng, PLANE_MIN_FRACTION if not found else 0.15)
        if mask.sum() < MIN_LAYER_ZONES:
            break
        found.append(remaining[mask])
        remaining = remaining[~mask]
    return found


def _plane_of(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    normal = np.linalg.svd(points - centroid, full_matrices=False)[2][2]
    return (normal if normal @ centroid > 0 else -normal), centroid


def _common_layer(entries: list) -> list[np.ndarray]:
    """Pick, in every view, the plane that is the same physical surface.

    A desk has parallel layers, such as a table and something flat standing on it,
    and the largest one differs between views. Normals agree across layers, so they
    give the mount rotation first. Offsets along that normal then separate the layers.
    """
    rng = np.random.default_rng(0)
    normals = [(T[:3, :3], _plane_of(pts[idx[0]])[0]) for T, pts, idx in entries]

    def spread(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        n0 = x[3:6] / np.linalg.norm(x[3:6])
        return np.concatenate([Ri @ R @ n - n0 for Ri, n in normals])

    best, cost = None, np.inf
    for start in Rotation.random(LAYER_STARTS, rng=rng):
        fit = least_squares(spread, np.r_[start.as_rotvec(), rng.normal(size=3)],
                            loss="soft_l1", f_scale=0.05, max_nfev=60)
        if fit.cost < cost:
            best, cost = fit.x, fit.cost
    R = Rotation.from_rotvec(best[:3]).as_matrix()
    n0 = best[3:6] / np.linalg.norm(best[3:6])

    # Offset of each candidate along n0 is linear in the unknown mount position t:
    #   offset + (n0 . Ri) t = d
    # Fit t and d while reassigning candidates, from every candidate as a starting layer,
    # and keep the fit that puts the most zones on one surface.
    rows = [n0 @ T[:3, :3] for T, _, _ in entries]
    offsets = [[float(n0 @ (T[:3, :3] @ R @ pts[i].mean(axis=0) + T[:3, 3])) for i in idx]
               for T, pts, idx in entries]
    sizes = [[len(i) for i in idx] for _, _, idx in entries]
    best_score, best_pick = -1, None
    for start in [o for row in offsets for o in row]:
        t, d = np.zeros(3), start
        for _ in range(6):
            gaps = [np.abs(np.array(row) + a @ t - d) for row, a in zip(offsets, rows)]
            pick = [int(np.argmin(g)) for g in gaps]
            A = np.array([[*a, -1.0] for a in rows])
            y = -np.array([row[k] for row, k in zip(offsets, pick)])
            keep = np.array([g[k] <= LAYER_GAP_MM for g, k in zip(gaps, pick)])
            if keep.sum() < 5:
                break
            solution = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
            t, d = solution[:3], float(solution[3])
        gaps = [np.abs(np.array(row) + a @ t - d) for row, a in zip(offsets, rows)]
        pick = [int(np.argmin(g)) for g in gaps]
        score = sum(size[k] for g, k, size in zip(gaps, pick, sizes) if g[k] <= LAYER_GAP_MM)
        if score > best_score:
            best_score = score
            best_pick = [k if g[k] <= LAYER_GAP_MM else None for g, k in zip(gaps, pick)]
    return [None if k is None else idx[k] for (_, _, idx), k in zip(entries, best_pick)]


def _observations(views: list[View], cfg: dict, correction: LutCorrection | None, mirror: bool,
                  center_fraction: float = 0.7):
    """Per-view zone distances and slopes, central zones only."""
    proc = cfg["processing"]
    kind = correction.kind if correction else "radial"
    rng = np.random.default_rng(0)
    entries = []
    for view in views:
        rows, cols = view.frames.shape[1:]
        raw, keep = temporal_filter(view, proc)
        distance = correction.apply(raw) if correction else raw
        distance = np.where(keep, distance, np.nan)
        distance[~edge_mask(distance, proc)] = np.nan
        xn, yn = zone_tangents(rows, cols, mirror)
        r, c = np.mgrid[0:rows, 0:cols]
        central = (np.abs(r - (rows - 1) / 2) <= center_fraction * rows / 2) & \
                  (np.abs(c - (cols - 1) / 2) <= center_fraction * cols / 2)
        good = np.isfinite(distance) & central
        good &= (distance >= proc["min_range_mm"]) & (distance <= proc["max_range_mm"])
        if good.sum() < MIN_LAYER_ZONES:
            continue
        points = distance_to_xyz(distance[good], kind, xn[good], yn[good])
        planes = _candidate_planes(points, rng)
        if planes:
            entries.append((pose_to_T(view.tcp_pose), points, planes,
                            distance[good], xn[good], yn[good]))
    if not entries:
        return []
    layered = any(len(e[2]) > 1 for e in entries) and len(entries) >= MIN_VIEWS
    picks = _common_layer([e[:3] for e in entries]) if layered else [e[2][0] for e in entries]
    return [(T, d[i], x[i], y[i]) for (T, _, _, d, x, y), i in zip(entries, picks) if i is not None]


def _plane(a: float, b: float, normal0: np.ndarray) -> np.ndarray:
    """Unit normal near normal0, parameterised without a singularity there."""
    u = np.cross(normal0, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(normal0, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal0, u)
    n = normal0 + a * u + b * v
    return n / np.linalg.norm(n)


def _residuals(x, obs, kind, normal0, fit_bias):
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    t = x[3:6]
    n = _plane(x[6], x[7], normal0)
    bias = x[9] if fit_bias else 0.0
    out = []
    for T, distance, xn, yn in obs:
        s = distance_to_xyz(distance - bias, kind, xn, yn)
        p = (s @ R.T + t) @ T[:3, :3].T + T[:3, 3]
        out.append(p @ n - x[8])
    return np.concatenate(out)


def _solve(obs, kind, mount_T: np.ndarray, fit_bias: bool, mirror: bool,
           loss: str = "soft_l1", max_nfev: int = 200) -> CalibrationResult:
    x0 = np.zeros(10)
    x0[:3] = Rotation.from_matrix(mount_T[:3, :3]).as_rotvec()
    x0[3:6] = mount_T[:3, 3]

    # Starting plane: fit to the points the initial mount predicts.
    pts = []
    for T, distance, xn, yn in obs:
        s = distance_to_xyz(distance, kind, xn, yn)
        pts.append((s @ mount_T[:3, :3].T + mount_T[:3, 3]) @ T[:3, :3].T + T[:3, 3])
    pts = np.vstack(pts)
    centroid = pts.mean(axis=0)
    normal0 = np.linalg.svd(pts - centroid, full_matrices=False)[2][2]
    x0[8] = centroid @ normal0

    free = np.array([True] * 9 + [fit_bias])

    def fun(y):
        x = x0.copy()
        x[free] = y
        return _residuals(x, obs, kind, normal0, fit_bias)

    before = fun(x0[free])
    fit = least_squares(fun, x0[free], loss=loss, f_scale=3.0, x_scale="jac", max_nfev=max_nfev)
    x = x0.copy()
    x[free] = fit.x
    residual = fun(fit.x)

    # Parameter sigma from J^T J: this is what exposes a poorly excited DOF.
    dof = max(len(residual) - int(free.sum()), 1)
    noise = float(np.sum(residual**2) / dof)
    jtj = fit.jac.T @ fit.jac
    covariance = np.linalg.pinv(jtj) * noise
    names = [n for n, f in zip(PARAM_NAMES, free) if f]
    sigma = dict(zip(names, np.sqrt(np.diag(covariance)).tolist()))

    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    pose = T_to_pose(T)
    return CalibrationResult(
        mirror_columns=mirror, xyz_mm=pose[:3], rpy_deg=pose[3:],
        range_bias_mm=float(x[9]) if fit_bias else 0.0,
        rms_mm=float(np.sqrt(np.mean(residual**2))),
        rms_before_mm=float(np.sqrt(np.mean(before**2))),
        sigma=sigma, plane_normal=_plane(x[6], x[7], normal0).tolist(),
        plane_offset_mm=float(x[8]), zones_used=len(residual),
    )


def search_mount(obs, kind, fit_bias: bool, mirror: bool) -> np.ndarray:
    """Global start for a sensor taped on at an unknown orientation.

    The plane residual has local minima far from the truth, so one local solve from
    a wrong guess can settle on a confident wrong answer. Score many rotations by
    how flat they make the data, refine the best few on a subset, keep the winner.
    """
    rng = np.random.default_rng(0)
    subset = []
    for T, distance, xn, yn in obs:
        pick = rng.choice(len(distance), min(SEARCH_ZONES_PER_VIEW, len(distance)), replace=False)
        subset.append((T, distance[pick], xn[pick], yn[pick]))
    sensor = [distance_to_xyz(d, kind, x, y) for _, d, x, y in subset]
    candidates = Rotation.concatenate([Rotation.create_group("O"),
                                       Rotation.random(SEARCH_RANDOM_ROTATIONS, rng=rng)])

    def flatness(R: np.ndarray) -> float:
        points = np.vstack([(s @ R.T) @ T[:3, :3].T + T[:3, 3] for s, (T, *_) in zip(sensor, subset)])
        return float(np.linalg.svd(points - points.mean(axis=0), compute_uv=False)[-1])

    ranked = sorted(candidates.as_matrix(), key=flatness)[:SEARCH_REFINED]
    best, best_rms = None, np.inf
    for R in ranked:
        start = np.eye(4)
        start[:3, :3] = R
        fit = _solve(subset, kind, start, fit_bias, mirror, loss="linear", max_nfev=60)
        if fit.rms_mm < best_rms:
            best, best_rms = pose_to_T([*fit.xyz_mm, *fit.rpy_deg]), fit.rms_mm
    return best


def calibrate_mount(views: list[View], cfg: dict, correction: LutCorrection | None = None
                    ) -> dict[bool, CalibrationResult]:
    """Solve the mount under both mirror hypotheses. Keys are mirror_columns.

    A configured mount is the starting point. With none, it is searched for.
    """
    known = cfg["mount"]["xyz_mm"] is not None and cfg["mount"]["rpy_deg"] is not None
    mount = pose_to_T([*cfg["mount"]["xyz_mm"], *cfg["mount"]["rpy_deg"]]) if known else None
    kind = correction.kind if correction else "radial"
    results = {}
    for mirror in (False, True):
        obs = _observations(views, cfg, correction, mirror)
        if sum(len(o[1]) for o in obs) < MIN_ZONES or len(obs) < MIN_VIEWS:
            raise ValueError(f"only {len(obs)} usable views: the target must fill the FOV "
                             f"in at least {MIN_VIEWS} differently tilted views")
        start = mount if known else search_mount(obs, kind, correction is None, mirror)
        results[mirror] = _solve(obs, kind, start, correction is None, mirror)
    return results


def unobserved(result: CalibrationResult) -> list[str]:
    """Mount parameters the views failed to pin down."""
    return [name for name, limit in MAX_SIGMA.items() if result.sigma[name] > limit]


def recommend(results: dict[bool, CalibrationResult], margin: float = 1.3) -> tuple[bool | None, str]:
    """Pick a mirror only if one hypothesis is clearly better."""
    straight, mirrored = results[False].rms_mm, results[True].rms_mm
    if straight * margin < mirrored:
        return False, f"mirror_columns=false ({straight:.2f} mm vs {mirrored:.2f} mm)"
    if mirrored * margin < straight:
        return True, f"mirror_columns=true ({mirrored:.2f} mm vs {straight:.2f} mm)"
    return None, f"inconclusive ({straight:.2f} vs {mirrored:.2f} mm): add more tilted views"
