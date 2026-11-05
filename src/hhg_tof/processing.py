"""Raw session -> fused point cloud. Reads only saved data, never hardware."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .geometry import pose_to_T, transform_points
from .sensor_model import LutCorrection, distance_to_xyz, zone_tangents
from .session import View


@dataclass
class ViewCloud:
    index: int
    points: np.ndarray    # (N, 3) base frame
    origin: np.ndarray    # (3,) sensor position, base frame
    zones_valid: int
    zones_total: int


@dataclass
class FusedCloud:
    points: np.ndarray
    normals: np.ndarray
    view_counts: np.ndarray


def mount_T(cfg: dict) -> np.ndarray:
    return pose_to_T([*cfg["mount"]["xyz_mm"], *cfg["mount"]["rpy_deg"]])


def temporal_filter(view: View, proc: dict) -> tuple[np.ndarray, np.ndarray]:
    """Collapse frames to one robust range per zone. Returns (range, keep mask)."""
    frames = np.where(view.frames > 0, view.frames.astype(float), np.nan)
    hits = np.isfinite(frames)
    valid_fraction = hits.mean(axis=0)
    usable = hits.any(axis=0)
    median = np.full(frames.shape[1:], np.nan)
    median[usable] = np.nanmedian(frames[:, usable], axis=0)
    mad = np.full_like(median, np.inf)
    mad[usable] = np.nanmedian(np.abs(frames[:, usable] - median[usable]), axis=0)
    keep = (valid_fraction >= proc["min_valid_fraction"]) & usable
    keep &= mad <= np.maximum(proc["max_mad_mm"], proc["max_mad_rel"] * np.nan_to_num(median))
    if view.snr is not None and proc["min_snr"] > 0:
        keep &= np.median(view.snr, axis=0) >= proc["min_snr"]
    return median, keep


def edge_mask(distance: np.ndarray, proc: dict) -> np.ndarray:
    """False for zones that jump from a 4-neighbour.

    A zone straddling a silhouette reports a blend of both surfaces, a floating
    point on neither. The surface it belongs to is seen properly from other views.
    """
    d = np.pad(distance, 1, constant_values=np.nan)
    limit = np.maximum(proc["edge_abs_mm"], proc["edge_rel"] * distance)
    keep = np.ones(distance.shape, bool)
    for neighbour in (d[:-2, 1:-1], d[2:, 1:-1], d[1:-1, :-2], d[1:-1, 2:]):
        with np.errstate(invalid="ignore"):
            keep &= ~(np.abs(distance - neighbour) > limit)
    return keep


def incidence_mask(xyz: np.ndarray, max_deg: float) -> np.ndarray:
    """False where the surface is too oblique to the ray to trust."""
    du = np.gradient(xyz, axis=1)
    dv = np.gradient(xyz, axis=0)
    normal = np.cross(du, dv)
    ray = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    length = np.linalg.norm(normal, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.abs(np.sum(normal * ray, axis=-1)) / length
    return ~(cosine < np.cos(np.radians(max_deg)))


def view_to_cloud(view: View, cfg: dict, T_tcp_sensor: np.ndarray,
                  correction: LutCorrection | None = None) -> ViewCloud:
    proc, sc = cfg["processing"], cfg["sensor"]
    rows, cols = view.frames.shape[1:]
    raw, keep = temporal_filter(view, proc)
    if correction is not None:
        distance, kind = correction.apply(raw), correction.kind
    else:
        distance, kind = raw - sc["range_bias_mm"], "radial"
    distance = np.where(keep, distance, np.nan)
    with np.errstate(invalid="ignore"):
        distance[(distance < proc["min_range_mm"]) | (distance > proc["max_range_mm"])] = np.nan
    distance[~edge_mask(distance, proc)] = np.nan

    xn, yn = zone_tangents(rows, cols, sc["mirror_columns"])
    xyz = distance_to_xyz(distance, kind, xn, yn)
    xyz[~incidence_mask(xyz, proc["max_incidence_deg"])] = np.nan

    good = np.isfinite(xyz).all(axis=-1)
    T = pose_to_T(view.tcp_pose) @ T_tcp_sensor
    return ViewCloud(view.index, transform_points(T, xyz[good]), T[:3, 3].copy(),
                     int(good.sum()), rows * cols)


def crop(points: np.ndarray, cfg: dict) -> np.ndarray:
    """Boolean mask: inside the region of interest and above the floor."""
    proc, scan = cfg["processing"], cfg["scan"]
    keep = np.ones(len(points), bool)
    if scan["center_mm"] is not None:
        radius = proc["roi_radius_mm"] or scan["object_radius_mm"] * 1.3
        keep &= np.linalg.norm(points - np.asarray(scan["center_mm"]), axis=1) <= radius
    if proc["floor_z_mm"] is not None:
        keep &= points[:, 2] > proc["floor_z_mm"] + proc["floor_margin_mm"]
    return keep


def fuse_voxels(points: np.ndarray, view_ids: np.ndarray, voxel_mm: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average per voxel. Returns (centroids, distinct views per voxel, voxel id per point)."""
    keys = np.floor(points / voxel_mm).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    counts = np.bincount(inverse).astype(float)
    centroids = np.stack([np.bincount(inverse, points[:, i]) / counts for i in range(3)], axis=1)
    pairs = np.unique(np.stack([inverse, view_ids], axis=1), axis=0)
    return centroids, np.bincount(pairs[:, 0], minlength=len(counts)), inverse


def remove_outliers(points: np.ndarray, neighbors: int, std_ratio: float) -> np.ndarray:
    """Statistical outlier mask on mean neighbour distance."""
    if len(points) <= neighbors:
        return np.ones(len(points), bool)
    distances, _ = cKDTree(points).query(points, k=neighbors + 1)
    mean = distances[:, 1:].mean(axis=1)
    return mean <= mean.mean() + std_ratio * mean.std()


def estimate_normals(points: np.ndarray, toward: np.ndarray, neighbors: int) -> np.ndarray:
    """PCA normals, flipped to face the direction each point was seen from."""
    k = min(neighbors, len(points))
    _, index = cKDTree(points).query(points, k=k)
    local = points[index] - points[index].mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))
    normals = vectors[:, :, 0]
    flip = np.sum(normals * (toward - points), axis=1) < 0
    normals[flip] *= -1
    return normals


def icp_refine(clouds: list[ViewCloud], cfg: dict) -> list[dict]:
    """Align each view to those before it, point-to-plane, within strict caps.

    Off by default. The arm's poses are better than the sensor's ~7 mm zones on
    smooth or symmetric objects, where ICP slides and quietly makes things worse.
    A correction beyond the caps is rejected and the view stays where the arm put it.
    """
    proc = cfg["processing"]
    log = []
    model = clouds[0].points
    for cloud in clouds[1:]:
        if len(cloud.points) < 30 or len(model) < 30:
            model = np.vstack([model, cloud.points])
            continue
        tree = cKDTree(model)
        _, index = tree.query(model, k=min(12, len(model)))
        local = model[index] - model[index].mean(axis=1, keepdims=True)
        normals = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))[1][:, :, 0]
        T = np.eye(4)
        for _ in range(20):
            moved = transform_points(T, cloud.points)
            distance, nearest = tree.query(moved)
            near = distance < 3 * proc["voxel_mm"]
            if near.sum() < 30:
                break
            p, q, n = moved[near], model[nearest[near]], normals[nearest[near]]
            A = np.hstack([np.cross(p, n), n])
            x, *_ = np.linalg.lstsq(A, np.sum((q - p) * n, axis=1), rcond=None)
            step = np.eye(4)
            step[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
            step[:3, 3] = x[3:]
            T = step @ T
            if np.linalg.norm(x) < 1e-6:
                break
        # Measure the correction about the cloud, not the base origin far away.
        centroid = cloud.points.mean(axis=0)
        shift = float(np.linalg.norm(transform_points(T, centroid[None])[0] - centroid))
        angle = float(np.degrees(Rotation.from_matrix(T[:3, :3]).magnitude()))
        accepted = shift <= proc["icp_max_translation_mm"] and angle <= proc["icp_max_rotation_deg"]
        if accepted:
            cloud.points = transform_points(T, cloud.points)
        log.append({"view": cloud.index, "shift_mm": shift, "rotation_deg": angle, "accepted": accepted})
        model = np.vstack([model, cloud.points])
    return log


def process_views(views: list[View], cfg: dict, correction: LutCorrection | None = None
                  ) -> tuple[FusedCloud, dict]:
    proc = cfg["processing"]
    T_tcp_sensor = mount_T(cfg)
    clouds = [view_to_cloud(v, cfg, T_tcp_sensor, correction) for v in views]
    for cloud in clouds:
        cloud.points = cloud.points[crop(cloud.points, cfg)]
    clouds = [c for c in clouds if len(c.points)]
    if not clouds:
        raise ValueError("no points left after filtering: check ROI, mount and range limits")
    icp_log = icp_refine(clouds, cfg) if proc["icp"] else []

    points = np.vstack([c.points for c in clouds])
    view_ids = np.concatenate([np.full(len(c.points), c.index) for c in clouds])
    origins = np.vstack([np.repeat(c.origin[None], len(c.points), axis=0) for c in clouds])

    fused, support, voxel_of = fuse_voxels(points, view_ids, proc["voxel_mm"])
    counts = np.bincount(voxel_of).astype(float)
    origin = np.stack([np.bincount(voxel_of, origins[:, i]) / counts for i in range(3)], axis=1)

    keep = support >= proc["min_views_per_voxel"]
    fused, support, origin = fused[keep], support[keep], origin[keep]
    if len(fused) == 0:
        raise ValueError("no voxel was seen by enough views: lower min_views_per_voxel or check the mount")
    keep = remove_outliers(fused, proc["outlier_neighbors"], proc["outlier_std"])
    fused, support, origin = fused[keep], support[keep], origin[keep]
    normals = estimate_normals(fused, origin, proc["normal_neighbors"])

    report = {
        "views": len(views),
        "views_with_points": len(clouds),
        "points_before_fusion": int(len(points)),
        "points_fused": int(len(fused)),
        "zones_valid_per_view": [c.zones_valid for c in clouds],
        "distance_kind": correction.kind if correction else "radial",
        "icp": icp_log,
    }
    return FusedCloud(fused, normals, support), report
