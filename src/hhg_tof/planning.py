"""Viewpoints around the object, and every check that can run before moving."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import T_to_pose, inv_T, look_at, pose_to_T, rot_z, segment_point_distance
from .robot.base import IkFailed

HALF_FOV_Y_TAN = 0.5  # narrower axis of the TMF8829 field of view


@dataclass
class Viewpoint:
    index: int
    elevation_deg: float
    azimuth_deg: float
    T_base_sensor: np.ndarray
    roll_deg: float = 0.0
    tcp_pose: list[float] = field(default_factory=list)
    joints: list[float] | None = None


def _view(index: int, center, radius: float, elevation: float, azimuth: float, roll: float) -> Viewpoint:
    e, a = np.radians(elevation), np.radians(azimuth)
    eye = center + radius * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    # d(eye)/d(elevation) is well defined even at the pole, where world-up is not.
    up = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
    return Viewpoint(index, elevation, azimuth, rot_z(look_at(eye, center, up), roll), roll)


def plan_views(scan: dict) -> list[Viewpoint]:
    """Rings of viewpoints on a sphere around the object, top ring first.

    Alternate rings run in opposite directions so each ring starts next to where
    the last one ended. That avoids long moves and the joint winding they cause.
    """
    center = np.asarray(scan["center_mm"], float)
    radius = float(scan["standoff_mm"])
    elevations, counts = scan["elevations_deg"], scan["views_per_ring"]
    if len(elevations) != len(counts):
        raise ValueError("elevations_deg and views_per_ring must have the same length")
    span, start = float(scan["azimuth_span_deg"]), float(scan["azimuth_start_deg"])
    cycle = scan["roll_cycle_deg"]
    views: list[Viewpoint] = []

    def add(elevation: float, azimuth: float) -> None:
        roll = scan["roll_deg"] + cycle[len(views) % len(cycle)]
        views.append(_view(len(views), center, radius, elevation, azimuth, roll))

    if scan["add_top_view"]:
        add(90.0, start)
    arcs = scan["ring_arcs_deg"] or [None] * len(elevations)
    if len(arcs) != len(elevations):
        raise ValueError("ring_arcs_deg must have one entry per ring, null for the default arc")
    for ring, (elevation, count) in enumerate(zip(elevations, counts)):
        ring_start, ring_span = arcs[ring] or (start, span)
        closed = ring_span >= 360.0
        steps = np.arange(count) / (count if closed else max(count - 1, 1))
        azimuths = ring_start + ring_span * steps
        for azimuth in azimuths[::-1] if ring % 2 else azimuths:
            add(elevation, float(azimuth))
    return views


@dataclass
class PlanReport:
    views: list[Viewpoint]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _workspace_errors(name: str, point: np.ndarray, workspace: dict) -> list[str]:
    return [
        f"{name} {axis}={value:.0f} mm is outside {workspace[axis]}"
        for axis, value in zip("xyz", point)
        if not workspace[axis][0] <= value <= workspace[axis][1]
    ]


def preflight(views: list[Viewpoint], cfg: dict, robot, T_tcp_sensor: np.ndarray) -> PlanReport:
    """Solve every view read-only and collect everything wrong with the plan.

    Sensor and TCP are checked, along with the straight lines between them. The
    rest of the arm is NOT modelled: fixtures near the elbow are on the operator.
    """
    scan, rc = cfg["scan"], cfg["robot"]
    center = np.asarray(scan["center_mm"], float)
    keepout = float(scan["object_radius_mm"]) + float(scan["clearance_mm"])
    workspace = scan["workspace_mm"]
    limits = np.asarray(rc["joint_limits_deg"], float)
    margin = float(rc["joint_margin_deg"])
    max_step = float(rc["max_joint_step_deg"])
    T_sensor_tcp = inv_T(T_tcp_sensor)
    errors: list[str] = []
    warnings: list[str] = []

    if scan["object_radius_mm"] > scan["standoff_mm"] * HALF_FOV_Y_TAN:
        warnings.append("object is wider than the short axis of the FOV at this standoff")
    if scan["standoff_mm"] <= keepout:
        errors.append(f"standoff {scan['standoff_mm']} mm is inside the {keepout:.0f} mm keep-out")

    status = robot.status()
    prev_joints = status.joints
    prev_sensor = (pose_to_T(status.tcp_pose) @ T_tcp_sensor)[:3, 3]
    prev_tcp = np.array(status.tcp_pose[:3])

    for view in views:
        tag = f"view {view.index} (el {view.elevation_deg:.0f}, az {view.azimuth_deg:.0f})"
        # The sensor's roll is free, so spend it on keeping the wrist near its last pose.
        best = None
        failures = []
        for roll in scan["roll_candidates_deg"]:
            T_sensor = rot_z(view.T_base_sensor, roll)
            T_tcp = T_sensor @ T_sensor_tcp
            pose = T_to_pose(T_tcp)
            try:
                joints = np.array(robot.solve_ik(pose, prev_joints))
            except IkFailed as exc:
                failures.append(str(exc))
                continue
            in_limits = np.all((joints >= limits[:, 0] + margin) & (joints <= limits[:, 1] - margin))
            step = float(np.abs(joints - np.asarray(prev_joints)).max())
            if not in_limits:
                failures.append("joint limit")
            elif best is None or step < best[0]:
                best = (step, roll, T_sensor, T_tcp, pose, joints)
        if best is None:
            errors.append(f"{tag}: no reachable roll ({'; '.join(sorted(set(failures)))})")
            continue
        step, roll, T_sensor, T_tcp, pose, joints = best
        view.T_base_sensor, view.tcp_pose = T_sensor, pose
        view.joints, view.roll_deg = joints.tolist(), view.roll_deg + roll

        if step > max_step:
            errors.append(f"{tag}: a joint would move {step:.0f} deg, over the {max_step:.0f} deg limit")
        sensor, tcp = T_sensor[:3, 3], T_tcp[:3, 3]
        if workspace:
            errors += _workspace_errors(f"{tag} sensor", sensor, workspace)
            errors += _workspace_errors(f"{tag} TCP", tcp, workspace)
        for name, a, b in (("sensor", prev_sensor, sensor), ("TCP", prev_tcp, tcp)):
            gap = segment_point_distance(a, b, center)
            if gap < keepout:
                errors.append(f"{tag}: {name} path passes {gap:.0f} mm from the object centre "
                              f"(keep-out {keepout:.0f} mm)")
        prev_joints, prev_sensor, prev_tcp = view.joints, sensor, tcp

    return PlanReport(views, errors, warnings)


def level_pose(pose) -> list[float]:
    """`pose` with its tool Z turned straight down by the smallest rotation."""
    T = pose_to_T(pose)
    z, down = T[:3, 2], np.array([0.0, 0.0, -1.0])
    axis = np.cross(z, down)
    angle = float(np.arctan2(np.linalg.norm(axis), z @ down))
    if np.linalg.norm(axis) > 1e-9:
        T[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix() @ T[:3, :3]
    return T_to_pose(T)


def wrist_views(center_pose, tilts_deg=(12.0, 22.0), per_tilt: int = 6, roll_deg: float = 40.0,
                lift_mm: float = 40.0) -> list[Viewpoint]:
    """Plane calibration views made of wrist turns plus a small vertical step.

    Works with an unknown mount, because no sensor position is needed. A sensor near
    the TCP keeps roughly its distance to the table, and only the flange swings, by
    tool length x sin(tilt). Tilt and roll both vary, as `calibration` requires. Tilt
    grows through the list, so the first moves are the smallest.

    The step is what settles the mirror. With rotation alone, "mirrored, looking at
    the plane" and "not mirrored, looking away from it" fit exactly equally well. Only
    a real translation says which way the sensor faces.
    """
    T0 = pose_to_T(center_pose)
    count = len(tilts_deg) * per_tilt
    views = []
    for i in range(count):
        tilt = tilts_deg[i // per_tilt]
        azimuth = 360.0 * (i % per_tilt) / per_tilt + 30.0 * (i // per_tilt)
        roll = roll_deg * float(np.sin(2 * np.pi * i / count))
        rel = np.eye(4)
        rel[:3, :3] = (Rotation.from_euler("z", azimuth, degrees=True)
                       * Rotation.from_euler("x", tilt, degrees=True)
                       * Rotation.from_euler("z", roll - azimuth, degrees=True)).as_matrix()
        T = T0 @ rel
        T[2, 3] += lift_mm * (0.0, 1.0, -1.0)[i % 3]
        views.append(Viewpoint(i, tilt, azimuth, T, roll, T_to_pose(T)))
    return views


def wrist_preflight(views: list[Viewpoint], cfg: dict, robot) -> PlanReport:
    """Real-controller IK for every view, chained from where the arm is now. Read-only."""
    rc = cfg["robot"]
    limits = np.asarray(rc["joint_limits_deg"], float)
    margin, max_step = float(rc["joint_margin_deg"]), float(rc["max_joint_step_deg"])
    errors: list[str] = []
    status = robot.status()
    if status.fault:
        errors.append(f"robot is not clean: {status.fault}")
    if status.tool_id != rc["expected_tool"]:
        errors.append(f"active tool is {status.tool_id}, config expects {rc['expected_tool']}")
    now = np.asarray(status.joints)
    if not np.all((now >= limits[:, 0] + margin) & (now <= limits[:, 1] - margin)):
        errors.append(f"arm starts within {margin:.0f} deg of a joint limit: {np.round(now, 1).tolist()}")
    previous = status.joints
    for view in views:
        tag = f"view {view.index} (tilt {view.elevation_deg:.0f}, az {view.azimuth_deg:.0f})"
        try:
            joints = np.array(robot.solve_ik(view.tcp_pose, previous))
        except IkFailed as exc:
            errors.append(f"{tag}: {exc}")
            continue
        if not np.all((joints >= limits[:, 0] + margin) & (joints <= limits[:, 1] - margin)):
            errors.append(f"{tag}: joint limit {np.round(joints, 1).tolist()}")
        step = float(np.abs(joints - np.asarray(previous)).max())
        if step > max_step:
            errors.append(f"{tag}: a joint would move {step:.0f} deg, over the {max_step:.0f} deg limit")
        view.joints = joints.tolist()
        previous = view.joints
    return PlanReport(views, errors, [])


def approach_path(start_pose, target_pose, start_joints, cfg: dict, robot,
                  max_step_deg: float = 25.0, max_segments: int = 8) -> tuple[list, list]:
    """Straight TCP path split so that no joint moves more than `max_step_deg` per leg.

    Returns (legs, errors); a leg is (pose, joints). Solved with real IK, read-only.
    """
    rc = cfg["robot"]
    limits = np.asarray(rc["joint_limits_deg"], float)
    margin = float(rc["joint_margin_deg"])
    a, b = pose_to_T(start_pose), pose_to_T(target_pose)
    turn = Slerp([0.0, 1.0], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))
    errors: list[str] = []
    for segments in range(1, max_segments + 1):
        legs, errors, previous = [], [], list(start_joints)
        for k in range(1, segments + 1):
            u = k / segments
            T = np.eye(4)
            T[:3, :3] = turn(u).as_matrix()
            T[:3, 3] = (1 - u) * a[:3, 3] + u * b[:3, 3]
            pose = T_to_pose(T)
            try:
                joints = np.array(robot.solve_ik(pose, previous))
            except IkFailed as exc:
                errors.append(f"leg {k}/{segments}: {exc}")
                break
            if not np.all((joints >= limits[:, 0] + margin) & (joints <= limits[:, 1] - margin)):
                errors.append(f"leg {k}/{segments}: joint limit {np.round(joints, 1).tolist()}")
                break
            if float(np.abs(joints - np.asarray(previous)).max()) > max_step_deg:
                errors.append(f"leg {k}/{segments}: joint step over {max_step_deg:.0f} deg")
                break
            legs.append((pose, joints.tolist()))
            previous = joints.tolist()
        if not errors:
            return legs, []
    return [], errors
