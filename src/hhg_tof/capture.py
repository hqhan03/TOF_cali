"""Stop-and-scan capture loop.

The arm stops for every view rather than scanning on the move. A 48x32 frame
takes about 190 ms, so at even 50 mm/s a moving frame smears across 10 mm, more
than a whole zone at scan distance, and no timestamp alignment recovers that.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import calibration
from .planning import Viewpoint
from .robot.base import RobotError
from .sensors.base import expected_grid_or_raise
from .session import View, read_views, write_meta, write_view

log = logging.getLogger(__name__)

MIN_JOG_VALID_PERCENT = 40.0
AUTO_HOLD_S = 2.0
AUTO_STILL_MM = 0.3
AUTO_STILL_DEG = 0.1
AUTO_MIN_ROTATION_DEG = 8.0
AUTO_ENOUGH_VIEWS = 10


def measure_view(index: int, planned_pose, cfg: dict, robot, sensor) -> View:
    """Read frames at the current pose, with the arm already at rest. Never moves it."""
    rc, sc = cfg["robot"], cfg["sensor"]
    before = robot.status()
    frames = sensor.read_frames(sc["frames_per_view"], sc["discard_frames"])
    after = robot.status()
    drift = float(np.linalg.norm(np.array(after.tcp_pose[:3]) - np.array(before.tcp_pose[:3])))
    if drift > rc["max_view_drift_mm"]:
        raise RobotError(f"view {index}: arm drifted {drift:.2f} mm during capture")
    expected_grid_or_raise(frames[0], sc["grid"])

    snr = np.stack([f.snr for f in frames]) if frames[0].snr is not None else None
    return View(
        index=index,
        frames=np.stack([f.distance_mm for f in frames]),
        # Not the planned pose: this is where the arm actually was.
        tcp_pose=np.array(after.tcp_pose),
        joints=np.array(after.joints),
        planned_pose=np.array(planned_pose),
        snr=snr,
    )


def _capture_view(view: Viewpoint, cfg: dict, robot, sensor) -> View:
    rc = cfg["robot"]
    # Re-solve from where the arm really is. The preflight chain was a prediction.
    joints = robot.solve_ik(view.tcp_pose)
    current = np.array(robot.status().joints)
    step = float(np.abs(np.array(joints) - current).max())
    if step > rc["max_joint_step_deg"]:
        raise RobotError(f"view {view.index}: joint step {step:.1f} deg is over the limit, "
                         "the arm would reconfigure")
    robot.move_linear(view.tcp_pose, joints, rc["vel_percent"], rc["ovl_percent"])
    robot.wait_settled(rc["settle_timeout_s"], rc["settle_tol_mm"], rc["settle_hold_s"])
    return measure_view(view.index, view.tcp_pose, cfg, robot, sensor)


def _check_tool(cfg: dict, robot) -> None:
    """The mount is solved against whatever frame the pose is reported in."""
    expected, tool = cfg["robot"]["expected_tool"], robot.status().tool_id
    if tool != expected:
        raise RobotError(f"active tool is {tool}, config expects {expected}. Set "
                         "robot.expected_tool to the tool the mount should be relative to.")


def _jog_progress(session: Path, cfg: dict) -> str:
    """Tell the operator whether the views so far are enough, and what is still missing."""
    views = read_views(session)
    if len(views) < calibration.MIN_VIEWS:
        return f"  need {calibration.MIN_VIEWS - len(views)} more views"
    try:
        results = calibration.calibrate_mount(views, cfg)
    except ValueError as exc:
        return f"  not solvable yet: {exc}"
    choice, reason = calibration.recommend(results)
    best = results[False if choice is None else choice]
    weak = calibration.unobserved(best)
    if choice is None or weak:
        return f"  not enough yet (mirror: {reason}; weak: {weak or 'none'}). Change tilt and roll."
    return f"  ENOUGH: rms {best.rms_mm:.2f} mm, {reason}. q to finish, or add more."


def _rotation_gap_deg(a, b) -> float:
    return float(np.degrees((Rotation.from_euler("xyz", a[3:6], degrees=True).inv()
                             * Rotation.from_euler("xyz", b[3:6], degrees=True)).magnitude()))


def _wait_for_new_pose(robot, tried: list, deadline: float, hold_s: float) -> bool:
    """Block until the arm rests at an orientation not tried yet. False on timeout."""
    anchor, since, reported = None, time.monotonic(), None
    while time.monotonic() < deadline:
        status = robot.status()
        pose = np.array(status.tcp_pose)
        if status.fault:
            # Nothing here commands motion, so a fault raised while a person jogs is
            # theirs to clear. Keep waiting, but never measure through it.
            if status.fault != reported:
                print(f"paused: {status.fault}. Check the pendant; capture resumes once it clears.",
                      flush=True)
            reported, anchor = status.fault, None
            time.sleep(0.05)
            continue
        if reported:
            print("fault cleared, resuming", flush=True)
            reported = None
        if (anchor is None or np.linalg.norm(pose[:3] - anchor[:3]) > AUTO_STILL_MM
                or _rotation_gap_deg(pose, anchor) > AUTO_STILL_DEG):
            anchor, since = pose, time.monotonic()
        elif (time.monotonic() - since >= hold_s
              and all(_rotation_gap_deg(pose, t) >= AUTO_MIN_ROTATION_DEG for t in tried)):
            return True
        time.sleep(0.05)
    return False


def run_jog_capture(cfg: dict, robot, sensor, session: Path, auto: bool = False,
                    max_views: int = 14, timeout_s: float = 1200.0,
                    hold_s: float = AUTO_HOLD_S) -> Path:
    """Save views at poses the operator jogged to: on Enter, or with `auto` by itself
    whenever the arm comes to rest at a new orientation.

    Makes no motion call, so it works with a read-only robot. That is the point: with
    the mount unknown, a planned orbit cannot be computed or trusted.
    """
    _check_tool(cfg, robot)
    rc = cfg["robot"]
    Path(session).mkdir(parents=True, exist_ok=True)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg,
        "sensor_serial": sensor.serial,
        "completed_views": [],
        "status": "running",
        "source": "jog",
    }
    write_meta(session, meta)
    deadline = time.monotonic() + timeout_s
    tried: list = []
    try:
        while len(meta["completed_views"]) < max_views:
            saved = len(meta["completed_views"])
            if auto:
                print(f"[{saved} saved] waiting for the arm to rest at a new orientation", flush=True)
                if not _wait_for_new_pose(robot, tried, deadline, hold_s):
                    print("time limit reached", flush=True)
                    break
            elif input(f"[{saved} saved] Enter = capture, q = finish: ").strip().lower() == "q":
                break
            _check_tool(cfg, robot)
            try:
                robot.wait_settled(rc["settle_timeout_s"], rc["settle_tol_mm"], rc["settle_hold_s"])
                pose = robot.status().tcp_pose
                view = measure_view(saved, pose, cfg, robot, sensor)
            except RobotError as exc:
                print(f"not saved: {exc}", flush=True)
                continue
            tried.append(np.array(pose))
            valid = 100.0 * float((view.frames > 0).mean())
            if valid < MIN_JOG_VALID_PERCENT:
                print(f"not saved: only {valid:.0f} % of zones see anything. Move closer, "
                      "the plane must fill the view.", flush=True)
                continue
            write_view(session, view)
            meta["completed_views"].append(saved)
            write_meta(session, meta)
            print(f"view {saved}: {valid:.0f} % valid, median range "
                  f"{float(np.median(view.frames[view.frames > 0])):.0f} mm", flush=True)
            progress = _jog_progress(session, cfg)
            print(progress, flush=True)
            if auto and "ENOUGH" in progress and saved + 1 >= AUTO_ENOUGH_VIEWS:
                break
        meta["status"] = "complete"
    except KeyboardInterrupt:
        meta["status"] = "complete"  # stopping by hand is normal, the saved views are good
    finally:
        if meta["status"] == "running":
            meta["status"] = "aborted"
        write_meta(session, meta)
    return Path(session)


def _move_legs(legs: list, cfg: dict, robot) -> None:
    """Move through (pose, joints) legs, refusing any leg that would reconfigure the arm."""
    rc = cfg["robot"]
    for index, (pose, joints) in enumerate(legs, 1):
        step = float(np.abs(np.array(joints) - np.array(robot.status().joints)).max())
        if step > rc["max_joint_step_deg"]:
            raise RobotError(f"leg {index}: joint step {step:.1f} deg is over the limit")
        log.info("move %d/%d", index, len(legs))
        robot.move_linear(pose, joints, rc["vel_percent"], rc["ovl_percent"])
        robot.wait_settled(rc["settle_timeout_s"], rc["settle_tol_mm"], rc["settle_hold_s"])


def retrace_legs(views) -> list:
    """The path back to the first view: the orbit in reverse, whatever `views` hold.

    A direct line cannot do this. The orbit winds J6 by about 170 deg, and a straight
    TCP path to the same orientation would jump joints. Every leg of the reverse path
    was already checked, and flown, on the way out.
    """
    return [(list(v.tcp_pose), list(v.joints)) for v in reversed(views[:-1])]


def run_approach(legs: list, cfg: dict, robot) -> None:
    """Move through preflighted (pose, joints) legs without measuring."""
    rc = cfg["robot"]
    moving = False
    try:
        robot.prepare_motion(rc["speed_percent"])
        moving = True
        _move_legs(legs, cfg, robot)
    except BaseException:
        if moving:
            robot.stop()
        raise
    finally:
        try:
            robot.restore()
        except Exception as exc:  # noqa: BLE001 - must not mask the abort reason
            log.error("could not restore the controller mode: %s", exc)


def run_scan(views: list[Viewpoint], cfg: dict, robot, sensor, session: Path) -> dict:
    """Run a preflighted plan. Views already saved survive an abort."""
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": cfg,
        "sensor_serial": sensor.serial,
        "completed_views": [],
        "status": "running",
    }
    write_meta(session, meta)
    moving = False
    try:
        robot.prepare_motion(cfg["robot"]["speed_percent"])
        moving = True
        for view in views:
            log.info("view %d/%d  el %.0f az %.0f", view.index + 1, len(views),
                     view.elevation_deg, view.azimuth_deg)
            write_view(session, _capture_view(view, cfg, robot, sensor))
            meta["completed_views"].append(view.index)
            write_meta(session, meta)
        meta["status"] = "complete"
        write_meta(session, meta)
        if cfg["scan"]["return_to_top"] and len(views) > 1 and views[0].elevation_deg == 90.0:
            log.info("returning to the top view")
            _move_legs(retrace_legs(views), cfg, robot)
    except BaseException as exc:
        # An abort must not trigger the return: the arm stops where it is.
        meta["status"] = f"{meta['status']}, then aborted: {exc!r}" if meta["status"] == "complete" \
            else f"aborted: {exc!r}"
        if moving:
            robot.stop()
        raise
    finally:
        write_meta(session, meta)
        try:
            robot.restore()
        except Exception as exc:  # noqa: BLE001 - must not mask the abort reason
            log.error("could not restore the controller mode: %s", exc)
    return meta
