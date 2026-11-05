"""hhg-tof command line. Only `scan`, `wrist-scan`, `approach` and `retrace` with --execute --confirm SCAN can move the arm."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import calibration, capture, planning, processing, session as session_io
from .config import load_config, require
from .geometry import inv_T, pose_to_T
from .pointcloud_io import read_ply, write_ply
from .robot.base import IkFailed, RobotError
from .robot.simulated import SimulatedRobot
from .sensor_model import LutCorrection
from .sensors.simulated import Box, Plane, SimNoise, SimulatedSensor, Sphere

CONFIRM_WORD = "SCAN"
MIN_WRIST_CLEARANCE_MM = 120.0
RETRACE_START_TOL_DEG = 1.0


def sim_scene(cfg: dict) -> tuple[list, object]:
    """Floor plus the configured object, sitting on the floor under the scan centre."""
    sim, scan = cfg["sim"], cfg["scan"]
    obj = sim["object"]
    center = np.asarray(scan["center_mm"], float)
    if obj["type"] == "sphere":
        shape = Sphere(center, obj["radius_mm"])
    elif obj["type"] == "box":
        shape = Box(center, obj["size_mm"])
    elif obj["type"] == "none":
        return [Plane(sim["floor_z_mm"])], None
    else:
        raise ValueError(f"unknown sim object {obj['type']!r}")
    return [Plane(sim["floor_z_mm"]), shape], shape


def build_robot(cfg: dict, allow_motion: bool = False):
    rc = cfg["robot"]
    if rc["backend"] == "sim":
        return SimulatedRobot(cfg["sim"]["start_pose"], cfg["sim"]["pose_noise_mm"])
    from .robot.fr5 import FR5Robot

    return FR5Robot(rc["ip"], rc["interface"], rc["expected_tool"], rc["expected_wobj"], allow_motion)


def build_sensor(cfg: dict, robot):
    sc, sim = cfg["sensor"], cfg["sim"]
    if sc["source"] == "sim":
        true_xyz = sim["true_mount_xyz_mm"] or cfg["mount"]["xyz_mm"]
        true_rpy = sim["true_mount_rpy_deg"] or cfg["mount"]["rpy_deg"]
        noise = SimNoise(sim["noise_mm"], sim["noise_rel"], sim["dropout"], sim["range_bias_mm"])
        return SimulatedSensor(robot, sim_scene(cfg)[0], pose_to_T([*true_xyz, *true_rpy]),
                               *sc["grid"], noise, sim["true_mirror_columns"])
    if sc["source"] == "nucleo":
        from .sensors.nucleo import NucleoSensor

        return NucleoSensor(sc["port"], sc["slot"], sc["board_id"], sc["baud"])
    if sc["source"] == "evm":
        from .sensors.evm import EvmSensor

        return EvmSensor(sc["evm_host"], sc["preconfig"])
    raise ValueError(f"unknown sensor source {sc['source']!r}")


def load_correction(cfg: dict, serial: str | None = None) -> LutCorrection | None:
    path = cfg["sensor"]["calibration_json"]
    return LutCorrection(path, *cfg["sensor"]["grid"], expected_serial=serial) if path else None


def require_scan_settings(cfg: dict) -> None:
    require(cfg, "mount.xyz_mm", "mount.rpy_deg", "scan.center_mm", "scan.object_radius_mm")
    if cfg["robot"]["backend"] != "sim":
        require(cfg, "scan.workspace_mm")


def print_report(report: planning.PlanReport) -> None:
    for view in report.views:
        joints = "unsolved" if view.joints is None else " ".join(f"{j:7.1f}" for j in view.joints)
        print(f"  {view.index:3d} el {view.elevation_deg:5.1f} az {view.azimuth_deg:6.1f} "
              f"roll {view.roll_deg:6.1f} | {joints}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(f"preflight: {'OK' if report.ok else 'FAILED'}, {len(report.views)} views")


def cmd_status(args, cfg) -> int:
    robot = build_robot(cfg)
    robot.connect()
    try:
        s = robot.status()
        print(f"TCP pose   {[round(v, 2) for v in s.tcp_pose]}")
        print(f"joints     {[round(v, 2) for v in s.joints]}")
        print(f"mode       {'manual' if s.robot_mode else 'auto'}   tool {s.tool_id}   wobj {s.wobj_id}")
        print(f"fault      {s.fault or 'none'}")
    finally:
        robot.disconnect()
    return 0


def cmd_sensor_test(args, cfg) -> int:
    sensor = build_sensor(cfg, build_robot(cfg))
    try:
        sensor.open()
        frames = sensor.read_frames(args.frames, 1)
        d = np.stack([f.distance_mm for f in frames])
        hits = d[d > 0]
        print(f"serial {sensor.serial or '(unknown)'}  grid {d.shape[1:]}  frames {len(frames)}")
        print(f"valid zones {100 * (d > 0).mean():.1f} %   median range {np.median(hits):.1f} mm")
        print(f"centre zone {np.median(d[:, d.shape[1] // 2, d.shape[2] // 2]):.1f} mm")
    finally:
        sensor.close()
    return 0


def _plan(cfg: dict, robot) -> planning.PlanReport:
    require_scan_settings(cfg)
    return planning.preflight(planning.plan_views(cfg["scan"]), cfg, robot, processing.mount_T(cfg))


def cmd_plan(args, cfg) -> int:
    robot = build_robot(cfg)
    robot.connect()
    try:
        report = _plan(cfg, robot)
    finally:
        robot.disconnect()
    print_report(report)
    return 0 if report.ok else 1


def cmd_approach(args, cfg) -> int:
    """Move to the first view of the scan in legs small enough for the joint-step limit."""
    real = cfg["robot"]["backend"] != "sim"
    if real and args.execute and args.confirm != CONFIRM_WORD:
        print(f"Refused: real motion needs --confirm {CONFIRM_WORD}")
        return 2
    robot = build_robot(cfg, allow_motion=real and args.execute)
    robot.connect()
    try:
        report = _plan(cfg, robot)
        other = [e for e in report.errors if not e.startswith("view 0 ") or "joint would move" not in e]
        if other:
            print_report(report)
            return 1
        status = robot.status()
        if status.fault:
            print(f"robot is not clean: {status.fault}")
            return 1
        legs, errors = planning.approach_path(status.tcp_pose, report.views[0].tcp_pose,
                                              status.joints, cfg, robot)
        for error in errors:
            print(f"ERROR: {error}")
        if errors:
            return 1
        for index, (pose, joints) in enumerate(legs, 1):
            print(f"  leg {index}/{len(legs)} tcp {np.round(pose[:3], 1).tolist()} joints {np.round(joints, 1).tolist()}")
        if not args.execute:
            print("Nothing moved. Add --execute --confirm SCAN to run.")
            return 0
        capture.run_approach(legs, cfg, robot)
        print("at the first view:", [round(v, 2) for v in robot.status().tcp_pose])
    finally:
        robot.disconnect()
    return 0


def cmd_retrace(args, cfg) -> int:
    """Fly a finished session's orbit backwards, to its first view. Moves only with --execute."""
    real = cfg["robot"]["backend"] != "sim"
    if real and args.execute and args.confirm != CONFIRM_WORD:
        print(f"Refused: real motion needs --confirm {CONFIRM_WORD}")
        return 2
    views = session_io.read_views(args.session)
    robot = build_robot(cfg, allow_motion=real and args.execute)
    robot.connect()
    try:
        status = robot.status()
        if status.fault:
            print(f"robot is not clean: {status.fault}")
            return 1
        gap = float(np.abs(np.array(status.joints) - views[-1].joints).max())
        if gap > RETRACE_START_TOL_DEG:
            print(f"arm is {gap:.1f} deg from the session's last view, so this path does not start here")
            return 1
        legs = capture.retrace_legs(views)
        steps = [float(np.abs(np.array(a[1]) - b[1]).max()) for a, b in zip(legs, [(0, views[-1].joints), *legs])]
        print(f"{len(legs)} legs back to {[round(float(v), 1) for v in views[0].tcp_pose]}, "
              f"largest joint step {max(steps):.1f} deg")
        if not args.execute:
            print("Nothing moved. Add --execute --confirm SCAN to run.")
            return 0
        capture.run_approach(legs, cfg, robot)
        print("back at:", [round(v, 2) for v in robot.status().tcp_pose])
    finally:
        robot.disconnect()
    return 0


def cmd_scan(args, cfg) -> int:
    real = cfg["robot"]["backend"] != "sim"
    if real and not args.execute:
        print("Nothing will move. This is a preflight; add --execute --confirm SCAN to run.")
    if real and args.execute and args.confirm != CONFIRM_WORD:
        print(f"Refused: real motion needs --confirm {CONFIRM_WORD}")
        return 2
    run = args.execute or not real
    robot = build_robot(cfg, allow_motion=real and args.execute)
    robot.connect()
    sensor = None
    try:
        report = _plan(cfg, robot)
        print_report(report)
        if not report.ok or not run:
            return 0 if report.ok else 1
        sensor = build_sensor(cfg, robot)
        sensor.open()
        path = session_io.new_session_dir(args.out, args.name)
        capture.run_scan(report.views, cfg, robot, sensor, path)
        print(f"session: {path}")
    finally:
        if sensor:
            sensor.close()
        robot.disconnect()
    return 0


def _session_config(args) -> dict:
    """The config the session was taken with, unless --config overrides it."""
    cfg = session_io.read_meta(args.session)["config"]
    return load_config(args.config) if args.config else cfg


def cmd_process(args, cfg) -> int:
    cfg = _session_config(args)
    if not cfg["sensor"]["mirror_verified"]:
        print("WARNING: column mirror has not been verified on this sensor (see README, step 3)")
    views = session_io.read_views(args.session)
    meta = session_io.read_meta(args.session)
    cloud, report = processing.process_views(views, cfg, load_correction(cfg, meta["sensor_serial"]))
    out = Path(args.session) / "cloud.ply"
    write_ply(out, cloud.points, cloud.normals, cloud.view_counts)
    (Path(args.session) / "report.json").write_text(json.dumps(report, indent=2))
    print(f"{report['points_fused']} points from {report['views_with_points']} views -> {out}")
    try:
        from . import scan_figure
    except ImportError:
        return 0
    try:
        scan_figure.draw_views(args.session, Path(args.session) / "cloud_views.png")
        scan_figure.draw(args.session, Path(args.session) / "scan_positions.png")
    except ImportError:
        print("matplotlib is not installed, skipping figures")
    return 0


def cmd_calibrate(args, cfg) -> int:
    cfg = _session_config(args)
    views = session_io.read_views(args.session)
    results = calibration.calibrate_mount(views, cfg, load_correction(cfg))
    choice, reason = calibration.recommend(results)
    print(f"mirror: {reason}")
    for mirror, r in results.items():
        print(f"  mirror={mirror!s:5}  rms {r.rms_before_mm:.2f} -> {r.rms_mm:.2f} mm, {r.zones_used} zones")
    best = results[choice if choice is not None else False]
    weak = calibration.unobserved(best)
    out = Path(args.session) / "mount_calibration.json"
    out.write_text(json.dumps({str(k): asdict(v) for k, v in results.items()}, indent=2))
    if choice is None or weak:
        print(f"REJECTED: do not use this mount. Poorly observed: {weak or 'mirror'}. "
              "Rescan with varied tilt and roll_cycle_deg.")
        return 1
    print(f"mount.xyz_mm:  {[round(v, 2) for v in best.xyz_mm]}")
    print(f"mount.rpy_deg: {[round(v, 3) for v in best.rpy_deg]}")
    print(f"range bias:    {best.range_bias_mm:.2f} mm")
    print("1-sigma:       " + ", ".join(f"{k} {v:.3f}" for k, v in best.sigma.items()))
    print(f"saved {out}")
    return 0


def _clear_ahead(sensor, cfg: dict) -> None:
    """Refuse to swing the wrist with something close in front of the sensor."""
    frames = np.stack([f.distance_mm for f in sensor.read_frames(5, 2)])
    hits = frames[frames > 0]
    if hits.size == 0:
        raise ValueError("sensor sees nothing: the plane must be in view before moving")
    near = float(np.percentile(hits, 5))
    print(f"sensor: {100 * (frames > 0).mean():.0f} % valid, nearest 5 % at {near:.0f} mm")
    if near < MIN_WRIST_CLEARANCE_MM:
        raise ValueError(f"something is {near:.0f} mm from the sensor, under the "
                         f"{MIN_WRIST_CLEARANCE_MM:.0f} mm needed to turn the wrist")


def cmd_wrist_scan(args, cfg) -> int:
    """Plane calibration by turning the wrist in place. Moves only with --execute."""
    real = cfg["robot"]["backend"] != "sim"
    if real and args.execute and args.confirm != CONFIRM_WORD:
        print(f"Refused: real motion needs --confirm {CONFIRM_WORD}")
        return 2
    robot = build_robot(cfg, allow_motion=real and args.execute)
    robot.connect()
    sensor = None
    try:
        start = robot.status().tcp_pose
        center = planning.level_pose(start) if args.level else start
        views = planning.wrist_views(center)[:args.views]
        report = planning.wrist_preflight(views, cfg, robot)
        print_report(report)
        if not report.ok:
            return 1
        sensor = build_sensor(cfg, robot)
        sensor.open()
        _clear_ahead(sensor, cfg)
        if not args.execute:
            print("Nothing moved. Add --execute --confirm SCAN to run.")
            return 0
        path = session_io.new_session_dir(args.out, args.name)
        capture.run_scan(report.views, cfg, robot, sensor, path)
        print(f"session: {path}")
    finally:
        if sensor:
            sensor.close()
        robot.disconnect()
    return 0


def cmd_jog_capture(args, cfg) -> int:
    """Collect a plane calibration at poses a person jogged to. Nothing here can move the arm."""
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    robot = build_robot(cfg)  # allow_motion stays False
    robot.connect()
    sensor = None
    try:
        sensor = build_sensor(cfg, robot)
        sensor.open()
        path = session_io.new_session_dir(args.out, args.name)
        capture.run_jog_capture(cfg, robot, sensor, path, args.auto, timeout_s=args.minutes * 60)
        print(f"session: {path}\nnext: hhg-tof calibrate {path}")
    except (KeyboardInterrupt, EOFError):
        return 0
    finally:
        if sensor:
            sensor.close()
        robot.disconnect()
    return 0


def cmd_monitor(args, cfg) -> int:
    """Own the robot stream and keep the sensor measuring, for `live` when no scan runs."""
    # SIGTERM too: a client that dies without Leave locks the EVM until it is replugged.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    robot = build_robot(cfg)  # allow_motion stays False
    robot.connect()
    sensor = None
    try:
        if not args.no_sensor:
            sensor = build_sensor(cfg, robot)
            sensor.open()
        print("monitoring, nothing will move. Stop this before `scan`. Ctrl-C to quit.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 0
    finally:
        if sensor:
            sensor.close()
        robot.disconnect()


def cmd_live(args, cfg) -> int:
    from . import live_view

    live_view.run(cfg, load_correction(cfg), args.demo, args.save, not args.no_window, args.duration)
    return 0


def cmd_scan_figure(args, cfg) -> int:
    from . import scan_figure

    out = args.save or str(Path(args.session) / "scan_positions.png")
    scan_figure.draw(args.session, out)
    print(f"saved {out}")
    return 0


def cmd_view(args, cfg) -> int:
    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cloud = read_ply(args.ply)
    p = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    art = ax.scatter(*p.T, c=p[:, 2], s=2, cmap="viridis")
    ax.set_box_aspect(np.ptp(p, axis=0))
    ax.set(xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
    fig.colorbar(art, label="z [mm]", shrink=0.7)
    if args.save:
        fig.savefig(args.save, dpi=150)
    else:
        plt.show()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hhg-tof", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, fn, help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", help="YAML config file")
        p.set_defaults(fn=fn)
        return p

    add("status", cmd_status, "read robot state (read-only)")
    add("sensor-test", cmd_sensor_test, "read a few frames, no robot").add_argument(
        "--frames", type=int, default=10)
    add("plan", cmd_plan, "solve and check every view (read-only)")
    p = add("scan", cmd_scan, "preflight, then capture with --execute")
    p.add_argument("--name", default="scan")
    p.add_argument("--out", default="data/scans")
    p.add_argument("--execute", action="store_true", help="really move the arm")
    p.add_argument("--confirm", help=f"must be {CONFIRM_WORD} for real motion")
    add("process", cmd_process, "session -> cloud.ply").add_argument("session")
    add("calibrate", cmd_calibrate, "solve sensor mount and mirror from a plane scan").add_argument("session")
    p = add("retrace", cmd_retrace, "fly a session's orbit backwards to its first view")
    p.add_argument("session")
    p.add_argument("--execute", action="store_true", help="really move the arm")
    p.add_argument("--confirm", help=f"must be {CONFIRM_WORD} for real motion")
    p = add("approach", cmd_approach, "move to the first view of the scan, in small legs")
    p.add_argument("--execute", action="store_true", help="really move the arm")
    p.add_argument("--confirm", help=f"must be {CONFIRM_WORD} for real motion")
    p = add("wrist-scan", cmd_wrist_scan, "plane calibration by turning the wrist in place")
    p.add_argument("--name", default="wrist")
    p.add_argument("--out", default="data/scans")
    p.add_argument("--views", type=int, default=12, help="run only the first N views")
    p.add_argument("--level", action="store_true", help="centre the views on tool Z pointing straight down")
    p.add_argument("--execute", action="store_true", help="really move the arm")
    p.add_argument("--confirm", help=f"must be {CONFIRM_WORD} for real motion")
    p = add("jog-capture", cmd_jog_capture, "save views at hand-jogged poses, no motion")
    p.add_argument("--name", default="plane")
    p.add_argument("--out", default="data/scans")
    p.add_argument("--auto", action="store_true", help="capture whenever the arm rests at a new angle")
    p.add_argument("--minutes", type=float, default=20.0, help="time limit with --auto")
    add("monitor", cmd_monitor, "publish robot state and run the sensor, no motion").add_argument(
        "--no-sensor", action="store_true", help="publish robot state only")
    p = add("live", cmd_live, "viewer: range image, robot pose, 3D points (subscribes only)")
    p.add_argument("--demo", action="store_true", help="simulated data, needs a sim config")
    p.add_argument("--save", help="also record to this .rrd file (replay: rerun FILE)")
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--duration", type=float, help="stop after this many seconds")
    p = add("scan-figure", cmd_scan_figure, "picture of a session: cloud plus where each view was taken")
    p.add_argument("session")
    p.add_argument("--save", help="output PNG, default <session>/scan_positions.png")
    p = add("view", cmd_view, "show a PLY")
    p.add_argument("ply")
    p.add_argument("--save", help="write a PNG instead of opening a window")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    try:
        return args.fn(args, cfg)
    except (RobotError, IkFailed, ValueError, TimeoutError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
