"""python -m hist_scan here|scan|range-check|tilt-check|process. Only --execute --confirm SCAN moves the arm."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from hhg_tof import capture, cli, planning, scan_figure, session as session_io
from hhg_tof.config import load_config
from hhg_tof.geometry import pose_to_T
from hhg_tof.pointcloud_io import read_ply

from . import figure, range_check, reconstruct, refine
from .evm_hist import HistogramSensor
from .session import HistogramFrames, write_view


def cmd_here(args) -> int:
    """Save histograms at the current pose. No motion."""
    cfg = load_config(args.config)
    path = session_io.new_session_dir(args.out, args.name)
    robot = cli.build_robot(cfg)
    robot.connect()
    sensor = HistogramSensor()
    try:
        sensor.open()
        pose = robot.status().tcp_pose
        wrapped = HistogramFrames(sensor, path)
        view = capture.measure_view(0, pose, cfg, robot, wrapped)
        session_io.write_view(path, view)
        session_io.write_meta(path, {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": cfg,
                                     "sensor_serial": sensor.serial, "completed_views": [0],
                                     "status": "complete", "source": "hist here"})
    finally:
        sensor.close()
        robot.disconnect()
    print(f"session: {path}")
    return 0


def cmd_scan(args) -> int:
    cfg = load_config(args.config)
    if args.execute and args.confirm != cli.CONFIRM_WORD:
        print(f"Refused: real motion needs --confirm {cli.CONFIRM_WORD}")
        return 2
    robot = cli.build_robot(cfg, allow_motion=args.execute)
    robot.connect()
    sensor = None
    try:
        cli.require_scan_settings(cfg)
        report = planning.preflight(planning.plan_views(cfg["scan"]), cfg, robot, cli.processing.mount_T(cfg))
        cli.print_report(report)
        if not report.ok or not args.execute:
            return 0 if report.ok else 1
        path = session_io.new_session_dir(args.out, args.name)
        sensor = HistogramFrames(HistogramSensor(), path)
        sensor.open()
        capture.run_scan(report.views, cfg, robot, sensor, path)
        print(f"session: {path}")
    finally:
        if sensor:
            sensor.close()
        robot.disconnect()
    return 0


def cmd_process(args) -> int:
    session = Path(args.session)
    cloud = reconstruct.build(session, need_views=args.need_views)
    print(f"bin scale {cloud.scale.mm_per_bin:.2f} mm/bin, zero at bin {cloud.scale.zero_bin:.2f}")
    print(json.dumps(cloud.stats))
    reconstruct.write_ply(session / "cloud_multipeak.ply", cloud)
    (session / "report_multipeak.json").write_text(json.dumps(cloud.stats, indent=2))
    if len(cloud.points):
        figure.draw(cloud, session / "cloud_multipeak_views.png")
        scan_figure.draw(session, session / "scan_positions.png", points=cloud.points)
    indices = [v.index for v in session_io.read_views(session)]
    figure.draw_all_histograms(session, indices, cloud.scale, session / "histograms_all_zones.png")
    figure.draw_histogram_summary(session, indices, cloud.scale, session / "histograms_summary.png")
    print(f"saved {session / 'cloud_multipeak.ply'}")
    refined = refine.build(session)
    reconstruct.write_ply(session / "cloud_refined.ply", refined)
    (session / "report_refined.json").write_text(json.dumps(refined.stats, indent=2))
    figure.draw(refined, session / "cloud_refined_views.png")
    print(json.dumps({k: v for k, v in refined.stats.items() if k != "response"}))
    return 0


def cmd_range_check(args) -> int:
    """Vertical moves over the desk. Moves only with --execute."""
    cfg = load_config(args.config)
    if args.execute and args.confirm != cli.CONFIRM_WORD:
        print(f"Refused: real motion needs --confirm {cli.CONFIRM_WORD}")
        return 2
    robot = cli.build_robot(cfg, allow_motion=args.execute)
    robot.connect()
    sensor = None
    try:
        pose = robot.status().tcp_pose
        zs = range_check.heights(pose[2], args.top, args.bottom, args.step)
        views = range_check.vertical_views(pose, zs)
        report = planning.wrist_preflight(views[1:], cfg, robot)
        report.views = views[:1] + report.views
        views[0].joints = robot.status().joints
        cli.print_report(report)
        print("heights:", [round(z) for z in zs])
        if not report.ok or not args.execute:
            return 0 if report.ok else 1
        sensor = cli.build_sensor(cfg, robot)
        sensor.open()
        path = session_io.new_session_dir(args.out, args.name)
        capture.run_scan(report.views, cfg, robot, sensor, path)
        print(f"session: {path}")
    finally:
        if sensor:
            sensor.close()
        robot.disconnect()
    return 0


def cmd_tilt_check(args) -> int:
    """Turn the sensor in place over the desk. Moves only with --execute."""
    cfg = load_config(args.config)
    if args.execute and args.confirm != cli.CONFIRM_WORD:
        print(f"Refused: real motion needs --confirm {cli.CONFIRM_WORD}")
        return 2
    robot = cli.build_robot(cfg, allow_motion=args.execute)
    robot.connect()
    sensor = None
    try:
        mount = cli.processing.mount_T(cfg)
        level = pose_to_T(robot.status().tcp_pose) @ mount  # the top view already looks straight down
        if abs(level[2, 2] + 1.0) > 1e-3:
            print("the sensor must look straight down first: return to a top view")
            return 1
        level[2, 3] = args.sensor_z
        views = range_check.tilt_views(level, mount, range_check.tilt_plan())
        legs, errors = planning.approach_path(robot.status().tcp_pose, views[0].tcp_pose,
                                              robot.status().joints, cfg, robot)
        for error in errors:
            print(f"ERROR: {error}")
        if errors:
            return 1
        print(f"approach in {len(legs)} legs to {[round(v, 1) for v in views[0].tcp_pose]}")
        start = legs[-1][1]

        class AtStart:
            def status(self):
                state = robot.status()
                state.joints = list(start)
                return state

            def solve_ik(self, pose, ref=None):
                return robot.solve_ik(pose, ref)

        report = planning.wrist_preflight(views, cfg, AtStart())
        cli.print_report(report)
        if not report.ok or not args.execute:
            return 0 if report.ok else 1
        sensor = cli.build_sensor(cfg, robot)
        sensor.open()
        capture.run_approach(legs, cfg, robot)
        path = session_io.new_session_dir(args.out, args.name)
        capture.run_scan(report.views, cfg, robot, sensor, path)
        print(f"session: {path}")
    finally:
        if sensor:
            sensor.close()
        robot.disconnect()
    return 0


def cmd_compare(args) -> int:
    """Each --pair is NAME DEFAULT_SESSION HIST_SESSION."""
    rows = []
    for name, default, hist in args.pair:
        cloud = read_ply(Path(default) / "cloud.ply")
        points = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1)
        rows.append((name, [("default settings", points), ("multi-peak", reconstruct.build(hist)),
                            ("refined multi-peak", refine.build(hist))]))
    figure.draw_comparison(rows, args.out)
    print(f"saved {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hist_scan", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("here", help="save histograms at the current pose, no motion")
    p.add_argument("--config", required=True)
    p.add_argument("--name", default="hist_here")
    p.add_argument("--out", default="data/scans")
    p.set_defaults(fn=cmd_here)
    p = sub.add_parser("scan", help="the normal scan, also saving every histogram")
    p.add_argument("--config", required=True)
    p.add_argument("--name", default="hist")
    p.add_argument("--out", default="data/scans")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm")
    p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("process", help="multi-peak point cloud from a saved session")
    p.add_argument("session")
    p.add_argument("--need-views", type=int, default=2, help="other views that must confirm a point")
    p.set_defaults(fn=cmd_process)
    p = sub.add_parser("range-check", help="vertical moves over a flat desk, to test the range scale")
    p.add_argument("--config", required=True)
    p.add_argument("--name", default="range_check")
    p.add_argument("--out", default="data/scans")
    p.add_argument("--top", type=float, default=476.0)
    p.add_argument("--bottom", type=float, default=76.0)
    p.add_argument("--step", type=float, default=45.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm")
    p.set_defaults(fn=cmd_range_check)
    p = sub.add_parser("tilt-check", help="turn the sensor in place over a flat desk, to test range against angle")
    p.add_argument("--config", required=True)
    p.add_argument("--name", default="tilt_check")
    p.add_argument("--out", default="data/scans")
    p.add_argument("--sensor-z", type=float, default=195.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--confirm")
    p.set_defaults(fn=cmd_tilt_check)
    p = sub.add_parser("compare", help="default cloud against multi-peak cloud, side by side")
    p.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NAME", "DEFAULT", "HIST"))
    p.add_argument("--out", default="data/scans/comparison_default_vs_multipeak.png")
    p.set_defaults(fn=cmd_compare)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
