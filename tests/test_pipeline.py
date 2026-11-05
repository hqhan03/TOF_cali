"""Capture + processing + calibration against simulated ground truth."""

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from hhg_tof import calibration, capture, cli, planning, processing, session as session_io
from hhg_tof.pointcloud_io import read_ply, write_ply
from hhg_tof.robot.base import RobotError
from hhg_tof.robot.simulated import SimulatedRobot
from hhg_tof.session import View


def scan(cfg, path):
    robot = cli.build_robot(cfg)
    sensor = cli.build_sensor(cfg, robot)
    views = planning.plan_views(cfg["scan"])
    report = planning.preflight(views, cfg, robot, processing.mount_T(cfg))
    assert report.ok, report.errors
    capture.run_scan(report.views, cfg, robot, sensor, path)
    return session_io.read_views(path)


@pytest.fixture(scope="module")
def sphere_views(tmp_path_factory):
    from hhg_tof.config import load_config
    cfg = load_config("config/sim_sphere.yaml")
    return scan(cfg, tmp_path_factory.mktemp("sphere"))


@pytest.fixture(scope="module")
def box_views(tmp_path_factory):
    from hhg_tof.config import load_config
    cfg = load_config("config/sim_box.yaml")
    return scan(cfg, tmp_path_factory.mktemp("box"))


def surface_rms(cfg, cloud):
    shape = cli.sim_scene(cfg)[1]
    return float(np.sqrt(np.mean(shape.sdf(cloud.points) ** 2)))


def test_sphere_reconstructs_to_the_true_surface(sphere_views, sim_cfg):
    cfg = sim_cfg()
    cloud, report = processing.process_views(sphere_views, cfg)
    assert report["points_fused"] > 300
    assert surface_rms(cfg, cloud) < 1.0


def test_views_really_cover_different_sides(sphere_views, sim_cfg):
    cfg = sim_cfg()
    cloud, _ = processing.process_views(sphere_views, cfg)
    centered = cloud.points - np.array(cfg["scan"]["center_mm"])
    assert (centered[:, 0] > 30).any() and (centered[:, 0] < -30).any()
    assert (centered[:, 1] > 30).any() and (centered[:, 1] < -30).any()
    assert (centered[:, 2] > 50).any()
    assert cloud.view_counts.max() >= 3


def test_box_sides_and_top_are_all_recovered(box_views, sim_cfg):
    cfg = sim_cfg("box")
    cloud, _ = processing.process_views(box_views, cfg)
    assert surface_rms(cfg, cloud) < 1.5
    shape = cli.sim_scene(cfg)[1]
    samples = shape.surface_samples(3000, np.random.default_rng(1))
    samples = samples[samples[:, 2] > 15]  # the underside and the table join are unseen
    assert (cKDTree(cloud.points).query(samples)[0] < 8).mean() > 0.7


def test_edge_filter_removes_floating_points(box_views, sim_cfg):
    """Without it, zones straddling the box edge float in mid-air."""
    cfg = sim_cfg("box")
    shape = cli.sim_scene(cfg)[1]
    T = processing.mount_T(cfg)

    def worst(edge_abs):
        cfg["processing"]["edge_abs_mm"] = edge_abs
        cfg["processing"]["edge_rel"] = 0.06 if edge_abs < 1e6 else 1e6
        cfg["processing"]["max_incidence_deg"] = 89.9
        points = np.vstack([processing.view_to_cloud(v, cfg, T).points for v in box_views])
        points = points[processing.crop(points, cfg)]
        return np.percentile(shape.sdf(points), 99.5)

    assert worst(20.0) < worst(1e9) * 0.6


def test_wrong_mount_makes_views_disagree(sphere_views, sim_cfg):
    cfg = sim_cfg()
    good = surface_rms(cfg, processing.process_views(sphere_views, cfg)[0])
    cfg["mount"]["xyz_mm"] = [15, 40, 60]
    bad = surface_rms(cfg, processing.process_views(sphere_views, cfg)[0])
    assert bad > 3 * good


def test_processing_uses_measured_pose_not_the_plan(sphere_views, sim_cfg):
    view = sphere_views[0]
    assert not np.array_equal(view.tcp_pose[:3], view.planned_pose[:3])
    assert np.linalg.norm(view.tcp_pose[:3] - view.planned_pose[:3]) < 1.0


def test_filtering_too_hard_is_an_error_not_an_empty_file(sphere_views, sim_cfg):
    cfg = sim_cfg()
    cfg["processing"]["max_range_mm"] = 40
    with pytest.raises(ValueError, match="no points"):
        processing.process_views(sphere_views, cfg)


def test_temporal_filter_ignores_dropouts_and_rejects_flicker():
    proc = dict(min_valid_fraction=0.6, max_mad_mm=8.0, max_mad_rel=0.03, min_snr=0)
    frames = np.full((10, 2, 2), 200.0, np.float32)
    frames[:3, 0, 0] = 0                    # dropouts: median survives
    frames[:8, 0, 1] = 0                    # mostly missing: rejected
    frames[:, 1, 0] = [100, 300] * 5        # flicker between surfaces: rejected
    view = View(0, frames, np.zeros(6), np.zeros(6), np.zeros(6))
    distance, keep = processing.temporal_filter(view, proc)
    assert keep.tolist() == [[True, False], [False, True]]
    assert distance[0, 0] == 200


def test_snr_gate():
    proc = dict(min_valid_fraction=0.6, max_mad_mm=8.0, max_mad_rel=0.03, min_snr=10)
    frames = np.full((5, 1, 2), 200.0, np.float32)
    snr = np.zeros((5, 1, 2), np.uint8)
    snr[:, 0, 1] = 50
    view = View(0, frames, np.zeros(6), np.zeros(6), np.zeros(6), snr)
    assert processing.temporal_filter(view, proc)[1].tolist() == [[False, True]]


def test_voxel_fusion_counts_distinct_views():
    points = np.array([[1.0, 1, 1], [2.0, 1, 1], [1.5, 1, 1], [50.0, 50, 50]])
    centroids, support, _ = processing.fuse_voxels(points, np.array([0, 0, 1, 2]), 4.0)
    order = np.argsort(centroids[:, 0])
    assert np.allclose(centroids[order][0], [1.5, 1, 1])
    assert support[order].tolist() == [2, 1]


def test_normals_face_the_viewer():
    x, y = np.meshgrid(np.arange(10.0), np.arange(10.0))
    points = np.stack([x.ravel(), y.ravel(), np.zeros(100)], axis=1)
    normals = processing.estimate_normals(points, points + [0, 0, 100], 8)
    assert np.allclose(normals, [0, 0, 1], atol=1e-6)


def test_ply_round_trip(tmp_path):
    points = np.array([[1.0, 2, 3], [4, 5, 6]])
    normals = np.array([[0.0, 0, 1], [0, 1, 0]])
    write_ply(tmp_path / "c.ply", points, normals, np.array([3, 300]))
    data = read_ply(tmp_path / "c.ply")
    assert np.allclose(data["x"], [1, 4])
    assert np.allclose(data["ny"], [0, 1])
    assert data["view_count"].tolist() == [3, 255]


class DriftingRobot:
    def __init__(self, robot):
        self._r = robot
        self._n = 0

    def __getattr__(self, name):
        return getattr(self._r, name)

    def status(self):
        s = self._r.status()
        self._n += 1
        s.tcp_pose[0] += self._n * 2.0
        return s


def test_arm_moving_during_capture_aborts_and_keeps_the_session(sim_cfg, tmp_path):
    cfg = sim_cfg()
    base = cli.build_robot(cfg)
    robot = DriftingRobot(base)
    sensor = cli.build_sensor(cfg, base)
    report = planning.preflight(planning.plan_views(cfg["scan"]), cfg, base, processing.mount_T(cfg))
    with pytest.raises(RobotError, match="drifted"):
        capture.run_scan(report.views, cfg, robot, sensor, tmp_path)
    assert session_io.read_meta(tmp_path)["status"].startswith("aborted")


def test_wrong_grid_aborts(sim_cfg, tmp_path):
    cfg = sim_cfg()
    robot = cli.build_robot(cfg)
    sensor = cli.build_sensor(cfg, robot)
    cfg["sensor"]["grid"] = [16, 16]
    report = planning.preflight(planning.plan_views(cfg["scan"]), cfg, robot, processing.mount_T(cfg))
    with pytest.raises(RuntimeError, match="grid"):
        capture.run_scan(report.views, cfg, robot, sensor, tmp_path)


def test_failed_restore_does_not_hide_the_abort_reason(sim_cfg, tmp_path):
    cfg = sim_cfg()
    robot = cli.build_robot(cfg)
    sensor = cli.build_sensor(cfg, robot)
    report = planning.preflight(planning.plan_views(cfg["scan"]), cfg, robot, processing.mount_T(cfg))

    def boom(*_):
        raise TimeoutError("sensor gone")

    def bad_restore():
        raise RobotError("restore failed")

    sensor.read_frames = boom
    robot.restore = bad_restore
    with pytest.raises(TimeoutError, match="sensor gone"):
        capture.run_scan(report.views, cfg, robot, sensor, tmp_path)


@pytest.fixture(scope="module")
def calibration_results(tmp_path_factory):
    from hhg_tof.config import load_config
    cfg = load_config("config/sim_calibration.yaml")
    views = scan(cfg, tmp_path_factory.mktemp("cal"))
    return cfg, calibration.calibrate_mount(views, cfg)


def test_calibration_recovers_the_true_mount(calibration_results):
    cfg, results = calibration_results
    choice, _ = calibration.recommend(results)
    assert choice is True
    best = results[True]
    assert np.allclose(best.xyz_mm, cfg["sim"]["true_mount_xyz_mm"], atol=1.0)
    assert np.allclose(best.rpy_deg, cfg["sim"]["true_mount_rpy_deg"], atol=0.2)
    assert abs(best.range_bias_mm - cfg["sim"]["range_bias_mm"]) < 1.0
    assert not calibration.unobserved(best)


def test_fixed_roll_calibration_is_flagged_as_unobservable(sim_cfg, tmp_path):
    cfg = sim_cfg("calibration")
    cfg["scan"]["roll_cycle_deg"] = [0]
    views = scan(cfg, tmp_path)
    results = calibration.calibrate_mount(views, cfg)
    assert calibration.recommend(results)[0] is None or calibration.unobserved(results[False])


def test_cli_end_to_end(tmp_path, capsys):
    assert cli.main(["scan", "--config", "config/sim_sphere.yaml", "--out", str(tmp_path), "--name", "t"]) == 0
    session = next(tmp_path.glob("*_t"))
    assert cli.main(["process", str(session)]) == 0
    assert (session / "cloud.ply").exists()
    report = json.loads((session / "report.json").read_text())
    assert report["views"] == 21
    assert cli.main(["view", str(session / "cloud.ply"), "--save", str(tmp_path / "c.png")]) == 0
    assert (tmp_path / "c.png").stat().st_size > 1000


def scan_with_recorded_moves(cfg, tmp_path, fail_after=None):
    robot = SimulatedRobot(cfg["sim"]["start_pose"])
    sensor = cli.build_sensor(cfg, robot)
    report = planning.preflight(planning.plan_views(cfg["scan"]), cfg, robot, processing.mount_T(cfg))
    if fail_after is not None:
        moves = robot.move_linear

        def failing(pose, joints, vel, ovl):
            if len(robot.commands) == fail_after:
                raise capture.RobotError("collision")
            moves(pose, joints, vel, ovl)

        robot.move_linear = failing
    return robot, report.views, sensor


def test_finished_scan_retraces_the_orbit_to_the_top_view(sim_cfg, tmp_path):
    cfg = sim_cfg("sphere")
    robot, views, sensor = scan_with_recorded_moves(cfg, tmp_path)
    capture.run_scan(views, cfg, robot, sensor, session_io.new_session_dir(tmp_path, "s"))
    n = len(views)
    assert len(robot.commands) == 2 * n - 1
    back = robot.commands[n:]
    assert back == [v.tcp_pose for v in reversed(views[:-1])]
    assert back[-1] == views[0].tcp_pose and views[0].elevation_deg == 90.0


def test_return_can_be_turned_off(sim_cfg, tmp_path):
    cfg = sim_cfg("sphere")
    cfg["scan"]["return_to_top"] = False
    robot, views, sensor = scan_with_recorded_moves(cfg, tmp_path)
    capture.run_scan(views, cfg, robot, sensor, session_io.new_session_dir(tmp_path, "s"))
    assert len(robot.commands) == len(views)


def test_aborted_scan_does_not_return(sim_cfg, tmp_path):
    cfg = sim_cfg("sphere")
    robot, views, sensor = scan_with_recorded_moves(cfg, tmp_path, fail_after=3)
    with pytest.raises(capture.RobotError):
        capture.run_scan(views, cfg, robot, sensor, session_io.new_session_dir(tmp_path, "s"))
    assert len(robot.commands) == 3


def test_return_refuses_a_leg_that_would_reconfigure_the_arm(sim_cfg):
    cfg = sim_cfg("sphere")
    robot = SimulatedRobot(cfg["sim"]["start_pose"])
    with pytest.raises(capture.RobotError, match="joint step"):
        capture._move_legs([(robot.pose, [90.0] * 6)], cfg, robot)
    assert robot.commands == []


def test_scan_figure_shows_each_view_at_its_measured_position(sim_cfg, tmp_path):
    from hhg_tof import scan_figure

    cfg = sim_cfg("sphere")
    cfg["scan"]["return_to_top"] = False
    robot, views, sensor = scan_with_recorded_moves(cfg, tmp_path)
    path = session_io.new_session_dir(tmp_path, "s")
    capture.run_scan(views, cfg, robot, sensor, path)
    cli.main(["process", str(path)])
    origins, axes, elevation = scan_figure.sensor_stations(path)
    assert len(origins) == len(views)
    assert np.allclose(np.linalg.norm(origins - cfg["scan"]["center_mm"], axis=1), cfg["scan"]["standoff_mm"], atol=2)
    out = tmp_path / "f.png"
    scan_figure.draw(path, out)
    assert out.stat().st_size > 20_000
    assert (path / "cloud_views.png").stat().st_size > 20_000  # process wrote these itself
    assert (path / "scan_positions.png").stat().st_size > 20_000
