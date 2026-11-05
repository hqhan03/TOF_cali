import numpy as np
import pytest

from hhg_tof import planning
from hhg_tof.geometry import T_to_pose, inv_T, look_at, pose_to_T, rot_z, segment_point_distance
from hhg_tof.processing import mount_T
from hhg_tof.robot.base import IkFailed, RobotStatus
from hhg_tof.robot.simulated import SimulatedRobot


def test_pose_matches_controller_convention():
    # R = Rz @ Ry @ Rx, written out by hand.
    rx, ry, rz = np.radians([10, 20, 30])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    assert np.allclose(pose_to_T([0, 0, 0, 10, 20, 30])[:3, :3], Rz @ Ry @ Rx)


def test_pose_round_trip_and_inverse():
    pose = [100, -50, 300, 170, -20, 45]
    T = pose_to_T(pose)
    assert np.allclose(T_to_pose(T), pose)
    assert np.allclose(T @ inv_T(T), np.eye(4))


def test_look_at_is_right_handed_and_aims_at_target():
    T = look_at([100, 0, 200], [0, 0, 0], [0, 0, 1])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)
    direction = np.array([-100, 0, -200]) / np.hypot(100, 200)
    assert np.allclose(T[:3, 2], direction)


def test_every_view_aims_at_the_centre_from_the_standoff(sim_cfg):
    scan = sim_cfg()["scan"]
    center = np.array(scan["center_mm"], float)
    for view in planning.plan_views(scan):
        eye = view.T_base_sensor[:3, 3]
        assert np.isclose(np.linalg.norm(eye - center), scan["standoff_mm"])
        assert np.allclose(view.T_base_sensor[:3, 2], (center - eye) / scan["standoff_mm"])
        assert eye[2] > center[2]


def test_rings_alternate_direction_to_keep_moves_short(sim_cfg):
    cfg = sim_cfg()
    eyes = np.array([v.T_base_sensor[:3, 3] for v in planning.plan_views(cfg["scan"])])
    assert np.linalg.norm(np.diff(eyes, axis=0), axis=1).max() < cfg["scan"]["standoff_mm"] * 1.2


def test_segment_distance():
    a, b = np.zeros(3), np.array([10.0, 0, 0])
    assert segment_point_distance(a, b, np.array([5.0, 3, 0])) == 3
    assert segment_point_distance(a, b, np.array([-4.0, 3, 0])) == 5


class ScriptedRobot(SimulatedRobot):
    """IK driven by a function of the target pose."""

    def __init__(self, ik):
        super().__init__([400, 0, 400, 180, 0, 0])
        self.ik = ik

    def solve_ik(self, pose, ref_joints=None):
        return self.ik(pose, ref_joints)


def run(cfg, robot):
    views = planning.plan_views(cfg["scan"])
    return planning.preflight(views, cfg, robot, mount_T(cfg))


def test_clean_plan_passes(sim_cfg):
    cfg = sim_cfg()
    report = run(cfg, SimulatedRobot(cfg["sim"]["start_pose"]))
    assert report.ok
    assert all(len(v.tcp_pose) == 6 for v in report.views)


def test_tcp_targets_put_the_sensor_on_the_view(sim_cfg):
    cfg = sim_cfg()
    report = run(cfg, SimulatedRobot(cfg["sim"]["start_pose"]))
    for view in report.views:
        assert np.allclose(pose_to_T(view.tcp_pose) @ mount_T(cfg), view.T_base_sensor, atol=1e-6)


def test_unreachable_view_is_an_error(sim_cfg):
    def ik(pose, ref):
        raise IkFailed("target pose unreachable")
    report = run(sim_cfg(), ScriptedRobot(ik))
    assert not report.ok
    assert "no reachable roll" in report.errors[0]


def test_joint_limit_is_an_error(sim_cfg):
    report = run(sim_cfg(), ScriptedRobot(lambda pose, ref: [172.0, 0, 0, 0, 0, 0]))
    assert not report.ok


def test_reconfiguration_is_an_error(sim_cfg):
    cfg = sim_cfg()
    cfg["scan"]["roll_candidates_deg"] = [0]  # otherwise a calmer roll is rightly chosen
    jumps = iter([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 100.0]] * 200)
    report = run(cfg, ScriptedRobot(lambda pose, ref: list(next(jumps))))
    assert any("would move" in e for e in report.errors)


def test_roll_with_the_smallest_joint_step_wins(sim_cfg):
    cfg = sim_cfg()
    cfg["scan"]["roll_candidates_deg"] = [0, 90]
    calls = iter(range(10_000))

    def ik(pose, ref):
        # Even calls are roll 0 (big step), odd are roll 90 (small step).
        return [40.0 if next(calls) % 2 == 0 else 1.0, 0, 0, 0, 0, 0]

    report = run(cfg, ScriptedRobot(ik))
    assert all(v.roll_deg == 90 for v in report.views)


def test_workspace_violation_is_an_error(sim_cfg):
    cfg = sim_cfg()
    cfg["scan"]["workspace_mm"] = {"x": [350, 900], "y": [-900, 900], "z": [100, 900]}
    report = run(cfg, SimulatedRobot(cfg["sim"]["start_pose"]))
    assert any("outside" in e for e in report.errors)


def test_path_through_the_object_is_an_error(sim_cfg):
    cfg = sim_cfg()
    # Start directly below, so the straight move up to the top view crosses the object.
    start = [*np.array(cfg["scan"]["center_mm"]) + [0, 0, -300], 180, 0, 0]
    report = run(cfg, SimulatedRobot(start))
    assert any("path passes" in e for e in report.errors)


def test_standoff_inside_keepout_is_an_error(sim_cfg):
    cfg = sim_cfg()
    cfg["scan"]["standoff_mm"] = 80
    assert any("keep-out" in e for e in run(cfg, SimulatedRobot(cfg["sim"]["start_pose"])).errors)


def test_a_ring_can_have_its_own_arc():
    from hhg_tof.config import load_config
    from hhg_tof import planning

    scan = load_config()["scan"]
    scan.update(center_mm=[500, 0, 100], object_radius_mm=50, elevations_deg=[70, 15], views_per_ring=[4, 5],
                ring_arcs_deg=[None, [-150, 300]], add_top_view=False)
    views = planning.plan_views(scan)
    assert [v.azimuth_deg for v in views[:4]] == [0, 90, 180, 270]
    assert sorted(v.azimuth_deg for v in views[4:]) == [-150, -75, 0, 75, 150]
