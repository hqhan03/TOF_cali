"""Mount calibration from hand-jogged poses, with no starting guess for the mount."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hhg_tof import calibration, capture, cli, planning, session as session_io
from hhg_tof.config import load_config
from hhg_tof.geometry import T_to_pose, inv_T, pose_to_T
from hhg_tof.robot.simulated import SimulatedRobot
from hhg_tof.sensors.simulated import Box, Plane, SimNoise, SimulatedSensor

# Taped on by hand: nowhere near any CAD value, and tilted a long way off the flange axis.
TRUE_MOUNTS = {
    "near_axis": ([35.0, -60.0, 20.0], [8.0, -12.0, 40.0]),
    "sideways": ([70.0, 25.0, 45.0], [95.0, 15.0, -130.0]),
    "backwards": ([-40.0, 55.0, -30.0], [170.0, 25.0, 100.0]),
}


def jog_cfg(true_mount, mirror=False, bias=0.0):
    cfg = load_config()
    cfg["robot"]["backend"] = "sim"
    cfg["sensor"]["source"] = "sim"
    cfg["sensor"]["frames_per_view"] = 5
    cfg["sim"].update(true_mount_xyz_mm=true_mount[0], true_mount_rpy_deg=true_mount[1],
                      true_mirror_columns=mirror, range_bias_mm=bias)
    return cfg


def jog_poses(true_T, count, tilt_deg=(15, 35), seed=3):
    """What a person jogging over a table produces: varied tilt direction and roll."""
    rng = np.random.default_rng(seed)
    for i in range(count):
        down = Rotation.from_euler("x", 180, degrees=True)
        tilt = Rotation.from_euler("zx", [rng.uniform(0, 360), rng.uniform(*tilt_deg)], degrees=True)
        roll = Rotation.from_euler("z", i * 97.0, degrees=True)
        T = np.eye(4)
        T[:3, :3] = (tilt * down * roll).as_matrix()
        T[:3, 3] = T[:3, :3] @ [0, 0, -rng.uniform(220, 380)]  # keep the same table patch in view
        yield T_to_pose(T @ inv_T(true_T))


CLUTTER = [Box([60, -40, 30], [80, 50, 60]), Box([-90, 70, 20], [50, 120, 40])]


def take_views(cfg, count=12, scene=(), **kwargs):
    sim = cfg["sim"]
    true_T = pose_to_T([*sim["true_mount_xyz_mm"], *sim["true_mount_rpy_deg"]])
    robot = SimulatedRobot([0, 0, 500, 180, 0, 0])
    noise = SimNoise(sim["noise_mm"], sim["noise_rel"], sim["dropout"], sim["range_bias_mm"])
    sensor = SimulatedSensor(robot, [Plane(0.0), *scene], true_T, *cfg["sensor"]["grid"], noise,
                             sim["true_mirror_columns"])
    views = []
    for index, pose in enumerate(jog_poses(true_T, count, **kwargs)):
        robot.pose = pose
        views.append(capture.measure_view(index, pose, cfg, robot, sensor))
    return views


@pytest.mark.parametrize("name", TRUE_MOUNTS)
def test_recovers_a_taped_mount_with_no_starting_guess(name):
    true_mount = TRUE_MOUNTS[name]
    cfg = jog_cfg(true_mount, bias=9.0)
    results = calibration.calibrate_mount(take_views(cfg), cfg)
    choice, _ = calibration.recommend(results)
    assert choice is False
    best = results[False]
    assert not calibration.unobserved(best)
    assert np.allclose(best.xyz_mm, true_mount[0], atol=2.0)
    angle = (Rotation.from_euler("xyz", best.rpy_deg, degrees=True).inv()
             * Rotation.from_euler("xyz", true_mount[1], degrees=True)).magnitude()
    assert np.degrees(angle) < 0.5
    assert best.range_bias_mm == pytest.approx(9.0, abs=2.0)


def test_objects_on_the_desk_do_not_spoil_the_fit():
    true_mount = TRUE_MOUNTS["sideways"]
    cfg = jog_cfg(true_mount, bias=9.0)
    results = calibration.calibrate_mount(take_views(cfg, scene=CLUTTER), cfg)
    assert calibration.recommend(results)[0] is False
    assert not calibration.unobserved(results[False])
    assert np.allclose(results[False].xyz_mm, true_mount[0], atol=3.0)


def test_dominant_plane_rejects_a_scene_with_no_plane():
    rng = np.random.default_rng(1)
    assert not calibration.dominant_plane(rng.uniform(-200, 200, (300, 3)), rng).any()


def test_mirrored_sensor_is_detected_without_a_starting_guess():
    cfg = jog_cfg(TRUE_MOUNTS["sideways"], mirror=True)
    results = calibration.calibrate_mount(take_views(cfg), cfg)
    assert calibration.recommend(results)[0] is True


def test_too_little_tilt_is_not_accepted():
    cfg = jog_cfg(TRUE_MOUNTS["near_axis"])
    views = take_views(cfg, tilt_deg=(0.5, 1.5))
    results = calibration.calibrate_mount(views, cfg)
    choice, _ = calibration.recommend(results)
    assert choice is None or calibration.unobserved(results[choice])


def test_jog_command_saves_views_and_never_moves_the_arm(tmp_path, monkeypatch, capsys):
    cfg = jog_cfg(TRUE_MOUNTS["near_axis"])
    sim = cfg["sim"]
    true_T = pose_to_T([*sim["true_mount_xyz_mm"], *sim["true_mount_rpy_deg"]])
    robot = SimulatedRobot([0, 0, 500, 180, 0, 0])
    sensor = SimulatedSensor(robot, [Plane(0.0)], true_T, 32, 48, SimNoise(1.0, 0.004, 0.02, 0.0))
    poses = iter(jog_poses(true_T, 8))

    def press_enter(prompt=""):
        pose = next(poses, None)
        if pose is None:
            return "q"
        robot.pose = pose
        return ""

    monkeypatch.setattr("builtins.input", press_enter)
    path = capture.run_jog_capture(cfg, robot, sensor, tmp_path / "s")
    assert len(session_io.read_views(path)) == 8
    assert session_io.read_meta(path)["status"] == "complete"
    assert robot.commands == []
    assert "ENOUGH" in capsys.readouterr().out


def test_jog_capture_refuses_the_wrong_tool(tmp_path):
    cfg = jog_cfg(TRUE_MOUNTS["near_axis"])
    cfg["robot"]["expected_tool"] = 1
    robot = SimulatedRobot([0, 0, 500, 180, 0, 0])
    with pytest.raises(capture.RobotError, match="tool"):
        capture.run_jog_capture(cfg, robot, None, tmp_path / "s")


class JoggedRobot(SimulatedRobot):
    """A person jogging: the pose changes every few reads, then holds."""

    def __init__(self, poses, reads_per_pose=12):
        super().__init__(poses[0])
        self.poses, self.reads_per_pose, self.reads = poses, reads_per_pose, 0

    def status(self):
        self.reads += 1
        self.pose = list(self.poses[min(self.reads // self.reads_per_pose, len(self.poses) - 1)])
        return super().status()


def auto_rig(poses, true_T):
    robot = JoggedRobot(poses)
    return robot, SimulatedSensor(robot, [Plane(0.0)], true_T, 32, 48, SimNoise(1.0, 0.004, 0.02, 0.0))


def test_auto_capture_saves_each_new_rest_and_skips_repeats(tmp_path):
    cfg = jog_cfg(TRUE_MOUNTS["near_axis"])
    sim = cfg["sim"]
    true_T = pose_to_T([*sim["true_mount_xyz_mm"], *sim["true_mount_rpy_deg"]])
    distinct = list(jog_poses(true_T, 3))
    robot, sensor = auto_rig([distinct[0], distinct[0], distinct[1], distinct[2]], true_T)
    path = capture.run_jog_capture(cfg, robot, sensor, tmp_path / "s", auto=True, timeout_s=3.0, hold_s=0.2)
    assert len(session_io.read_views(path)) == 3
    assert robot.commands == []


def test_auto_capture_rejects_a_view_with_no_target_and_keeps_going(tmp_path):
    cfg = jog_cfg(TRUE_MOUNTS["near_axis"])
    sim = cfg["sim"]
    true_T = pose_to_T([*sim["true_mount_xyz_mm"], *sim["true_mount_rpy_deg"]])
    poses = list(jog_poses(true_T, 2))
    sky = T_to_pose(pose_to_T([0, 0, 300, 0, 0, 0]) @ inv_T(true_T))  # looking up, sees nothing
    robot, sensor = auto_rig([sky, *poses], true_T)
    path = capture.run_jog_capture(cfg, robot, sensor, tmp_path / "s", auto=True, timeout_s=2.0, hold_s=0.2)
    assert len(session_io.read_views(path)) == 2


def test_a_fault_while_jogging_pauses_capture_instead_of_ending_it(tmp_path, capsys):
    class FaultyRobot(JoggedRobot):
        def status(self):
            status = super().status()
            status.main_code = 3 if 3 <= self.reads < 15 else 0
            return status

    cfg = jog_cfg(TRUE_MOUNTS["near_axis"])
    sim = cfg["sim"]
    true_T = pose_to_T([*sim["true_mount_xyz_mm"], *sim["true_mount_rpy_deg"]])
    robot = FaultyRobot(list(jog_poses(true_T, 2)))
    sensor = SimulatedSensor(robot, [Plane(0.0)], true_T, 32, 48, SimNoise(1.0, 0.004, 0.02, 0.0))
    path = capture.run_jog_capture(cfg, robot, sensor, tmp_path / "s", auto=True, timeout_s=2.0, hold_s=0.2)
    out = capsys.readouterr().out
    assert "paused" in out and "resuming" in out
    assert len(session_io.read_views(path)) >= 1


START = [393.6, 176.4, 475.7, -165.6, 3.6, -114.6]


def test_wrist_views_stay_near_the_tcp_and_vary_tilt_and_roll():
    views = planning.wrist_views(START)
    points = np.array([v.tcp_pose[:3] for v in views])
    assert np.allclose(points[:, :2], START[:2], atol=1e-6)
    assert np.abs(points[:, 2] - START[2]).max() == pytest.approx(40.0)
    gaps = [capture._rotation_gap_deg(START, v.tcp_pose) for v in views]
    assert 8 < gaps[0] < 45 and max(gaps) < 60
    assert np.mean(gaps[:6]) < np.mean(gaps[6:])          # gentle moves first
    assert len({round(v.roll_deg) for v in views}) > 5


class LimitedRobot(SimulatedRobot):
    joints = [41.6, -103.6, -67.7, -240.0, -78.3, -114.2]

    def status(self):
        status = super().status()
        status.joints = list(self.joints)
        return status

    def solve_ik(self, pose, ref_joints=None):
        return list(self.joints)


def test_wrist_preflight_passes_a_clean_arm():
    cfg = load_config()
    views = planning.wrist_views(START)
    assert planning.wrist_preflight(views, cfg, LimitedRobot(START)).ok


@pytest.mark.parametrize("change, expected", [
    (dict(main_code=3, sub_code=4), "not clean"),
    (dict(tool_id=1), "tool"),
])
def test_wrist_preflight_refuses_a_bad_controller_state(change, expected):
    class Bad(LimitedRobot):
        def status(self):
            status = super().status()
            for key, value in change.items():
                setattr(status, key, value)
            return status

    report = planning.wrist_preflight(planning.wrist_views(START), load_config(), Bad(START))
    assert not report.ok and any(expected in e for e in report.errors)


def test_wrist_preflight_refuses_an_arm_past_a_joint_limit():
    """J4 at -269.4 deg is what this robot reported with fault main=3 sub=4."""
    class Over(LimitedRobot):
        joints = [41.6, -103.6, -67.7, -269.4, -78.3, -114.2]

    report = planning.wrist_preflight(planning.wrist_views(START), load_config(), Over(START))
    assert not report.ok and any("joint limit" in e for e in report.errors)


def test_wrist_scan_refuses_without_the_confirm_word(capsys):
    assert cli.main(["wrist-scan", "--config", "config/fr5_evm_live.yaml", "--execute"]) == 2


def wrist_session(cfg, center, scene=()):
    sim = cfg["sim"]
    true_T = pose_to_T([*sim["true_mount_xyz_mm"], *sim["true_mount_rpy_deg"]])
    robot = SimulatedRobot(center)
    noise = SimNoise(sim["noise_mm"], sim["noise_rel"], sim["dropout"], sim["range_bias_mm"])
    sensor = SimulatedSensor(robot, [Plane(0.0), *scene], true_T, 32, 48, noise, sim["true_mirror_columns"])
    views = []
    for wrist in planning.wrist_views(center):
        robot.pose = wrist.tcp_pose
        views.append(capture.measure_view(wrist.index, wrist.tcp_pose, cfg, robot, sensor))
    return views


NEAR_FLANGE = ([12.0, -20.0, 25.0], [1.0, 2.5, 25.0])
LEVEL = [0.0, 0.0, 330.0, 180.0, 0.0, 30.0]


@pytest.mark.parametrize("mirror", [False, True])
def test_vertical_step_settles_the_mirror_for_wrist_views(mirror):
    cfg = jog_cfg(NEAR_FLANGE, mirror=mirror, bias=8.0)
    results = calibration.calibrate_mount(wrist_session(cfg, LEVEL), cfg)
    assert calibration.recommend(results)[0] is mirror
    best = results[mirror]
    assert not calibration.unobserved(best)
    assert np.allclose(best.xyz_mm, NEAR_FLANGE[0], atol=3.0)


def test_parallel_layers_do_not_mix():
    """A wide flat object 75 mm above the table, covering about half of each view."""
    cfg = jog_cfg(NEAR_FLANGE, mirror=True, bias=8.0)
    shelf = [Box([0, 130, 37.5], [600, 260, 75])]
    results = calibration.calibrate_mount(wrist_session(cfg, LEVEL, shelf), cfg)
    assert calibration.recommend(results)[0] is True
    assert results[True].rms_mm < 3.0


def test_level_pose_points_tool_z_down_and_keeps_position():
    level = planning.level_pose([75.7, 253.5, 386.8, -124.7, 22.6, -34.9])
    assert np.allclose(pose_to_T(level)[:3, 2], [0, 0, -1], atol=1e-9)
    assert np.allclose(level[:3], [75.7, 253.5, 386.8])


class StepIk(LimitedRobot):
    """IK whose joints scale with distance from the start, to exercise leg splitting."""

    def solve_ik(self, pose, ref_joints=None):
        joints = list(self.joints)
        joints[3] += (START[2] - pose[2]) * 0.5
        return joints


def test_approach_splits_a_long_move_into_small_legs():
    target = list(START)
    target[2] -= 100.0  # 50 deg of J4 in one go
    legs, errors = planning.approach_path(START, target, LimitedRobot.joints, load_config(), StepIk(START))
    assert not errors and len(legs) == 2
    assert np.allclose(legs[-1][0], target, atol=1e-6)


def test_approach_runs_every_leg_and_restores(monkeypatch):
    robot = LimitedRobot(START)
    calls = []
    monkeypatch.setattr(robot, "restore", lambda: calls.append("restore"))
    target = list(START)
    target[2] -= 50
    capture.run_approach([(START, LimitedRobot.joints), (target, LimitedRobot.joints)], load_config(), robot)
    assert robot.commands == [START, target] and calls == ["restore"]
