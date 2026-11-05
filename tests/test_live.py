"""Live viewer core: placement rules, map, kinematics, telemetry, stream ownership."""

import time

import numpy as np
import pytest

from hhg_tof import live, telemetry
from hhg_tof.config import load_config
from hhg_tof.geometry import T_to_pose, inv_T, pose_to_T
from hhg_tof.robot.base import RobotError
from hhg_tof.sensors.base import Frame

POSE = [0.0, 0.0, 300.0, 180.0, 0.0, 0.0]  # looking straight down from z = 300


def cfg_with(mount=True, tool=0):
    cfg = load_config()
    cfg["robot"]["expected_tool"] = tool
    if mount:
        cfg["mount"]["xyz_mm"], cfg["mount"]["rpy_deg"] = [0, 0, 0], [0, 0, 0]
    return cfg


def flat_frame(t, range_mm=300.0):
    """Radial ranges of a flat wall `range_mm` away, so filters keep every zone."""
    from hhg_tof.sensor_model import zone_tangents

    xn, yn = zone_tangents(32, 48)
    distance = (range_mm * np.sqrt(1 + xn**2 + yn**2)).astype(np.float32)
    return Frame(distance, 1, t)


def feed_robot(proc, t_end, moving=False, tool=0):
    for i, t in enumerate(np.arange(t_end - 1.0, t_end, 1 / 30)):
        pose = list(POSE)
        pose[0] += 3.0 * i if moving else 0.0
        proc.add_robot({"t": float(t), "tcp_pose": pose, "tool": tool})


def test_still_frame_lands_on_the_plane_and_enters_the_map():
    proc = live.LiveProcessor(cfg_with())
    feed_robot(proc, 100.0)
    result = proc.add_frame(flat_frame(100.0))
    assert result.still and result.placement == "placed"
    assert np.allclose(result.base_points[:, 2], 0.0, atol=1e-3)
    assert len(proc.map.points) > 100


def test_moving_frame_is_shown_but_not_mapped():
    proc = live.LiveProcessor(cfg_with())
    feed_robot(proc, 100.0, moving=True)
    result = proc.add_frame(flat_frame(100.0))
    assert not result.still and result.base_points is not None
    assert len(proc.map.points) == 0


def test_without_mount_points_stay_in_the_sensor_frame():
    proc = live.LiveProcessor(cfg_with(mount=False))
    feed_robot(proc, 100.0)
    result = proc.add_frame(flat_frame(100.0))
    assert result.base_points is None and result.T_base_sensor is None
    assert len(result.sensor_points) > 100
    assert len(proc.map.points) == 0


def test_stale_pose_is_not_used():
    proc = live.LiveProcessor(cfg_with())
    feed_robot(proc, 100.0)
    result = proc.add_frame(flat_frame(102.0))
    assert result.base_points is None
    assert len(proc.map.points) == 0


def test_wrong_active_tool_refuses_placement():
    """Tool 1 on this robot moves the reported pose by 136 mm."""
    proc = live.LiveProcessor(cfg_with(tool=0))
    feed_robot(proc, 100.0, tool=1)
    result = proc.add_frame(flat_frame(100.0))
    assert result.base_points is None and "tool 1" in result.placement


def test_gap_in_the_stream_does_not_count_as_still():
    history = live.PoseHistory()
    history.add(9.0, POSE)
    history.add(10.0, POSE)
    assert not history.still(9.6, 9.95)


def test_voxel_map_keeps_one_point_per_voxel_and_stays_bounded():
    voxels = live.VoxelMap(4.0, max_points=3)
    voxels.add(np.array([[1.0, 1, 1], [2.0, 2, 2]]))
    assert len(voxels.points) == 1
    assert np.allclose(voxels.points[0], 2.0)
    voxels.add(np.arange(30.0).reshape(10, 3) * 10)
    assert len(voxels.points) == 3


def test_kinematics_reproduce_the_real_robot():
    """Joints and pose read from this FR5 with tool 1 (136.4 mm along flange Z)."""
    joints = [112.43, -68.68, -78.26, -194.97, -90.0, -114.25]
    tcp = [-34.96, 351.21, 708.44, -51.5, 60.08, 57.02]
    tool = T_to_pose(inv_T(live.arm_transforms(joints)[-1]) @ pose_to_T(tcp))
    assert np.allclose(tool, [0, 0, 136.4, 0, 0, 0], atol=0.6)


def test_telemetry_reaches_a_subscriber():
    pytest.importorskip("zmq")
    pub = telemetry.TelemetryPublisher(lambda: {"t": 1.0}, port=5571, rate_hz=100)
    pub.start()
    sub = telemetry.TelemetrySubscriber(port=5571)
    try:
        deadline = time.monotonic() + 3
        messages = []
        while not messages and time.monotonic() < deadline:
            messages = sub.poll()
            time.sleep(0.01)
        assert messages[0] == {"t": 1.0}
    finally:
        sub.close()
        pub.stop()


def test_telemetry_survives_a_failing_read():
    pytest.importorskip("zmq")
    calls = []

    def read():
        calls.append(1)
        raise RobotError("robot state stream is stale")

    pub = telemetry.TelemetryPublisher(read, port=5572, rate_hz=200)
    pub.start()
    time.sleep(0.2)
    pub.stop()
    assert len(calls) > 3


def test_second_process_cannot_take_the_state_stream():
    first = telemetry.StreamOwnerLock("10.9.8.7")
    second = telemetry.StreamOwnerLock("10.9.8.7")
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_demo_records_every_stream(tmp_path):
    pytest.importorskip("rerun")
    from hhg_tof import live_view

    cfg = load_config("config/sim_box.yaml")
    out = tmp_path / "demo.rrd"
    live_view.run(cfg, None, demo=True, save=str(out), window=False, duration_s=3.0)
    assert out.stat().st_size > 50_000
