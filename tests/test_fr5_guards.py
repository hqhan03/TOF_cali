"""FR5Robot against a fake controller: gates, command order, and abort behaviour."""

import pytest

from hhg_tof import cli
from hhg_tof.robot import fr5, fr5_io
from hhg_tof.robot.base import IkFailed, MotionRefused, RobotError, SafetyStop
from tests.fake_fr5 import FakeFR5

POSE = [-410.0, 0.0, 300.0, 180.0, 0.0, 0.0]
JOINTS = [11.0, -90.0, 90.0, -90.0, -90.0, 0.0]


@pytest.fixture
def server(monkeypatch):
    fake = FakeFR5()
    monkeypatch.setattr(fr5_io, "RPC_PORT", fake.rpc.server_address[1])
    monkeypatch.setattr(fr5_io, "RT_PORT", fake.rt.getsockname()[1])
    monkeypatch.setattr(fr5, "STOP_PORT", fake.stop_sock.getsockname()[1])
    yield fake
    fake.close()


def connected(allow_motion=False, **kwargs):
    robot = fr5.FR5Robot("127.0.0.1", None, allow_motion=allow_motion, **kwargs)
    robot.connect()
    return robot


def test_connect_reads_state_and_sends_no_commands(server):
    robot = connected()
    status = robot.status()
    assert status.tcp_pose == server.pose
    assert status.fault is None
    robot.disconnect()
    assert server.writes == []


def test_nothing_moves_without_allow_motion(server):
    robot = connected()
    with pytest.raises(MotionRefused):
        robot.prepare_motion(10)
    with pytest.raises(MotionRefused):
        robot.move_linear(POSE, JOINTS, 10, 100)
    robot.disconnect()
    assert server.writes == []


def test_read_path_cannot_send_a_command(server):
    robot = connected()
    with pytest.raises(MotionRefused):
        robot._read("MoveL", [])
    robot.disconnect()


def test_auto_sequence_and_restore_match_verified_order(server):
    robot = connected(allow_motion=True)
    robot.prepare_motion(10)
    robot.move_linear(POSE, JOINTS, 20, 100)
    robot.restore()
    robot.disconnect()
    names = [c[0] for c in server.writes]
    assert names == ["Mode", "RobotEnable", "SetSpeed", "MoveL", "Mode"]
    assert server.writes[0][1] == (0,)
    assert server.writes[-1][1] == (1,)  # back to the manual mode it was found in


def test_movel_wire_format(server):
    robot = connected(allow_motion=True)
    robot.move_linear(POSE, JOINTS, 20, 100)
    robot.disconnect()
    (params,) = next(c for c in server.calls if c[0] == "MoveL")[1]
    assert len(params) == 33
    assert params[:6] == JOINTS and params[6:12] == POSE
    assert params[14] == 20 and params[16] == 100
    assert params[17] == -1.0  # blocking


def test_ik_uses_current_config(server):
    robot = connected()
    robot.solve_ik(POSE)
    robot.disconnect()
    assert next(c for c in server.calls if c[0] == "GetInverseKin")[1][2] == -1


def test_ik_failure_is_raised(server):
    server.ik_error = 112
    robot = connected()
    with pytest.raises(IkFailed):
        robot.solve_ik(POSE)
    robot.disconnect()


def test_speed_is_capped(server):
    robot = connected(allow_motion=True)
    with pytest.raises(MotionRefused):
        robot.prepare_motion(fr5.MAX_SPEED_PERCENT + 1)
    robot.disconnect()
    assert server.writes == []


def test_wrong_tool_is_refused(server):
    server.tool = 1
    robot = connected(allow_motion=True)
    with pytest.raises(MotionRefused, match="tool"):
        robot.prepare_motion(10)
    robot.disconnect()
    assert server.writes == []


def test_existing_fault_is_refused_and_not_reset(server):
    server.main_code = 5
    robot = connected(allow_motion=True)
    with pytest.raises(MotionRefused, match="not clean"):
        robot.prepare_motion(10)
    robot.disconnect()
    assert server.writes == []


def test_estop_during_move_stops_and_raises(server):
    robot = connected(allow_motion=True)
    server.move_seconds = 1.0
    import threading
    threading.Timer(0.1, lambda: setattr(server, "estop", 1)).start()
    with pytest.raises(SafetyStop):
        robot.move_linear(POSE, JOINTS, 20, 100)
    robot.disconnect()
    assert server.stop_packets == [fr5.STOP_PACKET]


def test_controller_rejecting_a_move_raises(server):
    server.move_result = 112
    robot = connected(allow_motion=True)
    with pytest.raises(RobotError, match="112"):
        robot.move_linear(POSE, JOINTS, 20, 100)
    robot.disconnect()


def test_wrong_network_route_is_refused(server, monkeypatch):
    monkeypatch.setattr(fr5, "route_interface", lambda ip: "wlp132s0")
    robot = fr5.FR5Robot("127.0.0.1", "enp131s0")
    with pytest.raises(RobotError, match="route"):
        robot.connect()


def real_config(tmp_path, server, *, workspace=True):
    path = tmp_path / "real.yaml"
    workspace_line = "workspace_mm: {x: [-900, 900], y: [-900, 900], z: [0, 900]}" if workspace else ""
    path.write_text(f"""
robot: {{backend: fr5, ip: 127.0.0.1, interface: null, max_joint_step_deg: 180}}
sensor: {{source: sim}}
mount: {{xyz_mm: [0, 40, 60], rpy_deg: [0, 0, 0]}}
scan:
  center_mm: [-450, 0, 60]
  object_radius_mm: 60
  {workspace_line}
sim: {{object: {{type: sphere, radius_mm: 60}}}}
""")
    return str(path)


def test_cli_without_execute_is_preflight_only(server, tmp_path):
    assert cli.main(["scan", "--config", real_config(tmp_path, server), "--out", str(tmp_path)]) == 0
    assert server.writes == []


def test_cli_execute_needs_the_confirm_word(server, tmp_path):
    code = cli.main(["scan", "--config", real_config(tmp_path, server), "--execute",
                     "--confirm", "yes", "--out", str(tmp_path)])
    assert code == 2
    assert server.calls == []


def test_cli_refuses_real_scan_without_workspace(server, tmp_path):
    code = cli.main(["scan", "--config", real_config(tmp_path, server, workspace=False),
                     "--execute", "--confirm", "SCAN", "--out", str(tmp_path)])
    assert code == 1
    assert server.writes == []


def test_cli_execute_runs_the_whole_real_path(server, tmp_path):
    code = cli.main(["scan", "--config", real_config(tmp_path, server), "--execute",
                     "--confirm", "SCAN", "--out", str(tmp_path), "--name", "e2e"])
    assert code == 0
    moves = [c for c in server.writes if c[0] == "MoveL"]
    assert len(moves) == 41  # 21 views, then 20 legs back to the top view
    assert server.writes[-1] == ("Mode", (1,))
    assert len(list(tmp_path.glob("*_e2e/view_*.npz"))) == 21


def test_command_layer_refuses_on_its_own(server):
    """The lowest gate, with nothing above it to help."""
    robot = connected()
    with pytest.raises(MotionRefused):
        robot._write("Mode", 0)
    robot.disconnect()
    assert server.writes == []


def test_command_layer_only_sends_listed_methods(server):
    robot = connected(allow_motion=True)
    with pytest.raises(MotionRefused):
        robot._write("ResetAllError")
    robot.disconnect()


def test_second_connection_cannot_cut_off_the_first(server):
    """The real controller drops the older client, which would blind a running scan."""
    first = connected()
    with pytest.raises(RobotError, match="owns the robot state"):
        connected()
    first.disconnect()
    connected().disconnect()


def test_failed_connect_releases_the_stream_lock(server, monkeypatch):
    monkeypatch.setattr(fr5_io, "RT_PORT", 1)
    monkeypatch.setattr(fr5_io.RealtimeState, "is_fresh", property(lambda self: False))
    with pytest.raises(RobotError):
        connected()
    robot = fr5.FR5Robot("127.0.0.1")
    assert robot._lock.acquire()
    robot._lock.release()


def test_restore_waits_until_the_stream_shows_the_mode(server, monkeypatch):
    """Approach then scan in one process left the real arm in auto: the second
    prepare_motion read the stale mode and restored to that."""
    robot = connected(allow_motion=True)
    robot.prepare_motion(10)
    reports = iter([0, 0, 1])
    real = robot.status

    def lagging():
        state = real()
        state.robot_mode = next(reports, 1)
        return state

    monkeypatch.setattr(robot, "status", lagging)
    robot.restore()
    assert next(reports, "done") == "done"
    robot.disconnect()
