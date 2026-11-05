"""Fairino FR5 access for scanning. THIS CLASS CAN MOVE THE ROBOT.

Facts below were measured on this robot by KDC-tactile-testing-platform, whose
fr5_io.py is copied here unmodified (verified again 2026-11-05: 1236-byte frames,
so the safety block is present).

- Auto motion needs Mode(0) -> RobotEnable(1) -> SetSpeed. The jog GUI's order
  is a different, manual path.
- MoveL's 33-value array and blendR = -1 (blocking) follow move_probe.py.
- GetInverseKin must use config = -1. For one unchanged pose, config 0 gave a
  solution 219 deg from the current joints. Same point, a different arm posture.
- A blocking MoveL occupies its RPC, so the 20004 stream is the only live view
  during a move. It agrees with GetActualTCPPose to 2 um.

NOT verified on hardware: stop(). StopMotion is not XML-RPC; the SDK sends it as
a text packet on TCP 8080, reproduced here from robot.cpp. Treat the teach
pendant E-stop as the real stop until someone has tested this one.

Guards: motion needs allow_motion=True, and the RPC layer refuses any method not
on an explicit list, so a typo cannot reach the controller as a command.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
import xmlrpc.client
from typing import Sequence

import numpy as np

from ..telemetry import StreamOwnerLock, TelemetryPublisher
from . import fr5_io
from .base import IkFailed, MotionRefused, RobotError, RobotStatus, SafetyStop

log = logging.getLogger(__name__)

READ_METHODS = {
    "GetActualTCPNum", "GetActualWObjNum", "GetInverseKin", "GetInverseKinRef", "GetSDKVersion",
}
WRITE_METHODS = {"Mode", "RobotEnable", "SetSpeed", "MoveL"}

STOP_PORT = 8080
STOP_PACKET = b"/f/bIII44III102III4IIISTOPIII/b/f"

MAX_SPEED_PERCENT = 30
MOVE_TIMEOUT_S = 120.0
MONITOR_PERIOD_S = 0.02
MODE_SETTLE_S = 2.0


def _proxy(ip: str, timeout: float) -> xmlrpc.client.ServerProxy:
    return xmlrpc.client.ServerProxy(
        f"http://{ip}:{fr5_io.RPC_PORT}", transport=fr5_io._TimeoutTransport(timeout), allow_none=True
    )


def route_interface(ip: str) -> str | None:
    """Name of the interface the kernel would use to reach `ip`."""
    try:
        words = subprocess.run(["ip", "route", "get", ip], capture_output=True, text=True,
                               timeout=2).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        return None
    return words[words.index("dev") + 1] if "dev" in words else None


def build_move_l(joints: Sequence[float], pose: Sequence[float], tool: int, vel: float, ovl: float) -> list:
    params = ([float(v) for v in joints] + [float(v) for v in pose]
              + [int(tool), 0,                         # tool, user
                 float(vel), 0.0, float(ovl),          # vel, acc (not open), ovl
                 -1.0, 0,                              # blendR -1 = blocking, blendMode
                 0.0, 0.0, 0.0, 0.0,                   # external axes
                 0, 0,                                 # search, offset_flag
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,         # offset pose
                 100.0, 0])                            # oacc, velAccParamMode
    assert len(params) == 33
    return params


class FR5Robot:
    def __init__(self, ip: str = fr5_io.DEFAULT_ROBOT_IP, interface: str | None = None,
                 expected_tool: int = 0, expected_wobj: int = 0, allow_motion: bool = False):
        self.ip, self.interface = ip, interface
        self.expected_tool, self.expected_wobj = expected_tool, expected_wobj
        self.allow_motion = allow_motion
        self._rt: fr5_io.RealtimeState | None = None
        self._rpc = None
        self._tool = self._wobj = None
        self._mode_before: int | None = None
        self._lock = StreamOwnerLock(ip)
        self._telemetry: TelemetryPublisher | None = None

    # ------------------------------------------------------------- transport

    def _read(self, method: str, *args):
        if method not in READ_METHODS:
            raise MotionRefused(f"{method} is not on the read-only list")
        return getattr(self._rpc, method)(*args)

    def _write(self, method: str, *args, proxy=None) -> int:
        if not self.allow_motion:
            raise MotionRefused(f"{method} refused: motion was not enabled")
        if method not in WRITE_METHODS:
            raise MotionRefused(f"{method} is not on the command list")
        code = int(getattr(proxy or self._rpc, method)(*args))
        if code != 0:
            raise RobotError(f"{method} failed: {fr5_io.describe_error(code)}")
        return code

    # ------------------------------------------------------------ connection

    def connect(self) -> None:
        """Open observation paths only. Does not enable servos or change mode."""
        if self.interface:
            actual = route_interface(self.ip)
            if actual != self.interface:
                # On WiFi, XML-RPC to this address would leave on the office network.
                raise RobotError(f"route to {self.ip} uses {actual}, expected {self.interface}")
        # 20004 serves one client: connecting would cut off a running scan's safety monitor.
        if not self._lock.acquire():
            raise RobotError("another hhg-tof process (scan or monitor) owns the robot state "
                             "stream; connecting would cut it off. Stop that process first.")
        try:
            self._open_streams()
        except BaseException:
            self.disconnect()
            raise
        self._telemetry = TelemetryPublisher(self._telemetry_message)
        self._telemetry.start()

    def _open_streams(self) -> None:
        self._rpc = _proxy(self.ip, fr5_io.RPC_TIMEOUT_S)
        self._read("GetSDKVersion")
        self._rt = fr5_io.RealtimeState(self.ip)
        self._rt.start()
        deadline = time.monotonic() + 3.0
        while not (self._rt.is_fresh and self._rt.safety_valid):
            if time.monotonic() > deadline:
                raise RobotError("no complete state frame on port 20004 within 3 s")
            time.sleep(0.02)
        self._tool = self._read("GetActualTCPNum", 0)[1]
        self._wobj = self._read("GetActualWObjNum", 0)[1]

    def disconnect(self) -> None:
        if self._telemetry:
            self._telemetry.stop()
            self._telemetry = None
        if self._rt:
            self._rt.stop()
            self._rt = None
        self._lock.release()

    def _telemetry_message(self) -> dict:
        s = self.status()
        return {"t": time.time(), "tcp_pose": s.tcp_pose, "joints": s.joints, "tool": s.tool_id,
                "wobj": s.wobj_id, "mode": s.robot_mode, "fault": s.fault or ""}

    def status(self) -> RobotStatus:
        rt = self._rt
        if rt is None or not rt.is_fresh:
            raise RobotError("robot state stream is stale")
        return RobotStatus(
            tcp_pose=list(rt.tcp_pose), joints=list(rt.joints), robot_mode=rt.robot_mode,
            main_code=rt.main_code, sub_code=rt.sub_code, motion_done=rt.motion_done,
            emergency_stop=rt.emergency_stop, collision_state=rt.collision_state,
            tool_id=self._tool, wobj_id=self._wobj,
        )

    # -------------------------------------------------------------- planning

    def solve_ik(self, pose: Sequence[float], ref_joints: Sequence[float] | None = None) -> list[float]:
        """Joints for `pose`, staying on the arm configuration of `ref_joints`.

        Without a reference, config -1 solves from the current joints. That is
        the path verified on this robot; the reference form lets a preflight
        chain a whole scan before the arm has gone anywhere.
        """
        target = [float(v) for v in pose]
        if ref_joints is None:
            result = self._read("GetInverseKin", 0, target, -1)
        else:
            result = self._read("GetInverseKinRef", 0, target, [float(v) for v in ref_joints])
        if result[0] != 0:
            raise IkFailed(fr5_io.describe_error(int(result[0])))
        return [float(v) for v in result[1:7]]

    # ---------------------------------------------------------------- motion

    def prepare_motion(self, speed_percent: int) -> None:
        if not self.allow_motion:
            raise MotionRefused("prepare_motion refused: motion was not enabled")
        if not 1 <= speed_percent <= MAX_SPEED_PERCENT:
            raise MotionRefused(f"speed {speed_percent}% is outside 1..{MAX_SPEED_PERCENT}")
        status = self.status()
        if status.fault:
            # Deliberately no ResetAllError: a fault is for a person to look at.
            raise MotionRefused(f"robot is not clean: {status.fault}")
        if (self._tool, self._wobj) != (self.expected_tool, self.expected_wobj):
            raise MotionRefused(
                f"active tool/wobj is {self._tool}/{self._wobj}, config expects "
                f"{self.expected_tool}/{self.expected_wobj}. The 20004 pose is reported in "
                "the active frames, so the sensor mount would be applied to the wrong frame."
            )
        self._mode_before = status.robot_mode
        self._write("Mode", 0)
        self._write("RobotEnable", 1)
        self._write("SetSpeed", int(speed_percent))

    def move_linear(self, pose: Sequence[float], joints: Sequence[float], vel: float, ovl: float) -> None:
        """Blocking MoveL, watched from the state stream while it runs."""
        if not self.allow_motion:
            raise MotionRefused("MoveL refused: motion was not enabled")
        params = build_move_l(joints, pose, self.expected_tool, vel, ovl)
        result: dict = {}

        def call() -> None:
            try:
                self._write("MoveL", params, proxy=_proxy(self.ip, MOVE_TIMEOUT_S))
            except Exception as exc:  # noqa: BLE001 - re-raised on the main thread
                result["error"] = exc

        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        try:
            while worker.is_alive():
                status = self.status()
                if status.fault:
                    raise SafetyStop(status.fault)
                time.sleep(MONITOR_PERIOD_S)
        except BaseException:
            self.stop()
            raise
        if "error" in result:
            raise result["error"]

    def wait_settled(self, timeout_s: float, tol_mm: float, hold_s: float) -> None:
        """Return once the pose has stayed within `tol_mm` for `hold_s`."""
        deadline = time.monotonic() + timeout_s
        anchor = np.array(self.status().tcp_pose[:3])
        since = time.monotonic()
        while time.monotonic() < deadline:
            status = self.status()
            if status.fault:
                raise SafetyStop(status.fault)
            now = np.array(status.tcp_pose[:3])
            if np.linalg.norm(now - anchor) > tol_mm or status.motion_done != 1:
                anchor, since = now, time.monotonic()
            elif time.monotonic() - since >= hold_s:
                return
            time.sleep(MONITOR_PERIOD_S)
        raise RobotError(f"arm did not settle within {timeout_s} s")

    def stop(self) -> None:
        """Best-effort stop. Unverified on hardware, see the module docstring."""
        try:
            with socket.create_connection((self.ip, STOP_PORT), timeout=1.0) as sock:
                sock.sendall(STOP_PACKET)
        except OSError as exc:
            log.error("STOP could not be sent (%s). USE THE E-STOP.", exc)

    def restore(self) -> None:
        """Put the controller back in the mode it was found in. Servos stay on."""
        if self._mode_before is not None and self.allow_motion:
            target = self._mode_before
            self._write("Mode", target)
            self._mode_before = None
            # The stream reports the change a moment later. A prepare_motion straight after
            # this would otherwise record the old mode as the one to restore, and the arm
            # would be left in auto. Seen on 2026-11-05: approach, then scan, in one process.
            deadline = time.monotonic() + MODE_SETTLE_S
            while self.status().robot_mode != target:
                if time.monotonic() > deadline:
                    raise RobotError(f"controller did not report mode {target} after restoring it")
                time.sleep(MONITOR_PERIOD_S)
