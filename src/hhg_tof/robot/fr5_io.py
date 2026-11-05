#!/usr/bin/env python3
"""
Low-level FR5 I/O — XML-RPC command channel (20003) + real-time state stream (20004).

PROVENANCE — READ BEFORE EDITING
--------------------------------
`FR5Commands`, `RealtimeState`, `describe_error`, `ERROR_NAMES` and the protocol
constant block below are COPIED VERBATIM from ``fr5_tcp_jog_gui.py``, which stays
in place unmodified as the manual teach/jog reference tool.

Do not "tidy up" the copied code. The byte offsets and the frame resync logic were
re-derived field-by-field from the installed FAIRINO C++ SDK
(fairino-cpp-sdk v2.3.4-robot3.9.4, ``robot_types.h`` struct ``_ROBOT_STATE_PKG``,
``#pragma pack(1)``) and reproduce the GUI's four offsets (6, 15, 16, 64) exactly.
That agreement is what validates both. Re-implementing the parser would throw away
the only cross-check we have while the robot is powered down.

The one intentional change is ADDITIVE and lives in a clearly marked section at the
bottom of `RealtimeState`: parsing of the safety / motion-status fields. The original
`_parse` body is untouched; the new fields are read by a separate `_parse_extended`.

Importing this module performs no I/O. Nothing here opens a socket until an
instance is constructed and explicitly used.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
import xmlrpc.client

# --------------------------------------------------------------------------
# Protocol constants (taken from libfairino: robot.cpp / robot_types.h)
# --------------------------------------------------------------------------

DEFAULT_ROBOT_IP = "192.168.58.2"
RPC_PORT = 20003          # XML-RPC command server
RT_PORT = 20004           # real-time state stream
RPC_TIMEOUT_S = 3.0

# StartJOG "ref" — which frame the jog is expressed in.
FRAME_JOINT = 0
FRAME_BASE = 2
FRAME_TOOL = 4
FRAME_WOBJ = 8

# StopJOG takes a *different* code than StartJOG for the same frame.
STOP_REF_FOR = {FRAME_JOINT: 1, FRAME_BASE: 3, FRAME_TOOL: 5, FRAME_WOBJ: 9}

FRAME_CHOICES = [
    ("Base frame", FRAME_BASE),
    ("Tool frame", FRAME_TOOL),
    ("Workpiece frame", FRAME_WOBJ),
]

# StartJOG "nb" — axis index. 1-3 translate, 4-6 rotate about the fixed axes.
AXIS_X, AXIS_Y, AXIS_Z = 1, 2, 3
AXIS_RX, AXIS_RY, AXIS_RZ = 4, 5, 6

DIR_NEG, DIR_POS = 0, 1

# How often we re-issue StartJOG while a button is held. Each StartJOG runs for
# at most `max_dis`, so re-issuing is what turns it into continuous motion.
JOG_REFRESH_HZ = 10.0

# Real-time packet layout. The struct is #pragma pack(1), so these are exact
# byte offsets from the start of the frame.
RT_HEADER = struct.Struct("<HBH")   # frame_head, frame_cnt, data_len
OFF_PROGRAM_STATE = 5               # uint8
OFF_ROBOT_STATE = 6                 # uint8
OFF_MAIN_CODE = 7                   # int32
OFF_SUB_CODE = 11                   # int32
OFF_ROBOT_MODE = 15                 # uint8
OFF_JT_CUR_POS = 16                 # double[6], deg
OFF_TL_CUR_POS = 64                 # double[6], mm + deg
RT_MIN_USEFUL_LEN = OFF_TL_CUR_POS + 48
RT_MAX_FRAME = 4096

ROBOT_STATE_NAMES = {1: "stopped", 2: "running", 3: "paused", 4: "drag-teach"}

# Error codes worth naming; anything else is reported numerically.
ERROR_NAMES = {
    -4: "XML-RPC command failed",
    -3: "XML-RPC communication failed",
    -2: "network communication error",
    -1: "other error",
    3: "wrong number of parameters",
    4: "parameter out of range",
    14: "instruction execution failed",
    18: "a program is already running",
    25: "computation failed",
    28: "inverse kinematics failed",
    29: "joint value out of limit",
    30: "non-resettable fault — power-cycle the control box",
    38: "singular pose",
    99: "safety stop triggered",
    112: "target pose unreachable",
}


def describe_error(code: int) -> str:
    if code == 0:
        return "ok"
    name = ERROR_NAMES.get(code)
    return f"error {code} ({name})" if name else f"error {code}"
# ==========================================================================
# ADDITION (not present in fr5_tcp_jog_gui.py) — safety / motion-status offsets
# ==========================================================================
# Derived from the SDK struct `_ROBOT_STATE_PKG` (robot_types.h, #pragma pack(1))
# by summing field widths from `frame_head` onward. The same summation reproduces
# the four offsets the GUI already uses (OFF_ROBOT_STATE=6, OFF_ROBOT_MODE=15,
# OFF_JT_CUR_POS=16, OFF_TL_CUR_POS=64), which is what validates the method.

OFF_JT_CUR_TOR = 384             # double[6], N*m  — joint torque (safety use)
OFF_FT_SENSOR_RAW_DATA = 452     # double[6]  — ROBOT-INTEGRATED FT (see warning)
OFF_FT_SENSOR_DATA = 500         # double[6]  — ROBOT-INTEGRATED FT (see warning)
OFF_FT_SENSOR_ACTIVE = 548       # uint8      — ROBOT-INTEGRATED FT (see warning)
OFF_EMERGENCY_STOP = 549         # uint8,  1 = E-stop pressed
OFF_MOTION_DONE = 550            # int32,  1 = motion completed
OFF_COLLISION_STATE = 559        # uint8,  1 = collision detected
OFF_SAFETY_STOP0 = 564           # uint8   — safety stop signal SI0
OFF_SAFETY_STOP1 = 565           # uint8   — safety stop signal SI1

# A frame must be at least this long before the safety block can be read.
RT_SAFETY_BLOCK_END = OFF_SAFETY_STOP1 + 1      # 566

# --------------------------------------------------------------------------
# ⚠  ROBOT-INTEGRATED FT  vs  EXPERIMENT EXTERNAL FT SENSOR
# --------------------------------------------------------------------------
# `ft_sensor_raw_data` / `ft_sensor_data` / `ft_sensor_active` in the FR5's own
# real-time stream belong to a force/torque sensor wired into the FAIRINO
# CONTROLLER. They are parsed here only so the safety block around them can be
# reached and so a future safety monitor can use them.
#
# They are NOT the force/torque sensor of the tactile experiment. The
# experiment's F/T device, its model, and its communication method are still
# UNVERIFIED, and it may not be connected to the robot controller at all.
#
# Therefore: nothing in this codebase may treat `robot_ft*` as the experiment's
# F/T measurement. The experiment F/T sensor gets its own module
# (`ft_interface.py`, a later task) and its own field names. Whether this stream
# carries any real data at all is UNVERIFIED until the robot is powered on and
# `robot_ft_active` is observed.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# XML-RPC command channel
# --------------------------------------------------------------------------


class _TimeoutTransport(xmlrpc.client.Transport):
    """Transport that bounds every call, so a dead robot can't hang the GUI."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self._timeout
        return conn


class FR5Commands:
    """Thin, thread-safe wrapper over the FR5's XML-RPC interface."""

    def __init__(self, ip: str, timeout: float = RPC_TIMEOUT_S) -> None:
        self.ip = ip
        self._lock = threading.Lock()
        self._proxy = xmlrpc.client.ServerProxy(
            f"http://{ip}:{RPC_PORT}/RPC2",
            transport=_TimeoutTransport(timeout),
            allow_none=True,
        )

    def _call(self, method: str, *args) -> int:
        with self._lock:
            result = getattr(self._proxy, method)(*args)
        try:
            return int(result)
        except (TypeError, ValueError):
            return 0

    # -- lifecycle ---------------------------------------------------------
    def get_sdk_version(self) -> int:
        # Cheap round-trip used purely to prove the RPC server is alive.
        return self._call("GetSDKVersion")

    def enable(self, on: bool) -> int:
        return self._call("RobotEnable", 1 if on else 0)

    def set_mode(self, manual: bool) -> int:
        """Mode(1) = manual, Mode(0) = auto. Jogging wants manual."""
        return self._call("Mode", 1 if manual else 0)

    def set_speed(self, percent: int) -> int:
        return self._call("SetSpeed", int(percent))

    # -- motion ------------------------------------------------------------
    def start_jog(
        self, ref: int, axis: int, direction: int, vel: float, acc: float, max_dis: float
    ) -> int:
        return self._call(
            "StartJOG", int(ref), int(axis), int(direction),
            float(vel), float(acc), float(max_dis),
        )

    def stop_jog(self, ref: int) -> int:
        """Decelerated stop for one frame."""
        return self._call("StopJOG", int(STOP_REF_FOR.get(ref, 3)))

    def imm_stop_jog(self) -> int:
        """Immediate stop of any jog, whatever frame it was started in."""
        return self._call("ImmStopJOG")

    def close(self) -> None:
        try:
            with self._lock:
                self._proxy("close")()
        except Exception:
            pass
# --------------------------------------------------------------------------
# Real-time state stream (port 20004)
# --------------------------------------------------------------------------


class RealtimeState:
    """Background reader for the FR5's 20004 state stream.

    Runs until stopped, reconnecting on its own. All published values are
    plain tuples/ints swapped in atomically, so the GUI can read them without
    locking.
    """

    def __init__(self, ip: str) -> None:
        self.ip = ip
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.connected = False
        self.last_update = 0.0
        self.tcp_pose: tuple[float, ...] | None = None
        self.joints: tuple[float, ...] | None = None
        self.robot_state = 0
        self.robot_mode = 0
        self.main_code = 0
        self.sub_code = 0

        # -- ADDITION: safety / motion-status block ------------------------
        # None means "not yet observed": every frame seen so far was shorter
        # than RT_SAFETY_BLOCK_END. That is distinct from an observed 0.
        self.safety_valid = False
        self.motion_done: int | None = None
        self.emergency_stop: int | None = None
        self.collision_state: int | None = None
        self.safety_stop0: int | None = None
        self.safety_stop1: int | None = None
        self.joint_torque: tuple[float, ...] | None = None

        # -- ADDITION: robot-integrated FT. NOT the experiment's F/T sensor.
        # See the warning block near OFF_FT_SENSOR_RAW_DATA before using these.
        self.robot_ft_raw: tuple[float, ...] | None = None
        self.robot_ft: tuple[float, ...] | None = None
        self.robot_ft_active: int | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.connected = False

    @property
    def is_fresh(self) -> bool:
        return self.connected and (time.monotonic() - self.last_update) < 1.5

    def _run(self) -> None:
        while not self._stop.is_set():
            sock = None
            try:
                sock = socket.create_connection((self.ip, RT_PORT), timeout=3.0)
                sock.settimeout(1.0)
                self.connected = True
                buf = bytearray()
                while not self._stop.is_set():
                    try:
                        chunk = sock.recv(8192)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk
                    self._consume(buf)
                    if len(buf) > 4 * RT_MAX_FRAME:  # resync guard
                        del buf[:-RT_MAX_FRAME]
            except OSError:
                pass
            finally:
                self.connected = False
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if not self._stop.is_set():
                self._stop.wait(1.0)

    def _consume(self, buf: bytearray) -> None:
        while True:
            idx = buf.find(b"\x5a\x5a")
            if idx < 0:
                # Keep the last byte: it might be the first half of a header.
                if len(buf) > 1:
                    del buf[: len(buf) - 1]
                return
            if idx:
                del buf[:idx]
            if len(buf) < RT_HEADER.size:
                return
            _, _cnt, data_len = RT_HEADER.unpack_from(buf, 0)
            total = 5 + data_len + 2  # header + payload + uint16 checksum
            if data_len <= 0 or total > RT_MAX_FRAME:
                del buf[:2]  # bogus header, keep hunting
                continue
            if len(buf) < total:
                return
            frame = bytes(buf[:total])
            del buf[:total]
            if len(frame) >= RT_MIN_USEFUL_LEN:
                self._parse(frame)

    def _parse(self, frame: bytes) -> None:
        try:
            joints = struct.unpack_from("<6d", frame, OFF_JT_CUR_POS)
            pose = struct.unpack_from("<6d", frame, OFF_TL_CUR_POS)
            self.robot_state = frame[OFF_ROBOT_STATE]
            self.robot_mode = frame[OFF_ROBOT_MODE]
            self.main_code = struct.unpack_from("<i", frame, OFF_MAIN_CODE)[0]
            self.sub_code = struct.unpack_from("<i", frame, OFF_SUB_CODE)[0]
        except struct.error:
            return
        self.joints = joints
        self.tcp_pose = pose
        self._parse_extended(frame)          # ADDITION
        self.last_update = time.monotonic()

    # -- ADDITION ---------------------------------------------------------
    def _parse_extended(self, frame: bytes) -> None:
        """Read the safety / motion-status block. Additive; see module docstring.

        Frames shorter than RT_SAFETY_BLOCK_END leave every field here at its
        previous value and are NOT an error. The original acceptance rule
        (`len(frame) >= RT_MIN_USEFUL_LEN` in `_consume`) is deliberately left
        untouched, so a short frame still yields a valid pose and joint reading
        exactly as it did in the jog GUI. Only the extra fields go unfilled.

        The real frame length is UNVERIFIED — it can only be measured against a
        powered-on controller — so this must stay defensive rather than assume
        the full struct is always present.
        """
        if len(frame) < RT_SAFETY_BLOCK_END:
            return
        try:
            motion_done = struct.unpack_from("<i", frame, OFF_MOTION_DONE)[0]
            joint_torque = struct.unpack_from("<6d", frame, OFF_JT_CUR_TOR)
            robot_ft_raw = struct.unpack_from("<6d", frame, OFF_FT_SENSOR_RAW_DATA)
            robot_ft = struct.unpack_from("<6d", frame, OFF_FT_SENSOR_DATA)
        except struct.error:
            return

        self.motion_done = motion_done
        self.emergency_stop = frame[OFF_EMERGENCY_STOP]
        self.collision_state = frame[OFF_COLLISION_STATE]
        self.safety_stop0 = frame[OFF_SAFETY_STOP0]
        self.safety_stop1 = frame[OFF_SAFETY_STOP1]
        self.joint_torque = joint_torque

        # Robot-integrated FT — NOT the experiment F/T sensor. See warning above.
        self.robot_ft_raw = robot_ft_raw
        self.robot_ft = robot_ft
        self.robot_ft_active = frame[OFF_FT_SENSOR_ACTIVE]

        self.safety_valid = True
