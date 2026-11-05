"""A fake controller speaking the real wire formats, for exercising FR5Robot.

Serves XML-RPC, a 20004 byte stream built to the real frame layout, and the STOP
port, on ephemeral localhost ports, and records every call it receives.
"""

import socket
import socketserver
import struct
import threading
import time
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer

from hhg_tof.robot import fr5_io

FRAME_LEN = 1236  # measured on this robot


class _Quiet(SimpleXMLRPCRequestHandler):
    def log_message(self, *args):
        pass


class FakeFR5:
    def __init__(self):
        self.calls: list[tuple] = []
        self.tool, self.wobj = 0, 0
        self.pose = [-300.0, 0.0, 400.0, 180.0, 0.0, 0.0]
        self.joints = [10.0, -90.0, 90.0, -90.0, -90.0, 0.0]
        self.mode = 1
        self.estop = 0
        self.main_code = 0
        self.ik_error = 0
        self.ik_joints = None
        self.move_seconds = 0.0
        self.move_result = 0
        self.stop_packets: list[bytes] = []
        self._alive = True

        self.rpc = SimpleXMLRPCServer(("127.0.0.1", 0), requestHandler=_Quiet, allow_none=True,
                                      logRequests=False)
        self.rpc.register_instance(self, allow_dotted_names=False)
        self.rpc.register_function(self._dispatch_any("GetSDKVersion", [0, "fake"]), "GetSDKVersion")
        self._spawn(self.rpc.serve_forever)

        self.rt = socket.create_server(("127.0.0.1", 0))
        self._spawn(self._serve_rt)
        self.stop_sock = socket.create_server(("127.0.0.1", 0))
        self._spawn(self._serve_stop)

    def _spawn(self, fn):
        threading.Thread(target=fn, daemon=True).start()

    def _dispatch_any(self, name, value):
        def fn(*args):
            self.calls.append((name, args))
            return value
        return fn

    # Commands. Registered as an instance so each is one plain method.
    def _record(self, name, *args):
        self.calls.append((name, args))

    def GetActualTCPNum(self, flag):
        return [0, self.tool]

    def GetActualWObjNum(self, flag):
        return [0, self.wobj]

    def GetInverseKin(self, type_, pose, config):
        self._record("GetInverseKin", type_, pose, config)
        return [self.ik_error, *(self.ik_joints or self.joints)]

    def GetInverseKinRef(self, type_, pose, ref):
        self._record("GetInverseKinRef", type_, pose, ref)
        return [self.ik_error, *(self.ik_joints or ref)]

    def Mode(self, mode):
        self._record("Mode", mode)
        self.mode = mode
        return 0

    def RobotEnable(self, on):
        self._record("RobotEnable", on)
        return 0

    def SetSpeed(self, percent):
        self._record("SetSpeed", percent)
        return 0

    def MoveL(self, params):
        self._record("MoveL", params)
        time.sleep(self.move_seconds)
        self.pose = list(params[6:12])
        self.joints = list(params[0:6])
        return self.move_result

    def _frame(self) -> bytes:
        f = bytearray(FRAME_LEN)
        struct.pack_into("<HBH", f, 0, 0x5A5A, 0, FRAME_LEN - 7)
        f[fr5_io.OFF_ROBOT_STATE] = 1
        struct.pack_into("<i", f, fr5_io.OFF_MAIN_CODE, self.main_code)
        f[fr5_io.OFF_ROBOT_MODE] = self.mode
        struct.pack_into("<6d", f, fr5_io.OFF_JT_CUR_POS, *self.joints)
        struct.pack_into("<6d", f, fr5_io.OFF_TL_CUR_POS, *self.pose)
        f[fr5_io.OFF_EMERGENCY_STOP] = self.estop
        struct.pack_into("<i", f, fr5_io.OFF_MOTION_DONE, 1)
        return bytes(f)

    def _serve_rt(self):
        while self._alive:
            try:
                conn, _ = self.rt.accept()
            except OSError:
                return
            try:
                while self._alive:
                    conn.sendall(self._frame())
                    time.sleep(0.008)
            except OSError:
                pass

    def _serve_stop(self):
        while self._alive:
            try:
                conn, _ = self.stop_sock.accept()
            except OSError:
                return
            self.stop_packets.append(conn.recv(256))
            conn.close()

    @property
    def writes(self) -> list[tuple]:
        return [c for c in self.calls if c[0] in ("Mode", "RobotEnable", "SetSpeed", "MoveL")]

    def close(self):
        self._alive = False
        self.rpc.shutdown()
        self.rpc.server_close()
        self.rt.close()
        self.stop_sock.close()
