"""STM32 Nucleo binary UART source (custom PCB, ~/Documents/TMF8829_Single_Test).

Host-triggered: each "A:<slot>" returns exactly one frame, so a frame is never
older than the request that asked for it. That matters here because frames must
be taken only after the arm has settled.

Protocol reimplemented from the firmware's emitBinaryFrame, as TMF8829_cali
does, so that protected project stays untouched. The stream carries distance
only: no SNR, so quality gating relies on temporal statistics.

The firmware reads host config in a ~10 s window after reset. Reset the Nucleo
just before opening this source.
"""

from __future__ import annotations

import struct
import threading
import time
import zlib
from collections import deque

import numpy as np

from ..sensor_model import TICK_MM
from .base import Frame

MAGIC = b"TMF\x29"
HEADER_FMT = "<4s10BH3I4BI"
HEADER_LEN = struct.calcsize(HEADER_FMT)
CRC_LEN = 4
MAX_PAYLOAD = 48 * 32 * 2


def parse_stream(buf: bytearray) -> list[tuple[int, Frame]]:
    """Consume complete frames from `buf` in place. Returns (slot_id, frame) pairs.

    Text lines between frames are discarded. A partial frame stays in `buf`.
    """
    out: list[tuple[int, Frame]] = []
    while True:
        start = buf.find(MAGIC)
        if start < 0:
            # Keep a possible partial magic: serial reads can split it.
            del buf[: max(0, len(buf) - (len(MAGIC) - 1))]
            return out
        del buf[:start]
        if len(buf) < HEADER_LEN:
            return out
        fields = struct.unpack_from(HEADER_FMT, buf)
        version, header_len, _board, slot, _role, _fp, rows, cols = fields[1:9]
        payload_len, frame_num = fields[11], fields[12]
        temp0 = fields[15]
        if (version != 1 or header_len != HEADER_LEN or not rows or not cols
                or payload_len != rows * cols * 2 or payload_len > MAX_PAYLOAD):
            del buf[:1]
            continue
        total = HEADER_LEN + payload_len + CRC_LEN
        if len(buf) < total:
            return out
        body = bytes(buf[: HEADER_LEN + payload_len])
        (crc,) = struct.unpack_from("<I", buf, HEADER_LEN + payload_len)
        if crc != zlib.crc32(body):
            del buf[:1]
            continue
        ticks = np.frombuffer(body, "<u2", offset=HEADER_LEN).reshape(rows, cols)
        out.append((slot, Frame(
            distance_mm=ticks.astype(np.float32) * TICK_MM,
            frame_number=frame_num,
            host_time=time.time(),
            temperature_c=float(temp0),
        )))
        del buf[:total]


class NucleoSensor:
    def __init__(self, port: str, slot: int, board_id: int = 1, baud: int = 921600,
                 frame_timeout_s: float = 2.0):
        self.port, self.slot, self.board_id, self.baud = port, slot, board_id, baud
        self.frame_timeout_s = frame_timeout_s
        self.serial = ""
        self._ser = None
        self._frames: deque[Frame] = deque(maxlen=8)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        import serial  # optional dependency

        self._ser = serial.Serial(self.port, self.baud, timeout=0.01)
        self._ser.reset_input_buffer()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        for command in (f"B,{self.board_id}", f"S:{self.slot}", "M,H", "O,B"):
            self._send(command)
            time.sleep(0.03)

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._ser:
            self._ser.close()

    def _send(self, text: str) -> None:
        self._ser.write((text + "\n").encode("ascii"))
        self._ser.flush()

    def _read_loop(self) -> None:
        buf = bytearray()
        while not self._stop.is_set():
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                buf.extend(chunk)
                for slot, frame in parse_stream(buf):
                    if slot == self.slot:
                        self._frames.append(frame)

    def _request(self) -> Frame:
        self._frames.clear()
        self._send(f"A:{self.slot}")
        deadline = time.monotonic() + self.frame_timeout_s
        while time.monotonic() < deadline:
            if self._frames:
                return self._frames.popleft()
            time.sleep(0.002)
        raise TimeoutError(f"no frame from slot {self.slot} on {self.port}")

    def read_frames(self, count: int, discard: int = 0) -> list[Frame]:
        for _ in range(discard):
            self._request()
        return [self._request() for _ in range(count)]
