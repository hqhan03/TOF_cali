"""ams TMF8829 EVM source over ZeroMQ (Raspberry Pi USB sensor bridge).

Written from TMF8829_zeromq_protocol.md and checked against the vendor parser in
tests/test_sensor_sources.py. Confirmed live on 2026-11-05: control, parsing and
48x32 assembly, at 5.3 fps. The PC side needs a link-local address to reach
169.254.0.2, and the bridge's USB storage interface ignored (see README).

Control: REQ on 5557. Data: SUB on 5558. Unlike the Nucleo stream, this one
carries SNR, which is a much better quality gate than temporal spread alone.

Two things the protocol document does not say, both from the vendor server:
- Sub-frame 1 carries frame number N + 1, not N.
- Set Preconfiguration leaves the server's fp mode stale; only Get Configuration
  refreshes it. Measuring without that makes the server expect one result per set
  and silently drop every sub-frame 1 (seen live: numbers step by 2, sub-frame 0 only).
"""

from __future__ import annotations

import struct
import time

import numpy as np

from ..sensor_model import FP_MODE_GRID, TICK_MM
from .base import Frame

CONTAINER_LEN = 84
CONTAINER_MAGIC = 0xFE5E1234
PRE_HEADER_LEN = 5
FRAME_HEADER_LEN = 16
FID_RESULTS = 0x10
FID_REF_SPAD = 0x30
REF_SPAD_FRAME_LEN = 16 + 32 + 12  # header + 8 x uint32 + footer

REQ_IDENTIFY, REQ_LEAVE, REQ_MEASURE, REQ_STOP = 0x01, 0x03, 0x10, 0x11
REQ_GET_CONFIG, REQ_PRECONFIG = 0x20, 0x24
ERR_OK, ERR_PASSIVE = 0x00, 0x01

PRECONFIG = {
    "8x8": 64, "8x8_long_range": 65, "8x8_high_accuracy": 66,
    "16x16": 67, "16x16_high_accuracy": 68,
    "32x32": 69, "32x32_high_accuracy": 70,
    "48x32": 71, "48x32_high_accuracy": 72,
}


def parse_result_frame(frame: bytes) -> dict:
    """Decode one pre-header + result frame into peak-0 distance and SNR."""
    head = frame[PRE_HEADER_LEN:]
    fid, layout = head[0], head[1]
    if fid & 0xF0 != FID_RESULTS:
        raise ValueError("not a result frame")
    fp_mode = fid & 0x0F
    n_peaks = layout & 0x07
    has_signal, has_noise, has_xtalk = bool(layout & 0x08), bool(layout & 0x10), bool(layout & 0x20)
    peak_size = 3 + 2 * has_signal
    pixel_size = 2 * has_noise + 2 * has_xtalk + n_peaks * peak_size
    rows, cols = FP_MODE_GRID[fp_mode]
    split = fp_mode >= 3
    rows = rows // 2 if split else rows
    data = np.frombuffer(head, np.uint8, rows * cols * pixel_size, FRAME_HEADER_LEN)
    data = data.reshape(rows, cols, pixel_size)
    first = 2 * has_noise + 2 * has_xtalk
    if n_peaks == 0:
        ticks = np.zeros((rows, cols), np.uint16)
        snr = np.zeros((rows, cols), np.uint8)
    else:
        ticks = data[..., first].astype(np.uint16) | (data[..., first + 1].astype(np.uint16) << 8)
        snr = data[..., first + 2]
    return {
        "fp_mode": fp_mode,
        "sub_frame": (layout >> 6) & 1,
        "split": split,
        "frame_number": struct.unpack_from("<I", head, 4)[0],
        "temperature_c": float(np.frombuffer(head, np.int8, 1, 10)[0]),
        "distance_mm": ticks.astype(np.float32) * TICK_MM,
        "snr": snr.copy(),
    }


def split_container(message: bytes) -> list[bytes]:
    """Return the result frames (with pre-headers) of one published message."""
    (magic,) = struct.unpack_from("<I", message, 0)
    if magic != CONTAINER_MAGIC:
        raise ValueError(f"bad container magic 0x{magic:08X}")
    data = message[CONTAINER_LEN:]
    frames, i = [], 0
    while i + PRE_HEADER_LEN + 4 <= len(data):
        fid = data[i + PRE_HEADER_LEN]
        if fid & 0xF0 == FID_REF_SPAD:
            i += PRE_HEADER_LEN + REF_SPAD_FRAME_LEN  # fixed size, payload field unused
            continue
        (payload,) = struct.unpack_from("<H", data, i + PRE_HEADER_LEN + 2)
        size = PRE_HEADER_LEN + 4 + payload
        if fid & 0xF0 == FID_RESULTS:
            frames.append(data[i:i + size])
        i += size
    return frames


def merge_sub_frames(parts: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Interleave split 32x32/48x32 halves: sub-frame 0 even rows, 1 odd rows."""
    if not parts[0]["split"]:
        return parts[0]["distance_mm"], parts[0]["snr"]
    even = next(p for p in parts if p["sub_frame"] == 0)
    odd = next(p for p in parts if p["sub_frame"] == 1)
    rows, cols = even["distance_mm"].shape
    distance = np.empty((rows * 2, cols), np.float32)
    snr = np.empty((rows * 2, cols), np.uint8)
    distance[0::2], distance[1::2] = even["distance_mm"], odd["distance_mm"]
    snr[0::2], snr[1::2] = even["snr"], odd["snr"]
    return distance, snr


class FrameAssembler:
    """Published messages -> whole frames, pairing sub-frame N with N + 1."""

    def __init__(self):
        self.half: dict | None = None  # sub-frame 0 waiting for its sub-frame 1

    def feed(self, message: bytes) -> list[Frame]:
        frames = []
        for raw in split_container(message):
            part = parse_result_frame(raw)
            if part["split"] and part["sub_frame"] == 0:
                self.half = part
                continue
            parts = [part]
            if part["split"]:
                if self.half is None or self.half["frame_number"] + 1 != part["frame_number"]:
                    continue
                parts = [self.half, part]
                self.half = None
            distance, snr = merge_sub_frames(parts)
            frames.append(Frame(distance, part["frame_number"], time.time(), snr,
                                part["temperature_c"]))
        return frames


class EvmListener:
    """Subscribe to the data port only. Having no control socket, it cannot disturb
    the sensor or whoever is driving it, and any number of listeners can run."""

    def __init__(self, host: str = "169.254.0.2"):
        self.host = host
        self.assembler = FrameAssembler()
        self._sub = None

    def open(self) -> None:
        import zmq

        self._sub = zmq.Context.instance().socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect(f"tcp://{self.host}:5558")

    def poll(self, timeout_ms: int = 0) -> list[Frame]:
        frames = []
        wait = timeout_ms
        while self._sub.poll(wait):
            frames += self.assembler.feed(self._sub.recv())
            wait = 0
        return frames

    def close(self) -> None:
        if self._sub is not None:
            self._sub.close()
            self._sub = None


class EvmSensor:
    def __init__(self, host: str = "169.254.0.2", preconfig: str = "48x32_high_accuracy",
                 frame_timeout_s: float = 5.0):
        self.host, self.preconfig, self.frame_timeout_s = host, preconfig, frame_timeout_s
        self.serial = ""
        self._client_id = 0
        self._ctx = self._cmd = self._sub = None
        self._active = False
        self._assembler = FrameAssembler()

    def _request(self, request_id: int, payload: bytes = b"") -> bytes:
        self._cmd.send(bytes([request_id]) + struct.pack("<I", self._client_id) + payload)
        if not self._cmd.poll(int(self.frame_timeout_s * 1000)):
            raise TimeoutError(f"EVM at {self.host} did not answer request 0x{request_id:02X}")
        reply = self._cmd.recv()
        if reply[0] not in (ERR_OK, ERR_PASSIVE):
            raise RuntimeError(f"EVM request 0x{request_id:02X} failed: 0x{reply[0]:02X}")
        self._client_id = struct.unpack_from("<I", reply, 1)[0]
        self._active = reply[0] == ERR_OK
        return reply[5:]

    def open(self) -> None:
        import zmq  # optional dependency

        self._ctx = zmq.Context.instance()
        self._cmd = self._ctx.socket(zmq.REQ)
        self._cmd.setsockopt(zmq.LINGER, 0)
        self._cmd.connect(f"tcp://{self.host}:5557")
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect(f"tcp://{self.host}:5558")
        info = self._request(REQ_IDENTIFY)
        self.serial = str(struct.unpack_from("<I", info, 4)[0])
        if not self._active:
            self.close()
            raise RuntimeError("another EVM client is active. Close it, or replug the EVM if "
                               "that client died: the server never expires one.")
        self._request(REQ_PRECONFIG, bytes([PRECONFIG[self.preconfig]]))
        self._request(REQ_GET_CONFIG)  # refreshes the server's fp mode, see module docstring
        self._request(REQ_MEASURE)

    def close(self) -> None:
        if self._cmd is None:
            return
        import zmq

        try:
            if self._active:
                self._request(REQ_STOP)
            self._request(REQ_LEAVE)
        except (TimeoutError, RuntimeError, zmq.ZMQError):
            pass
        self._cmd.close()
        self._sub.close()
        self._cmd = None

    def _next_frame(self) -> Frame:
        deadline = time.monotonic() + self.frame_timeout_s
        while time.monotonic() < deadline:
            if not self._sub.poll(100):
                continue
            frames = self._assembler.feed(self._sub.recv())
            if frames:
                return frames[0]
        if self._assembler.half is not None:
            raise TimeoutError(f"EVM at {self.host} publishes sub-frame 0 without sub-frame 1; "
                               "the server's fp mode is stale, see the evm.py docstring")
        raise TimeoutError(f"no measurement data from EVM at {self.host}")

    def _drain(self) -> None:
        """Drop frames queued during the move: they were taken while moving."""
        while self._sub.poll(0):
            self._sub.recv()
        self._assembler.half = None

    def read_frames(self, count: int, discard: int = 0) -> list[Frame]:
        self._drain()
        frames = [self._next_frame() for _ in range(count + discard)]
        return frames[discard:]
