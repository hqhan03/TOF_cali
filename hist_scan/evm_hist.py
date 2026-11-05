"""TMF8829 EVM client that also streams the raw histogram of every zone.

Frame layout is from TMF8829_zeromq_protocol.md and Tmf8829AppCommon.getAllHistogramResults.
A 48x32 measurement is 12 histogram frames plus 2 result frames: each histogram frame
holds 4 reference histograms and a 16 x 8 block of 64-bin zones, 24 bits per bin.
"""

from __future__ import annotations

import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hhg_tof.sensor_model import TICK_MM
from hhg_tof.sensors.evm import CONTAINER_LEN, CONTAINER_MAGIC, FID_REF_SPAD, FID_RESULTS, PRE_HEADER_LEN, \
    PRECONFIG, REF_SPAD_FRAME_LEN, REQ_GET_CONFIG, REQ_IDENTIFY, REQ_LEAVE, REQ_MEASURE, REQ_PRECONFIG, REQ_STOP

FID_HISTOGRAM = 0x20
REQ_SET_CONFIG = 0x21
FRAME_HEADER_LEN = 16
ROWS, COLS, BINS = 32, 48, 64
BLOCK_ROWS, BLOCK_COLS = 16, 8
REF_PIXELS = 4
CHIP_PEAKS = 2                        # measured: the chip refuses 4 in this mode with "#Err,Config -2"
PAGE_BASE = 0x22                      # the configuration page starts at this register
PAGE_NR_PEAKS = 0x2A - PAGE_BASE
PAGE_HISTOGRAMS = 0x2B - PAGE_BASE
DEFAULT_VENDOR = Path.home() / "Desktop/TMF8829_custom_PCB_software/TMF8829_EVM_Docs/TMF8829_Driver_Python_v2025-11-14"


def split_frames(message: bytes) -> tuple[list[bytes], list[bytes]]:
    """(result frames, histogram frames) of one published message."""
    if struct.unpack_from("<I", message, 0)[0] != CONTAINER_MAGIC:
        raise ValueError("bad container magic")
    data = message[CONTAINER_LEN:]
    results, histograms, i = [], [], 0
    while i + PRE_HEADER_LEN + 4 <= len(data):
        fid = data[i + PRE_HEADER_LEN] & 0xF0
        if fid == FID_REF_SPAD:
            i += PRE_HEADER_LEN + REF_SPAD_FRAME_LEN
            continue
        payload = struct.unpack_from("<H", data, i + PRE_HEADER_LEN + 2)[0]
        size = PRE_HEADER_LEN + 4 + payload
        if fid == FID_RESULTS:
            results.append(data[i:i + size])
        elif fid == FID_HISTOGRAM:
            histograms.append(data[i:i + size])
        i += size
    return results, histograms


def decode_histogram_frame(frame: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    """(sub-frame index, reference (4, 64), zones (16, 8, 64))."""
    head = frame[PRE_HEADER_LEN:]
    layout = head[1]
    count = (REF_PIXELS + BLOCK_ROWS * BLOCK_COLS) * BINS
    raw = np.frombuffer(head, np.uint8, count * 3, FRAME_HEADER_LEN).reshape(-1, 3).astype(np.uint32)
    values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    reference = values[:REF_PIXELS * BINS].reshape(REF_PIXELS, BINS)
    return layout, reference, values[REF_PIXELS * BINS:].reshape(BLOCK_ROWS, BLOCK_COLS, BINS)


def assemble_histograms(frames: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """12 frames -> zones (32, 48, 64) and reference (12, 4, 64), in result-grid order."""
    if len(frames) != 12:
        raise ValueError(f"expected 12 histogram frames, got {len(frames)}")
    zones = np.zeros((ROWS, COLS, BINS), np.uint32)
    reference = np.zeros((12, REF_PIXELS, BINS), np.uint32)
    for frame in frames:
        layout, ref, block = decode_histogram_frame(frame)
        left = COLS // 2 if layout % 2 else 0
        column = 1 if layout in (2, 3, 8, 9) else 2 if layout in (4, 5, 10, 11) else 0
        row = 1 if layout > 5 else 0
        zones[row::2, left + column:left + COLS // 2:3] = block
        reference[layout] = ref
    return zones, reference


def decode_result_peaks(frame: bytes) -> tuple[int, int, np.ndarray, np.ndarray]:
    """(sub-frame, frame number, distance mm (16, 48, P), snr (16, 48, P))."""
    head = frame[PRE_HEADER_LEN:]
    layout = head[1]
    peaks = layout & 0x07
    has_signal, has_noise, has_xtalk = bool(layout & 0x08), bool(layout & 0x10), bool(layout & 0x20)
    peak_size = 3 + 2 * has_signal
    first = 2 * has_noise + 2 * has_xtalk
    pixel = first + peaks * peak_size
    data = np.frombuffer(head, np.uint8, 16 * COLS * pixel, FRAME_HEADER_LEN).reshape(16, COLS, pixel)
    distance = np.zeros((16, COLS, peaks), np.float32)
    snr = np.zeros((16, COLS, peaks), np.uint8)
    for p in range(peaks):
        at = first + p * peak_size
        distance[..., p] = (data[..., at].astype(np.uint16) | (data[..., at + 1].astype(np.uint16) << 8)) * TICK_MM
        snr[..., p] = data[..., at + 2]
    return (layout >> 6) & 1, struct.unpack_from("<I", head, 4)[0], distance, snr


@dataclass
class Measurement:
    histogram: np.ndarray    # (32, 48, 64) counts
    reference: np.ndarray    # (12, 4, 64) counts
    distance_mm: np.ndarray  # (32, 48, P) peaks reported by the chip, 0 = none
    snr: np.ndarray          # (32, 48, P)
    frame_number: int
    host_time: float


def decode_measurement(message: bytes) -> Measurement | None:
    results, frames = split_frames(message)
    if len(frames) != 12 or len(results) != 2:
        return None
    zones, reference = assemble_histograms(frames)
    parts = {}
    for raw in results:
        sub, number, distance, snr = decode_result_peaks(raw)
        parts[sub] = (number, distance, snr)
    if set(parts) != {0, 1}:
        return None
    distance = np.empty((ROWS, COLS, parts[0][1].shape[-1]), np.float32)
    snr = np.empty(distance.shape, np.uint8)
    distance[0::2], distance[1::2] = parts[0][1], parts[1][1]
    snr[0::2], snr[1::2] = parts[0][2], parts[1][2]
    return Measurement(zones, reference, distance, snr, parts[0][0], time.time())


class HistogramSensor:
    """Active EVM client with histogram dumping and several peaks per zone turned on."""

    def __init__(self, host: str = "169.254.0.2", preconfig: str = "48x32_high_accuracy",
                 peaks: int = CHIP_PEAKS, timeout_s: float = 8.0, vendor: Path = DEFAULT_VENDOR):
        self.host, self.preconfig, self.peaks, self.timeout_s = host, preconfig, peaks, timeout_s
        self.vendor = Path(vendor)
        self.serial = ""
        self.config: dict = {}
        self._client = 0
        self._active = False
        self._cmd = self._sub = None

    def _request(self, request: int, payload: bytes = b"") -> bytes:
        self._cmd.send(bytes([request]) + struct.pack("<I", self._client) + payload)
        if not self._cmd.poll(int(self.timeout_s * 1000)):
            raise TimeoutError(f"EVM did not answer request 0x{request:02X}")
        reply = self._cmd.recv()
        if reply[0] not in (0, 1):
            raise RuntimeError(f"EVM request 0x{request:02X} failed: 0x{reply[0]:02X}")
        self._client = struct.unpack_from("<I", reply, 1)[0]
        self._active = reply[0] == 0
        return reply[5:]

    def _converter(self):
        for path in (self.vendor, self.vendor / "tmf8829"):
            if str(path) not in sys.path:
                sys.path.append(str(path))
        from register_page_converter import RegisterPageConverter
        from tmf8829_config_page import Tmf8829_config_page

        return RegisterPageConverter, Tmf8829_config_page

    def open(self) -> None:
        import zmq

        context = zmq.Context.instance()
        self._cmd = context.socket(zmq.REQ)
        self._cmd.setsockopt(zmq.LINGER, 0)
        self._cmd.connect(f"tcp://{self.host}:5557")
        self._sub = context.socket(zmq.SUB)
        self._sub.setsockopt(zmq.LINGER, 0)
        self._sub.setsockopt(zmq.RCVHWM, 8)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.connect(f"tcp://{self.host}:5558")
        info = self._request(REQ_IDENTIFY)
        self.serial = str(struct.unpack_from("<I", info, 4)[0])
        if not self._active:
            self.close()
            raise RuntimeError("another EVM client is active")
        try:
            self._start()
        except BaseException:
            self.close()  # a client that never leaves locks the EVM until its server restarts
            raise

    def _start(self) -> None:
        self._request(REQ_PRECONFIG, bytes([PRECONFIG[self.preconfig]]))
        # Edit the page in place: the vendor dict round trip zeroes the reserved byte at 0x5E.
        page = bytearray(self._request(REQ_GET_CONFIG))
        page[PAGE_NR_PEAKS] = (page[PAGE_NR_PEAKS] & ~0x07) | self.peaks
        page[PAGE_HISTOGRAMS] |= 0x01
        self._request(REQ_SET_CONFIG, bytes(page))
        convert, regs = self._converter()
        self.config = convert.readPageToDict(self._request(REQ_GET_CONFIG), regs())
        if self.config["histograms"] != 1:
            raise RuntimeError("the sensor did not accept histogram dumping")
        self._request(REQ_MEASURE)

    def _restore_defaults(self) -> None:
        """Preconfiguration does NOT clear histogram dumping, so it has to be undone here."""
        page = bytearray(self._request(REQ_GET_CONFIG))
        page[PAGE_NR_PEAKS] = (page[PAGE_NR_PEAKS] & ~0x07) | 1
        page[PAGE_HISTOGRAMS] &= ~0x01
        self._request(REQ_SET_CONFIG, bytes(page))
        self._request(REQ_GET_CONFIG)  # the server re-reads its mode from this

    def read(self, count: int, discard: int = 1) -> list[Measurement]:
        """Measurements taken after this call, skipping `discard` to clear the pipeline."""
        while self._sub.poll(0):
            self._sub.recv()
        out: list[Measurement] = []
        deadline = time.monotonic() + self.timeout_s + count * 3.0
        seen = 0
        while len(out) < count:
            if time.monotonic() > deadline:
                raise TimeoutError(f"got {len(out)} of {count} histogram measurements")
            if not self._sub.poll(200):
                continue
            measurement = decode_measurement(self._sub.recv())
            if measurement is None:
                continue
            seen += 1
            if seen > discard:
                out.append(measurement)
        return out

    def close(self) -> None:
        if self._cmd is None:
            return
        import zmq

        try:
            if self._active:
                self._request(REQ_STOP)
                self._restore_defaults()
            self._request(REQ_LEAVE)
        except (TimeoutError, RuntimeError, zmq.ZMQError):
            pass
        self._cmd.close()
        self._sub.close()
        self._cmd = None
