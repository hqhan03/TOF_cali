"""Frame parsers, including a cross-check against ams' own Python driver."""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from hhg_tof.sensors import evm, nucleo

VENDOR = Path.home() / "Desktop/TMF8829_cali/TMF8829_EVM_Docs/TMF8829_Driver_Python_v2025-11-14"


def binary_frame(ticks: np.ndarray, slot=7, frame_num=3) -> bytes:
    rows, cols = ticks.shape
    payload = ticks.astype("<u2").tobytes()
    header = struct.pack(nucleo.HEADER_FMT, nucleo.MAGIC, 1, nucleo.HEADER_LEN, 1, slot, 0, 5,
                         rows, cols, 0, 0, len(payload), frame_num, 10, 20, 30, 31, 32, 0, 1)
    body = header + payload
    return body + struct.pack("<I", zlib.crc32(body))


def test_nucleo_converts_ticks_to_mm():
    ticks = np.full((32, 48), 800)
    ((slot, frame),) = nucleo.parse_stream(bytearray(binary_frame(ticks)))
    assert slot == 7 and frame.frame_number == 3
    assert np.all(frame.distance_mm == 200.0)


def test_nucleo_survives_byte_at_a_time_delivery():
    data = b"text line\n" + binary_frame(np.full((16, 16), 400))
    buf, frames = bytearray(), []
    for i in range(len(data)):
        buf.append(data[i])
        frames += nucleo.parse_stream(buf)
    assert len(frames) == 1


def test_nucleo_drops_bad_crc_and_recovers():
    bad = bytearray(binary_frame(np.full((8, 8), 100)))
    bad[-1] ^= 0xFF
    good = binary_frame(np.full((8, 8), 200))
    frames = nucleo.parse_stream(bytearray(bytes(bad) + good))
    assert len(frames) == 1
    assert frames[0][1].distance_mm[0, 0] == 50.0


def result_frame(fp_mode, layout, rows, cols, ticks, snr, frame_no=9):
    """A result frame as the chip sends it, with a 5-byte pre-header."""
    pixels = bytearray()
    for t, s in zip(ticks.ravel(), snr.ravel()):
        if layout & 0x10:
            pixels += b"\x11\x00"  # noise, must be skipped
        pixels += struct.pack("<HB", int(t), int(s))
        if layout & 0x08:
            pixels += b"\x22\x00"  # signal
        peak_size = 5 if layout & 0x08 else 3
        pixels += bytes(peak_size) * ((layout & 7) - 1)
    payload_len = 12 + len(pixels) + 12
    head = struct.pack("<BBHI3bBHH", 0x10 | fp_mode, layout, payload_len, frame_no, 25, 26, 27, 0, 0, 0)
    footer = struct.pack("<IIBBH", 1, 2, 1, 0, 0xE0F0)
    return b"\x00" * 5 + head + bytes(pixels) + footer


def container(frames) -> bytes:
    body = b"".join(frames)
    return (struct.pack("<IB3xIBxHI", evm.CONTAINER_MAGIC, 2, 76 + len(body), 3, 0, 123)
            + bytes(64) + body)


def split_pair(rng, layout=0x01):
    even = rng.integers(1, 4000, (16, 48))
    odd = rng.integers(1, 4000, (16, 48))
    snr = rng.integers(0, 255, (16, 48))
    return (even, odd, snr, [result_frame(5, layout, 16, 48, even, snr),
                             result_frame(5, layout | 0x40, 16, 48, odd, snr)])


def test_evm_merges_sub_frames_to_even_and_odd_rows(rng):
    even, odd, snr, frames = split_pair(rng)
    parts = [evm.parse_result_frame(f) for f in evm.split_container(container(frames))]
    distance, out_snr = evm.merge_sub_frames(parts)
    assert distance.shape == (32, 48)
    assert np.allclose(distance[0::2], even * 0.25)
    assert np.allclose(distance[1::2], odd * 0.25)
    assert np.array_equal(out_snr[0::2], snr)


def test_evm_parses_noise_and_signal_fields(rng):
    ticks = rng.integers(1, 4000, (16, 16))
    snr = rng.integers(0, 255, (16, 16))
    part = evm.parse_result_frame(result_frame(2, 0x1A, 16, 16, ticks, snr))  # noise, signal, 2 peaks
    assert np.allclose(part["distance_mm"], ticks * 0.25)
    assert np.array_equal(part["snr"], snr)


def test_evm_rejects_foreign_message():
    with pytest.raises(ValueError):
        evm.split_container(b"\x00" * 100)


def test_evm_matches_vendor_parser(rng):
    if not VENDOR.exists():
        pytest.skip("vendor driver not on this machine")
    sys.path.insert(0, str(VENDOR))
    sys.path.insert(0, str(VENDOR / "tmf8829"))
    try:
        common = pytest.importorskip("tmf8829.tmf8829_application_common")
    except Exception as exc:  # vendor package pulls in its own optional deps
        pytest.skip(f"vendor import failed: {exc}")
    even, odd, snr, frames = split_pair(rng, layout=0x09)  # 1 peak + signal
    vendor = common.Tmf8829AppCommon.getFullPixelResult(
        [bytearray(f) for f in frames], toMM=True, deleteNone=False)
    distance, _ = evm.merge_sub_frames([evm.parse_result_frame(f) for f in frames])
    expected = np.array([[px["peaks"][0]["distance"] for px in row] for row in vendor])
    assert np.allclose(distance, expected)


class FakeSub:
    """SUB socket replaying published messages, then going quiet."""

    def __init__(self, messages):
        self.messages = list(messages)

    def poll(self, timeout_ms=0):
        return bool(self.messages)

    def recv(self):
        return self.messages.pop(0)


def live_pair(rng, frame_no):
    """Sub-frame 1 carries the next frame number, as the sensor sends it."""
    ticks = rng.integers(1, 4000, (2, 16, 48))
    snr = rng.integers(0, 255, (16, 48))
    return ticks, [result_frame(5, 0x01, 16, 48, ticks[0], snr, frame_no),
                   result_frame(5, 0x41, 16, 48, ticks[1], snr, frame_no + 1)]


def test_evm_assembles_sub_frames_with_consecutive_numbers(rng):
    ticks, frames = live_pair(rng, 100)
    sensor = evm.EvmSensor(frame_timeout_s=0.05)
    sensor._sub = FakeSub([container(frames)])
    frame = sensor._next_frame()
    assert np.allclose(frame.distance_mm[0::2], ticks[0] * 0.25)
    assert np.allclose(frame.distance_mm[1::2], ticks[1] * 0.25)


def test_evm_does_not_pair_sub_frames_of_different_measurements(rng):
    _, old = live_pair(rng, 100)
    ticks, new = live_pair(rng, 102)
    sensor = evm.EvmSensor(frame_timeout_s=0.05)
    sensor._sub = FakeSub([container([old[0]]), container(new)])  # 101 was lost
    frame = sensor._next_frame()
    assert np.allclose(frame.distance_mm[0::2], ticks[0] * 0.25)


def test_evm_names_the_cause_when_only_one_half_arrives(rng):
    """Seen live: the server publishes sub-frame 0 only if it has the wrong fp mode."""
    halves = [live_pair(rng, n)[1][0] for n in (100, 102, 104)]
    sensor = evm.EvmSensor(frame_timeout_s=0.05)
    sensor._sub = FakeSub([container([f]) for f in halves])
    with pytest.raises(TimeoutError, match="sub-frame"):
        sensor._next_frame()


class FakeReq:
    """REQ socket answering every request with one error code."""

    def __init__(self, code):
        self.code, self.sent = code, []

    def setsockopt(self, *args):
        pass

    def connect(self, *args):
        pass

    def close(self):
        pass

    def send(self, message):
        self.sent.append(message[0])

    def poll(self, timeout_ms):
        return True

    def recv(self):
        return bytes([self.code]) + struct.pack("<I", 7) + bytes(76)


class FakeContext:
    def __init__(self, req):
        self.sockets = [req, FakeSub([])]
        self.sockets[1].setsockopt = self.sockets[1].connect = lambda *a: None
        self.sockets[1].close = lambda: None

    def socket(self, kind):
        return self.sockets.pop(0)


def open_evm(monkeypatch, code):
    zmq = pytest.importorskip("zmq")
    req = FakeReq(code)
    monkeypatch.setattr(zmq.Context, "instance", lambda: FakeContext(req))
    return evm.EvmSensor(), req


def test_evm_reads_config_between_preconfig_and_measure(monkeypatch):
    """Preconfig alone leaves the server's fp mode stale, which loses sub-frame 1."""
    sensor, req = open_evm(monkeypatch, evm.ERR_OK)
    sensor.open()
    assert req.sent == [evm.REQ_IDENTIFY, evm.REQ_PRECONFIG, evm.REQ_GET_CONFIG, evm.REQ_MEASURE]


def test_evm_passive_client_leaves_without_touching_the_measurement(monkeypatch):
    sensor, req = open_evm(monkeypatch, evm.ERR_PASSIVE)
    with pytest.raises(RuntimeError, match="another EVM client"):
        sensor.open()
    assert req.sent == [evm.REQ_IDENTIFY, evm.REQ_LEAVE]
