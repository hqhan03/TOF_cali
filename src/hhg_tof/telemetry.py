"""One-way robot telemetry for out-of-process viewers, plus the 20004 ownership lock.

The FR5 serves port 20004 to ONE client: a new connection cuts the previous one off
(measured 2026-11-05). So a viewer must never connect there itself. During a scan
that would blind the safety monitor in `FR5Robot.move_linear`, and `RealtimeState`'s
auto-reconnect would then make the two processes evict each other forever.

Instead the one process that owns the stream republishes it on a local ZMQ PUB
socket. PUB never blocks on a slow or absent subscriber, and nothing here feeds back
into control: a viewer can start, stall or die at any moment without affecting it.
"""

from __future__ import annotations

import fcntl
import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

TELEMETRY_PORT = 5560
RATE_HZ = 30.0
BIND_RETRY_S = 1.0  # zmq releases a port asynchronously after close


class StreamOwnerLock:
    """Exclusive claim on a controller's state stream, released even if the process dies."""

    def __init__(self, ip: str):
        self.path = Path(tempfile.gettempdir()) / f"hhg_tof_fr5_{ip}.lock"
        self._file = None

    def acquire(self) -> bool:
        handle = open(self.path, "w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._file = handle
        return True

    def release(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


class TelemetryPublisher:
    """Republishes `read()` at a fixed rate. Failures here must never reach control."""

    def __init__(self, read: Callable[[], dict], port: int = TELEMETRY_PORT, rate_hz: float = RATE_HZ):
        self.read, self.port, self.period = read, port, 1.0 / rate_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _bind(self):
        import zmq

        sock = zmq.Context.instance().socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.SNDHWM, 10)
        deadline = time.monotonic() + BIND_RETRY_S
        while True:
            try:
                sock.bind(f"tcp://127.0.0.1:{self.port}")
                return sock
            except zmq.ZMQError:
                if time.monotonic() > deadline:
                    sock.close()
                    raise
                time.sleep(0.05)

    def _run(self) -> None:
        import zmq

        try:
            sock = self._bind()
        except (ImportError, zmq.ZMQError) as exc:
            log.warning("telemetry disabled: %s", exc)
            return
        try:
            while not self._stop.wait(self.period):
                try:
                    sock.send_string(json.dumps(self.read()), zmq.NOBLOCK)
                except Exception as exc:  # noqa: BLE001 - viewers are never worth a control fault
                    log.debug("telemetry skipped: %s", exc)
        finally:
            sock.close()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="telemetry")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


class TelemetrySubscriber:
    def __init__(self, port: int = TELEMETRY_PORT):
        import zmq

        self._sock = zmq.Context.instance().socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(f"tcp://127.0.0.1:{port}")

    def poll(self) -> list[dict]:
        """Every message queued since the last call, oldest first. Never waits."""
        messages = []
        while self._sock.poll(0):
            messages.append(json.loads(self._sock.recv_string()))
        return messages

    def close(self) -> None:
        self._sock.close()
