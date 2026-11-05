"""`hhg-tof live`: range image, robot pose and 3D points in one Rerun window.

A separate process that only subscribes. Frames come from the EVM's PUB port and
robot state from the telemetry port, so starting, stalling or killing this viewer
cannot affect a scan. It holds no control socket and never connects to the robot.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .geometry import T_to_pose, pose_to_T
from .live import LiveFrame, LiveProcessor, arm_polyline
from .processing import mount_T
from .sensor_model import zone_tangents
from .sensors.base import Frame
from .telemetry import TelemetrySubscriber

POLL_MS = 30
TRAIL_POINTS = 600
MAP_PERIOD_S = 1.0
STATUS_PERIOD_S = 0.5
FRUSTUM_MM = 300.0
SILENT_AFTER_S = 2.0


def range_colors(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Near = red, far = blue."""
    u = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (np.stack([1 - u, 1 - np.abs(2 * u - 1), u], axis=1) * 255).astype(np.uint8)


def frustum_strips(rows: int, cols: int, mirror: bool) -> list[np.ndarray]:
    xn, yn = zone_tangents(rows, cols, mirror)
    corners = [np.array([xn[r, c], yn[r, c], 1.0]) * FRUSTUM_MM
               for r, c in ((0, 0), (0, -1), (-1, -1), (-1, 0))]
    return [np.array([np.zeros(3), c]) for c in corners] + [np.array(corners + corners[:1])]


class HardwareSource:
    def __init__(self, cfg: dict):
        from .sensors.evm import EvmListener

        self.evm = EvmListener(cfg["sensor"]["evm_host"])
        self.evm.open()
        self.telemetry = TelemetrySubscriber()

    def poll(self) -> tuple[list[dict], list[Frame]]:
        frames = self.evm.poll(POLL_MS)
        return self.telemetry.poll(), frames

    def close(self) -> None:
        self.evm.close()
        self.telemetry.close()


class DemoSource:
    """Simulated orbit with no hardware, for trying the viewer. It has no kinematics,
    so joints are absent and the arm is not drawn."""

    MOVE_S, DWELL_S, FRAME_S = 1.2, 1.4, 0.19

    def __init__(self, cfg: dict):
        from . import planning
        from .cli import build_robot, build_sensor

        self.cfg = cfg
        self.robot = build_robot(cfg)
        self.sensor = build_sensor(cfg, self.robot)
        report = planning.preflight(planning.plan_views(cfg["scan"]), cfg, self.robot, mount_T(cfg))
        self.poses = [pose_to_T(v.tcp_pose) for v in report.views]
        self.start = time.time()
        self.next_frame = self.start

    def _pose_at(self, elapsed: float) -> list[float]:
        leg = self.MOVE_S + self.DWELL_S
        index, phase = divmod(elapsed, leg)
        a = self.poses[int(index) % len(self.poses)]
        b = self.poses[(int(index) + 1) % len(self.poses)]
        u = min(phase / self.MOVE_S, 1.0)
        T = np.eye(4)
        T[:3, 3] = (1 - u) * a[:3, 3] + u * b[:3, 3]
        T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(u).as_matrix()
        return T_to_pose(T)

    def poll(self) -> tuple[list[dict], list[Frame]]:
        time.sleep(POLL_MS / 1000)
        now = time.time()
        pose = self._pose_at(now - self.start)
        self.robot.pose = pose
        message = {"t": now, "tcp_pose": pose, "joints": None, "tool": self.cfg["robot"]["expected_tool"],
                   "wobj": 0, "mode": 0, "fault": ""}
        frames = []
        if now >= self.next_frame:
            self.next_frame = now + self.FRAME_S
            frames = self.sensor.read_frames(1)
            frames[0].host_time = now
        return [message], frames

    def close(self) -> None:
        pass


class Renderer:
    def __init__(self, cfg: dict, save: str | None, window: bool):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr, self.cfg = rr, cfg
        self.max_range = cfg["processing"]["max_range_mm"]
        self.trail: deque = deque(maxlen=TRAIL_POINTS)
        self.last_map = self.last_status = 0.0
        self.last_robot = self.last_frame = None
        self.frame_times: deque = deque(maxlen=20)
        self.robot_msg: dict | None = None
        self.placement = "waiting for frames"

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/world", name="Base frame"),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="/tof/range", name="Range [mm]"),
                        rrb.Spatial2DView(origin="/tof/snr", name="SNR"),
                    ),
                    rrb.Spatial3DView(origin="/sensor_frame", name="Sensor frame (no mount needed)"),
                    rrb.Horizontal(
                        rrb.TimeSeriesView(origin="/plots/joints", name="Joints [deg]"),
                        rrb.TextDocumentView(origin="/status", name="Status"),
                    ),
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
        rr.init("hhg_tof_live")
        sinks = []
        if window:
            rr.spawn(connect=False, memory_limit="2GB")
            sinks.append(rr.GrpcSink())
        if save:
            sinks.append(rr.FileSink(save))
        rr.set_sinks(*sinks, default_blueprint=blueprint)
        self._static()

    def _static(self) -> None:
        rr, sc = self.rr, self.cfg["sensor"]
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("sensor_frame", rr.ViewCoordinates.RDF, static=True)
        axes = rr.Arrows3D(vectors=np.eye(3) * 80, colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                           radii=2.0)
        rr.log("world/base_axes", axes, static=True)
        rr.log("world/tcp/axes", axes, static=True)
        frustum = rr.LineStrips3D(frustum_strips(*sc["grid"], sc["mirror_columns"]),
                                  colors=[255, 200, 0], radii=0.8)
        rr.log("sensor_frame/frustum", frustum, static=True)

    def _time(self, t: float) -> None:
        self.rr.set_time("wall", timestamp=t)

    def robot(self, m: dict) -> None:
        rr = self.rr
        self._time(m["t"])
        self.robot_msg, self.last_robot = m, time.time()
        T = pose_to_T(m["tcp_pose"])
        rr.log("world/tcp", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        self.trail.append(T[:3, 3].copy())
        if len(self.trail) > 1:
            rr.log("world/trail", rr.LineStrips3D([np.array(self.trail)], colors=[120, 120, 120], radii=0.6))
        if m.get("joints"):
            rr.log("world/arm", rr.LineStrips3D([arm_polyline(m["joints"])], colors=[200, 200, 220], radii=6.0))
            for i, angle in enumerate(m["joints"], 1):
                rr.log(f"plots/joints/J{i}", rr.Scalars(angle))

    def frame(self, r: LiveFrame, map_points: np.ndarray) -> None:
        rr = self.rr
        f = r.frame
        self._time(f.host_time)
        self.last_frame = time.time()
        self.frame_times.append(self.last_frame)
        self.placement = r.placement
        rr.log("tof/range", rr.DepthImage(f.distance_mm, meter=1.0, colormap="turbo",
                                          depth_range=(0.0, self.max_range)))
        if f.snr is not None:
            rr.log("tof/snr", rr.Image(f.snr, color_model="L"))
        rr.log("plots/valid_percent", rr.Scalars(r.valid_percent))
        if f.temperature_c is not None:
            rr.log("plots/temperature_c", rr.Scalars(f.temperature_c))
        colors = range_colors(r.sensor_points[:, 2], 0.0, self.max_range) if len(r.sensor_points) else None
        rr.log("sensor_frame/points", rr.Points3D(r.sensor_points, colors=colors, radii=1.5))
        if r.base_points is None:
            rr.log("world/sensor", rr.Clear(recursive=True))
            rr.log("world/frame_points", rr.Clear(recursive=False))
        else:
            T = r.T_base_sensor
            rr.log("world/sensor", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            sc = self.cfg["sensor"]
            rr.log("world/sensor/frustum", rr.LineStrips3D(frustum_strips(*sc["grid"], sc["mirror_columns"]),
                                                           colors=[255, 200, 0], radii=0.8))
            color = [80, 255, 120] if r.still else [255, 120, 60]
            rr.log("world/frame_points", rr.Points3D(r.base_points, colors=color, radii=1.5))
        if time.time() - self.last_map >= MAP_PERIOD_S and len(map_points):
            self.last_map = time.time()
            rr.log("world/map", rr.Points3D(map_points, colors=[170, 190, 255], radii=1.2))

    def status(self, proc: LiveProcessor) -> None:
        now = time.time()
        if now - self.last_status < STATUS_PERIOD_S:
            return
        self.last_status = now
        self._time(now)
        m = self.robot_msg

        def age(t):
            if t is None:
                return "**no data**"
            return f"**stale {now - t:.0f} s**" if now - t > SILENT_AFTER_S else "live"

        times = self.frame_times
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else 0.0
        mount = "set" if proc.T_mount is not None else "**none: base-frame points are off**"
        if proc.T_mount is not None and not self.cfg["mount"]["verified"]:
            mount = "set, **unverified**"
        lines = [
            f"- Robot: {age(self.last_robot)}",
            f"- Sensor: {age(self.last_frame)}, {fps:.1f} fps",
            f"- Placement: {self.placement}",
            f"- Mount: {mount}",
            f"- Mirror: {'verified' if self.cfg['sensor']['mirror_verified'] else '**unverified**'}",
            f"- Map: {len(proc.map.points)} points",
        ]
        if m:
            lines += [f"- Tool {m.get('tool')}, wobj {m.get('wobj')}",
                      f"- TCP: {[round(v, 1) for v in m['tcp_pose']]}"]
            if m.get("fault"):
                lines.append(f"- **Fault: {m['fault']}**")
        self.rr.log("status", self.rr.TextDocument("\n".join(lines), media_type=self.rr.MediaType.MARKDOWN))


def run(cfg: dict, correction, demo: bool, save: str | None, window: bool, duration_s: float | None) -> None:
    proc = LiveProcessor(cfg, correction)
    renderer = Renderer(cfg, save, window)
    source = DemoSource(cfg) if demo else HardwareSource(cfg)
    deadline = None if duration_s is None else time.monotonic() + duration_s
    try:
        while deadline is None or time.monotonic() < deadline:
            messages, frames = source.poll()
            for message in messages:
                proc.add_robot(message)
                renderer.robot(message)
            for frame in frames:
                result = proc.add_frame(frame)
                renderer.frame(result, proc.map.points)
            renderer.status(proc)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        renderer.rr.disconnect()
