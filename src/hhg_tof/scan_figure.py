"""Figure of a finished scan: the fused cloud with where each view was taken from."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import session as session_io
from .geometry import pose_to_T
from .pointcloud_io import read_ply
from .processing import mount_T

# Fixed categorical order, validated for colour-vision deficiency on the light surface.
RING_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
CLOUD = "#8a8985"
PATH = "#c9c8c3"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def sensor_stations(session: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measured sensor positions, viewing axes, and elevation seen from the scan centre."""
    meta = session_io.read_meta(session)
    cfg = meta["config"]
    mount = mount_T(cfg)
    center = np.asarray(cfg["scan"]["center_mm"], float)
    poses = [pose_to_T(v.tcp_pose) @ mount for v in session_io.read_views(session)]
    origins = np.array([T[:3, 3] for T in poses])
    axes = np.array([T[:3, 2] for T in poses])
    offsets = origins - center
    elevation = np.degrees(np.arcsin(offsets[:, 2] / np.linalg.norm(offsets, axis=1)))
    return origins, axes, elevation


def _rings(elevation: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Group views by rounded elevation, highest first."""
    rounded = np.round(elevation / 5) * 5
    out = []
    for value in sorted(set(rounded.tolist()), reverse=True):
        mask = rounded == value
        name = "Top view" if value >= 85 else f"{value:.0f}° ring"
        out.append((f"{name} ({int(mask.sum())})", mask))
    return out


def draw(session: str | Path, out: str | Path, ray_mm: float = 90.0, points: np.ndarray | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    session = Path(session)
    if points is None:
        cloud = read_ply(session / "cloud.ply")
        points = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1)
    origins, axes, elevation = sensor_stations(session)
    center = np.asarray(session_io.read_meta(session)["config"]["scan"]["center_mm"], float)
    rings = _rings(elevation)

    fig = plt.figure(figsize=(14, 7), facecolor=SURFACE)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d", facecolor=SURFACE)
    plan = fig.add_subplot(1, 2, 2, facecolor=SURFACE)

    ax3.scatter(*points.T, s=1.5, c=CLOUD, depthshade=False, linewidths=0)
    ax3.plot(*origins.T, color=PATH, lw=1)
    plan.scatter(points[:, 0], points[:, 1], s=2, c=CLOUD, linewidths=0, label=f"Point cloud ({len(points)})")
    plan.plot(origins[:, 0], origins[:, 1], color=PATH, lw=1, zorder=1)
    for (label, mask), color in zip(rings, RING_COLORS):
        o, a = origins[mask], axes[mask]
        ax3.scatter(*o.T, s=44, c=color, edgecolors=SURFACE, linewidths=1, depthshade=False)
        ax3.quiver(*o.T, *(a * ray_mm).T, color=color, lw=1.2, arrow_length_ratio=0.15)
        plan.scatter(o[:, 0], o[:, 1], s=64, c=color, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
        plan.quiver(o[:, 0], o[:, 1], a[:, 0], a[:, 1], color=color, angles="xy", scale_units="xy",
                    scale=1 / ray_mm, width=0.0028, zorder=2)

    everything = np.vstack([points, origins])
    ax3.set_box_aspect(np.ptp(everything, axis=0))
    ax3.view_init(elev=24, azim=-58)
    ax3.set(xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
    ax3.set_title("Scan positions and point cloud", color=INK, loc="left", fontsize=12)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"]["color"] = "#e6e5e1"
        axis.label.set_color(INK_SOFT)
    ax3.tick_params(colors=INK_SOFT, labelsize=8)

    plan.annotate("to robot base", xy=(0.03, 0.04), xycoords="axes fraction", color=INK_SOFT, fontsize=9)
    direction = -center[:2] / np.linalg.norm(center[:2])
    plan.annotate("", xy=(0.03 + 0.09 * direction[0] + 0.09, 0.09 + 0.09 * direction[1] + 0.05),
                  xytext=(0.12, 0.14), xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=1))
    plan.set_aspect("equal", adjustable="datalim")
    plan.set_xlabel("x [mm]", color=INK_SOFT)
    plan.set_ylabel("y [mm]", color=INK_SOFT)
    plan.set_title("Seen from above", color=INK, loc="left", fontsize=12)
    plan.tick_params(colors=INK_SOFT, labelsize=8)
    plan.grid(color="#eeede9", lw=0.6)
    plan.set_axisbelow(True)
    for spine in plan.spines.values():
        spine.set_color("#d9d8d3")
    plan.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SOFT, markerscale=1)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def draw_views(session: str | Path, out: str | Path) -> None:
    """Front (x-z), side (y-z) and top (x-y) projections, all at one mm scale.

    Colour is height, in one hue, so the same point reads the same in every panel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cloud = read_ply(Path(session) / "cloud.ply")
    points = np.stack([cloud["x"], cloud["y"], cloud["z"]], axis=1)
    ramp = LinearSegmentedColormap.from_list("height", ["#b7d3f6", "#5598e7", "#256abf", "#0d366b"])
    panels = [("Front (x-z, looking along +y)", 0, 2, "x [mm]", "z [mm]"),
              ("Side (y-z, looking along -x)", 1, 2, "y [mm]", "z [mm]"),
              ("Top (x-y, looking down)", 0, 1, "x [mm]", "y [mm]")]
    half = float(np.ptp(points, axis=0).max()) / 2 * 1.08
    middle = (points.max(axis=0) + points.min(axis=0)) / 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE, layout="constrained")
    for ax, (title, u, v, xlabel, ylabel) in zip(axes, panels):
        art = ax.scatter(points[:, u], points[:, v], c=points[:, 2], cmap=ramp, s=3, linewidths=0)
        ax.set_xlim(middle[u] - half, middle[u] + half)
        ax.set_ylim(middle[v] - half, middle[v] + half)
        ax.set_aspect("equal")
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=INK, loc="left", fontsize=12)
        ax.set_xlabel(xlabel, color=INK_SOFT)
        ax.set_ylabel(ylabel, color=INK_SOFT)
        ax.tick_params(colors=INK_SOFT, labelsize=8)
        ax.grid(color="#eeede9", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#d9d8d3")
    bar = fig.colorbar(art, ax=axes, shrink=0.75, pad=0.015)
    bar.set_label("z [mm]", color=INK_SOFT)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
