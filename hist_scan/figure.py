"""Front, side and top views of a multi-peak cloud, plus raw histograms of chosen zones."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hhg_tof.scan_figure import INK, INK_SOFT, RING_COLORS, SURFACE

from .peaks import BinScale
from .reconstruct import FAR, KIND_NAMES, NEAR, SINGLE, Cloud
from .session import read_view

COLORS = {SINGLE: "#8a8985", NEAR: RING_COLORS[1], FAR: RING_COLORS[0]}


def draw(cloud: Cloud, out: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = cloud.points
    half = float(np.ptp(p, axis=0).max()) / 2 * 1.08
    middle = (p.max(axis=0) + p.min(axis=0)) / 2
    panels = [("Front (x-z, looking along +y)", 0, 2, "x [mm]", "z [mm]"),
              ("Side (y-z, looking along -x)", 1, 2, "y [mm]", "z [mm]"),
              ("Top (x-y, looking down)", 0, 1, "x [mm]", "y [mm]")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), facecolor=SURFACE, layout="constrained")
    for ax, (title, u, v, xlabel, ylabel) in zip(axes, panels):
        for kind in (SINGLE, NEAR, FAR):
            m = cloud.kind == kind
            ax.scatter(p[m, u], p[m, v], s=4, c=COLORS[kind], linewidths=0,
                       label=f"{KIND_NAMES[kind]} ({int(m.sum())})")
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
    axes[2].legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SOFT, markerscale=2)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _ramp():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("counts", [SURFACE, "#cde2fb", "#5598e7", "#1c5cab", "#0d366b"])


def _style(ax) -> None:
    ax.tick_params(colors=INK_SOFT, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#d9d8d3")


def draw_all_histograms(session: str | Path, indices: list[int], scale: BinScale, out: str | Path) -> None:
    """Every zone of every view. One panel per view: a row per zone, sorted by strongest bin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    columns = 5
    rows = int(np.ceil(len(indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 2.7 * rows), facecolor=SURFACE,
                             layout="constrained", squeeze=False)
    span = [float(scale.to_mm(-0.5)), float(scale.to_mm(63.5))]
    norm = LogNorm(vmin=1, vmax=3000)
    art = None
    for ax, index in zip(axes.flat, indices):
        mean = read_view(session, index)["histogram"].mean(axis=0).reshape(-1, 64)
        order = np.argsort(mean[:, 14:].argmax(axis=1), kind="stable")
        art = ax.imshow(np.clip(mean[order], 1, None), aspect="auto", cmap=_ramp(), norm=norm,
                        extent=[*span, len(mean), 0], interpolation="nearest")
        ax.set_title(f"View {index}", color=INK, loc="left", fontsize=9)
        ax.set_xlim(span[0], 1300)
        _style(ax)
    for ax in axes.flat[len(indices):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("distance [mm]", color=INK_SOFT, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("zones, sorted", color=INK_SOFT, fontsize=8)
    bar = fig.colorbar(art, ax=axes, shrink=0.5, pad=0.01)
    bar.set_label("counts per measurement", color=INK_SOFT)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def draw_histogram_summary(session: str | Path, indices: list[int], scale: BinScale, out: str | Path) -> None:
    """All zones added up: one row per view, then the whole scan as one curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    per_view = np.array([read_view(session, i)["histogram"].mean(axis=0).sum(axis=(0, 1)) for i in indices])
    span = [float(scale.to_mm(-0.5)), float(scale.to_mm(63.5))]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 7), facecolor=SURFACE, layout="constrained",
                                      sharex=True, height_ratios=[3, 2])
    art = top.imshow(np.clip(per_view, 1, None), aspect="auto", cmap=_ramp(), norm=LogNorm(),
                     extent=[*span, len(indices) - 0.5, -0.5], interpolation="nearest")
    top.set_title("All zones of each view", color=INK, loc="left", fontsize=12)
    top.set_ylabel("view", color=INK_SOFT)
    bar = fig.colorbar(art, ax=top, pad=0.01)
    bar.set_label("counts per measurement", color=INK_SOFT)
    bar.ax.tick_params(colors=INK_SOFT, labelsize=8)
    bar.outline.set_visible(False)
    bottom.step(scale.to_mm(np.arange(64)), per_view.sum(axis=0), where="mid", color=RING_COLORS[0], lw=2)
    bottom.set_yscale("log")
    bottom.set_title("Whole scan", color=INK, loc="left", fontsize=12)
    bottom.set_xlabel("distance [mm]", color=INK_SOFT)
    bottom.set_ylabel("counts per measurement", color=INK_SOFT)
    bottom.grid(color="#eeede9", lw=0.6)
    bottom.set_axisbelow(True)
    bottom.set_xlim(span[0], 1300)
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        _style(ax)
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def draw_comparison(rows: list[tuple[str, list[tuple[str, object]]]], out: str | Path) -> None:
    """Per object, one row per method, one mm scale per object.

    `rows` holds (object name, [(method, default points or a Cloud), ...]).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = [("Front (x-z)", 0, 2), ("Side (y-z)", 1, 2), ("Top (x-y)", 0, 1)]
    names = "xyz"
    total = sum(len(methods) for _, methods in rows)
    fig, axes = plt.subplots(total, 3, figsize=(13.5, 4.5 * total), facecolor=SURFACE, layout="constrained")
    at = 0
    for name, methods in rows:
        clouds = [m.points if isinstance(m, Cloud) else m for _, m in methods]
        both = np.vstack(clouds)
        half = float(np.ptp(both, axis=0).max()) / 2 * 1.06
        middle = (both.max(axis=0) + both.min(axis=0)) / 2
        for (method, data), points in zip(methods, clouds):
            for ax, (title, u, v) in zip(axes[at], views):
                if isinstance(data, Cloud):
                    for kind in (SINGLE, NEAR, FAR):
                        m = data.kind == kind
                        ax.scatter(points[m, u], points[m, v], s=3, c=COLORS[kind], linewidths=0,
                                   label=f"{KIND_NAMES[kind]} ({int(m.sum())})")
                else:
                    ax.scatter(points[:, u], points[:, v], s=3, c=COLORS[SINGLE], linewidths=0)
                ax.set_xlim(middle[u] - half, middle[u] + half)
                ax.set_ylim(middle[v] - half, middle[v] + half)
                ax.set_aspect("equal")
                ax.set_facecolor(SURFACE)
                ax.set_title(title, color=INK_SOFT, loc="left", fontsize=10)
                ax.set_xlabel(f"{names[u]} [mm]", color=INK_SOFT, fontsize=9)
                ax.set_ylabel(f"{names[v]} [mm]", color=INK_SOFT, fontsize=9)
                ax.grid(color="#eeede9", lw=0.6)
                ax.set_axisbelow(True)
                _style(ax)
            axes[at][0].set_title(f"{name}, {method}: {len(points)} points", color=INK, loc="left", fontsize=12,
                                  fontweight="bold", pad=18)
            axes[at][0].text(0, 1.015, views[0][0], transform=axes[at][0].transAxes, color=INK_SOFT, fontsize=10)
            if isinstance(data, Cloud):
                axes[at][1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False,
                                   fontsize=9, labelcolor=INK_SOFT, markerscale=2.5)
            at += 1
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
