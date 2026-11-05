"""Scan configuration, loaded from YAML over built-in defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "robot": {
        "backend": "fr5",              # fr5 | sim
        "ip": "192.168.58.2",
        "interface": "enp131s0",
        "expected_tool": 0,
        "expected_wobj": 0,
        "speed_percent": 10,
        "vel_percent": 20,
        "ovl_percent": 100,
        "settle_tol_mm": 0.05,
        "settle_hold_s": 0.4,
        "settle_timeout_s": 10.0,
        "max_joint_step_deg": 45.0,
        "joint_limits_deg": [[-175, 175], [-265, 85], [-162, 162], [-265, 85], [-175, 175], [-175, 175]],
        "joint_margin_deg": 5.0,
        "max_view_drift_mm": 0.2,
    },
    "sensor": {
        "source": "nucleo",            # nucleo | evm | sim
        "grid": [32, 48],              # rows, cols
        "port": "/dev/ttyACM0",
        "slot": 7,
        "board_id": 1,
        "baud": 921600,
        "evm_host": "169.254.0.2",
        "preconfig": "48x32_high_accuracy",
        "frames_per_view": 15,
        "discard_frames": 2,
        "calibration_json": None,
        "mirror_columns": False,
        "mirror_verified": False,
        "range_bias_mm": 0.0,          # subtracted from radial range when no LUT is used
    },
    "mount": {"xyz_mm": None, "rpy_deg": None, "verified": False},
    "scan": {
        "center_mm": None,
        "object_radius_mm": None,
        "clearance_mm": 40.0,
        "standoff_mm": 250.0,
        # Checked with real FR5 IK, object 450 mm from the base axis: 2 rings and
        # 45 deg roll steps complete a full orbit. Three rings down to 25 deg only
        # passed as a 270 deg arc.
        "elevations_deg": [70, 40],
        "views_per_ring": [8, 12],
        "azimuth_start_deg": 0.0,
        "azimuth_span_deg": 360.0,
        "ring_arcs_deg": None,         # per ring [start, span] or null, e.g. to skip a side a low ring cannot pass
        "add_top_view": True,
        "return_to_top": True,         # after a complete scan, retrace the orbit to the top view
        "roll_deg": 0.0,
        "roll_cycle_deg": [0],         # plane calibration needs several, see calibration.py
        "roll_candidates_deg": [0, 45, -45, 90, -90, 135, -135, 180],
        "workspace_mm": None,          # {x: [min, max], y: [...], z: [...]}, for TCP and sensor
    },
    "processing": {
        "min_valid_fraction": 0.6,
        "max_mad_mm": 8.0,
        "max_mad_rel": 0.03,
        "min_snr": 0,
        "min_range_mm": 30.0,
        "max_range_mm": 800.0,
        "edge_abs_mm": 20.0,
        "edge_rel": 0.06,
        "max_incidence_deg": 75.0,
        "roi_radius_mm": None,         # default: object_radius_mm * 1.3
        "floor_z_mm": None,            # points at or below this are dropped
        "floor_margin_mm": 5.0,
        "voxel_mm": 4.0,
        # 1 on purpose. Zones are ~7 mm apart at scan distance, wider than a voxel, so
        # two views rarely land in one voxel and 2 deletes real surface (box coverage
        # 100 % -> 67 % in simulation). Raise it only to fight ghosts seen in one view.
        "min_views_per_voxel": 1,
        "outlier_neighbors": 12,
        "outlier_std": 2.0,
        "normal_neighbors": 16,
        "icp": False,
        "icp_max_translation_mm": 5.0,
        "icp_max_rotation_deg": 2.0,
    },
    "sim": {
        "start_pose": [-300.0, 0.0, 400.0, 180.0, 0.0, 0.0],
        "object": {"type": "sphere", "radius_mm": 60.0},
        "floor_z_mm": 0.0,
        "true_mount_xyz_mm": None,     # default: same as mount
        "true_mount_rpy_deg": None,
        "true_mirror_columns": False,
        "range_bias_mm": 0.0,
        "noise_mm": 1.0,
        "noise_rel": 0.004,
        "dropout": 0.02,
        "pose_noise_mm": 0.05,
    },
}


def _merge(base: dict, override: dict, path: str = "") -> dict:
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"unknown config key: {path}{key}")
        if isinstance(base[key], dict) and isinstance(value, dict) and key not in ("object", "workspace_mm"):
            _merge(base[key], value, f"{path}{key}.")
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        _merge(cfg, overrides)
    return cfg


def require(cfg: dict, *dotted: str) -> None:
    """Fail closed: real-robot values have no defaults because they cannot be guessed."""
    missing = []
    for key in dotted:
        node: Any = cfg
        for part in key.split("."):
            node = node[part]
        if node is None:
            missing.append(key)
    if missing:
        raise ValueError("set these in the config first: " + ", ".join(missing))
