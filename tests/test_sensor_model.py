import json
import math
from pathlib import Path

import numpy as np
import pytest

from hhg_tof.sensor_model import LutCorrection, distance_to_xyz, zone_tangents

CAMPAIGNS = Path.home() / "Desktop/TMF8829_cali/calibration_campaigns"
SLOT7 = (CAMPAIGNS / "2026-06-14_7sensor_48x32_ha/per_sensor/serial_5013663_slot_7/calibration"
         / "tmf8829_serial_5013663_slot_7_48x32_ha_calibration.json")
MISLABELED_16X16 = (CAMPAIGNS / "2026-06-19_7sensor_16x16_100iter/per_sensor/serial_unknown_slot_1_slot_1"
                    / "calibration/tmf8829_serial_unknown_slot_1_48x32_ha_calibration.json")


def vendor_z_correction(px, py, X, Y):
    """Tmf8829AppCommon.zCorrection, transcribed."""
    x = (px - X / 2 + 0.5) / (X * 3.0 / 4.0)
    y = (py - Y / 2 + 0.5) / Y
    return math.sqrt(1 + x * x + y * y), x, y


@pytest.mark.parametrize("rows,cols", [(8, 8), (16, 16), (32, 32), (32, 48)])
def test_rays_match_vendor(rows, cols):
    xn, yn = zone_tangents(rows, cols)
    for r, c in [(0, 0), (rows - 1, cols - 1), (rows // 2, 1)]:
        factor, x, y = vendor_z_correction(c, r, cols, rows)
        assert np.isclose(xn[r, c], x) and np.isclose(yn[r, c], y)
        assert np.isclose(np.sqrt(1 + xn[r, c] ** 2 + yn[r, c] ** 2), factor)


def test_field_of_view_matches_datasheet():
    xn, yn = zone_tangents(32, 48)
    # Outer edge of the outer zone, in degrees; datasheet: 67.9 x 52.8.
    edge_x = abs(xn[0, 0]) + (xn[0, 1] - xn[0, 0]) / 2
    edge_y = abs(yn[0, 0]) + (yn[1, 0] - yn[0, 0]) / 2
    assert abs(2 * math.degrees(math.atan(edge_x)) - 67.9) < 1.0
    assert abs(2 * math.degrees(math.atan(edge_y)) - 52.8) < 1.0


def test_flat_wall_radial_becomes_constant_depth():
    xn, yn = zone_tangents(32, 48)
    radial = 300.0 * np.sqrt(1 + xn**2 + yn**2)
    assert np.allclose(distance_to_xyz(radial, "radial", xn, yn)[..., 2], 300.0)


def test_plane_z_is_not_cosine_corrected_again():
    xn, yn = zone_tangents(32, 48)
    assert np.allclose(distance_to_xyz(np.full((32, 48), 300.0), "plane_z", xn, yn)[..., 2], 300.0)


def test_mirror_flips_only_x():
    xn, yn = zone_tangents(32, 48)
    mx, my = zone_tangents(32, 48, mirror_columns=True)
    assert np.allclose(mx, -xn) and np.allclose(my, yn)


def lut_file(tmp_path, rows=2, cols=2, serial="111"):
    pairs = [{"raw_mm": 110.0, "true_mm": 100.0}, {"raw_mm": 220.0, "true_mm": 200.0}]
    path = tmp_path / "lut.json"
    path.write_text(json.dumps({
        "actual_grid": {"rows": rows, "cols": cols}, "device_serial": serial,
        "lut": [[pairs] * cols] * rows,
    }))
    return path


def test_lut_interpolates_and_holes_outside_range(tmp_path):
    lut = LutCorrection(lut_file(tmp_path), 2, 2)
    out = lut.apply(np.array([[165.0, 110.0], [50.0, 300.0]]))
    assert np.isclose(out[0, 0], 150.0) and np.isclose(out[0, 1], 100.0)
    assert np.isnan(out[1, 0]) and np.isnan(out[1, 1])


def test_lut_refuses_wrong_grid_and_wrong_sensor(tmp_path):
    with pytest.raises(ValueError, match="grid"):
        LutCorrection(lut_file(tmp_path, 16, 16), 32, 48)
    with pytest.raises(ValueError, match="sensor"):
        LutCorrection(lut_file(tmp_path), 2, 2, expected_serial="222")


@pytest.mark.skipif(not MISLABELED_16X16.exists(), reason="calibration data not on this machine")
def test_real_mislabeled_file_is_caught_by_its_contents():
    assert "48x32" in MISLABELED_16X16.name
    with pytest.raises(ValueError, match="16x16"):
        LutCorrection(MISLABELED_16X16, 32, 48)


@pytest.mark.skipif(not SLOT7.exists(), reason="calibration data not on this machine")
def test_real_lut_output_is_plane_depth_not_radial():
    """The claim in sensor_model.py, tested on the measured calibration.

    A wall at true depth 200 mm has radial range 200*c at a corner zone. If the
    LUT maps to plane depth, feeding it the raw value measured there returns
    about 200, and a flat wall stays flat.
    """
    data = json.loads(SLOT7.read_text())
    lut = LutCorrection(SLOT7, 32, 48)
    raw = np.array([[next((p["raw_mm"] for p in data["lut"][r][c] if p["true_mm"] == 200.0), np.nan)
                     for c in range(48)] for r in range(32)])
    depth = lut.apply(raw)
    xn, yn = zone_tangents(32, 48)
    points = distance_to_xyz(depth, "plane_z", xn, yn)
    assert np.nanstd(points[..., 2]) < 1.0
    # The wrong interpretation curves the same wall by tens of millimetres.
    wrong = distance_to_xyz(depth, "radial", xn, yn)
    assert np.nanmax(wrong[..., 2]) - np.nanmin(wrong[..., 2]) > 30
