"""Raw-histogram experiment: frame layout, peak rules, multi-surface points."""

import struct
import sys

import numpy as np
import pytest

from hist_scan import evm_hist, peaks, reconstruct
from hist_scan.evm_hist import BINS, COLS, DEFAULT_VENDOR, ROWS


def hist_frame(layout, zones, reference=None):
    reference = np.zeros((4, BINS), np.uint32) if reference is None else reference
    values = np.r_[reference.ravel(), zones.ravel()].astype(np.uint32)
    body = np.stack([values & 255, (values >> 8) & 255, (values >> 16) & 255], 1).astype(np.uint8).tobytes()
    payload = 12 + len(body) + 12
    head = struct.pack("<BBHI3bBHH", 0x25, layout, payload, 7, 25, 26, 27, 0, 0, 0)
    return b"\x00" * 5 + head + body + struct.pack("<IIBBH", 1, 2, 1, 0, 0xE0F0)


def twelve_frames(truth):
    frames = []
    for layout in range(12):
        left = 24 if layout % 2 else 0
        column = 1 if layout in (2, 3, 8, 9) else 2 if layout in (4, 5, 10, 11) else 0
        row = 1 if layout > 5 else 0
        frames.append(hist_frame(layout, truth[row::2, left + column:left + 24:3]))
    return frames


def test_histogram_frames_rebuild_every_zone(rng):
    truth = rng.integers(0, 1 << 20, (ROWS, COLS, BINS), dtype=np.uint32)
    zones, _ = evm_hist.assemble_histograms(twelve_frames(truth))
    assert np.array_equal(zones, truth)


def test_histogram_layout_matches_the_vendor_decoder(rng):
    if not DEFAULT_VENDOR.exists():
        pytest.skip("vendor driver not on this machine")
    sys.path[:0] = [str(DEFAULT_VENDOR), str(DEFAULT_VENDOR / "tmf8829")]
    common = pytest.importorskip("tmf8829.tmf8829_application_common")
    truth = rng.integers(0, 1000, (ROWS, COLS, BINS), dtype=np.uint32)
    frames = twelve_frames(truth)
    _, vendor = common.Tmf8829AppCommon.getAllHistogramResults([bytearray(f) for f in frames])
    zones, _ = evm_hist.assemble_histograms(frames)
    for r, c in ((0, 0), (1, 2), (5, 7), (31, 47), (16, 24), (20, 25)):
        assert list(vendor[r][c].bin) == zones[r, c].tolist()


SCALE = peaks.BinScale(31.0, 10.8)


def pulse(position, height, bins=BINS):
    x = np.arange(bins)
    return height * np.exp(-0.5 * ((x - position) / 0.8) ** 2)


def test_two_surfaces_give_two_peaks_nearest_first():
    h = pulse(18, 60) + pulse(28, 400)
    found = peaks.zone_peaks(h, h, SCALE, peaks.PeakRules())
    assert len(found) == 2
    assert found[0][0] == pytest.approx(SCALE.to_mm(18), abs=3)
    assert found[1][0] == pytest.approx(SCALE.to_mm(28), abs=3)


def test_one_surface_gives_one_peak():
    h = pulse(25, 400)
    assert len(peaks.zone_peaks(h, h, SCALE, peaks.PeakRules())) == 1


def test_crosstalk_near_time_zero_is_not_a_surface():
    h = pulse(11, 900) + pulse(25, 100)
    found = peaks.zone_peaks(h, h, SCALE, peaks.PeakRules())
    assert len(found) == 1 and found[0][0] > 300


def test_noise_is_not_a_surface(rng):
    h = rng.poisson(2.0, BINS).astype(float)
    assert peaks.zone_peaks(h, h, SCALE, peaks.PeakRules()) == []


def test_stray_light_shared_by_all_zones_is_removed():
    """A bright lid at bin 17 leaks into zones that see the table at bin 28."""
    table = pulse(28, 100) + pulse(17, 6)
    grid = np.tile(table, (8, 8, 1))
    grid[:2, :2] = pulse(17, 700)
    cleaned = peaks.remove_stray(grid)
    found = peaks.zone_peaks(cleaned[6, 6] + pulse(28, 100), grid[6, 6], SCALE, peaks.PeakRules())
    assert len(found) == 1 and found[0][0] == pytest.approx(SCALE.to_mm(28), abs=3)


def test_bin_scale_recovers_the_line_despite_second_surfaces(rng):
    true = peaks.BinScale(31.0, 10.8)
    positions = rng.uniform(16, 40, 300)
    hists = np.array([pulse(p, 300) for p in positions])
    chip = true.to_mm(positions)
    chip[:30] += 300.0  # zones where the chip reported the other surface
    fit = peaks.fit_bin_scale(hists, chip)
    assert fit.mm_per_bin == pytest.approx(31.0, abs=0.5)
    assert fit.zero_bin == pytest.approx(10.8, abs=0.3)


def test_points_seen_from_one_view_only_are_dropped():
    wall = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [50, 50, 50.0]])
    views = np.array([0, 1, 2, 0])
    keep = reconstruct.seen_from_other_views(wall, views, 5.0, 2)
    assert keep.tolist() == [True, True, True, False]


def test_peeling_keeps_two_close_surfaces_a_peak_search_cannot_split():
    from hist_scan import refine

    response = pulse(2, 1.0, refine.BEFORE + refine.AFTER + 1)
    h = pulse(26, 900) + pulse(23, 200)
    found = refine.peel(h, h, response, 14, refine.RefineRules(), 1)
    assert len(found) == 2


def test_peeling_does_not_invent_a_surface_from_the_tail():
    from hist_scan import refine

    x = np.arange(refine.BEFORE + refine.AFTER + 1) - refine.BEFORE
    response = np.where(x < 0, np.exp(x * 2.0), np.exp(-x / 2.5))
    h = np.zeros(BINS)
    h[28 - refine.BEFORE:28 + refine.AFTER + 1] = 800 * response
    found = refine.peel(h, h, response, 14, refine.RefineRules(), 1)
    assert len(found) == 1


def test_a_peak_in_one_measurement_only_is_rejected():
    from hist_scan import refine

    stack = np.zeros((15, 2, 2, BINS))
    stack[3, 0, 0, 30] = 500
    stack[:, 1, 1, 30] = 40
    rules = refine.RefineRules()
    assert not refine.present_enough(stack, 0, 0, 30, rules)
    assert refine.present_enough(stack, 1, 1, 30, rules)


def test_a_zone_with_no_neighbour_at_its_range_has_no_support():
    from hist_scan import refine

    ranges = {(5, 5): [300.0], (5, 6): [305.0], (6, 5): [298.0], (9, 9): [500.0]}
    rules = refine.RefineRules()
    assert refine.neighbour_support(ranges, 5, 5, 300.0, rules) == 2
    assert refine.neighbour_support(ranges, 9, 9, 500.0, rules) == 0


def test_range_check_path_goes_up_down_and_back():
    from hist_scan import range_check

    zs = range_check.heights(300, 400, 100, 50)
    assert zs[0] == 300 and zs[-1] == 300 and max(zs) == 400 and min(zs) == 100
    assert np.abs(np.diff(zs)).max() <= 50


def test_range_check_views_change_only_height():
    from hist_scan import range_check

    views = range_check.vertical_views([500, 100, 300, 180, 2, 48], [300, 350])
    assert views[1].tcp_pose == [500, 100, 350.0, 180, 2, 48]


def test_range_scale_is_recovered():
    from hist_scan import range_check

    z = np.arange(100.0, 500, 40)
    slope, offset, residual = range_check.fit_scale(z, 1.1 * (z - z.mean()) + 400)
    assert slope == pytest.approx(1.1) and offset == pytest.approx(400) and np.abs(residual).max() < 1e-9


def test_tilt_plan_returns_level_between_directions():
    from hist_scan import range_check

    plan = range_check.tilt_plan((15.0, 30.0), (0.0, 90.0))
    assert plan == [(0.0, 0.0), (15.0, 0.0), (30.0, 0.0), (15.0, 0.0), (0.0, 0.0),
                    (15.0, 90.0), (30.0, 90.0), (15.0, 90.0), (0.0, 90.0)]


def test_tilted_views_keep_the_sensor_at_one_point_and_range_follows_the_cosine():
    from hhg_tof.geometry import look_at, pose_to_T
    from hist_scan import range_check

    level = look_at([500, 100, 200], [500, 100, 0], [1, 0, 0])
    mount = pose_to_T([34, -62, -28, 1, 2, 26])
    views = range_check.tilt_views(level, mount, [(0.0, 0.0), (30.0, 90.0)])
    for view in views:
        assert np.allclose((pose_to_T(view.tcp_pose) @ mount)[:3, 3], [500, 100, 200], atol=1e-6)
    tilted = pose_to_T(views[1].tcp_pose) @ mount
    expected = range_check.plane_range(tilted, np.array([0, 0, 1.0]), -90.0)
    assert expected == pytest.approx(290 / np.cos(np.radians(30)))
