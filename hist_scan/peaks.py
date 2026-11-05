"""Peaks in raw histograms: one zone can see a see-through surface and what lies behind it.

Three things in the raw data are not surfaces, all measured on this sensor:
- Bins around time zero hold internal crosstalk, in every zone.
- A strong target leaks into every other zone at its own distance. The per-bin median
  over zones estimates that, because any real surface fills only part of the view.
- Single counts are Poisson noise, so a peak has to clear the noise of its background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

MIN_FIT_ZONES = 30
MIN_FIT_BIN = 14


@dataclass
class BinScale:
    mm_per_bin: float
    zero_bin: float

    def to_mm(self, bin_position):
        return (np.asarray(bin_position, float) - self.zero_bin) * self.mm_per_bin


@dataclass
class PeakRules:
    min_range_mm: float = 60.0      # inside this is crosstalk
    min_counts: float = 8.0         # after stray removal, in counts per measurement
    noise_sigmas: float = 5.0
    min_ratio: float = 0.05         # to the strongest peak of the zone
    min_separation_bins: int = 3    # the pulse is about 2 bins wide
    max_peaks: int = 2


def centroid(histogram: np.ndarray, peak: int) -> float:
    """Sub-bin position from the peak bin and its neighbours."""
    lo, hi = max(peak - 1, 0), min(peak + 2, len(histogram))
    weights = histogram[lo:hi].astype(float)
    return float((weights * np.arange(lo, hi)).sum() / weights.sum())


def fit_bin_scale(histograms: np.ndarray, chip_mm: np.ndarray) -> BinScale:
    """Bins -> radial mm, from zones where the chip reported the strongest peak.

    histograms (N, bins) and chip_mm (N,). Least squares with outlier rejection, since a
    zone with two surfaces can report one while the histogram maximum is the other.
    """
    positions, distances = [], []
    for h, mm in zip(histograms, chip_mm):
        peak = int(np.argmax(h[MIN_FIT_BIN:])) + MIN_FIT_BIN
        if mm > 0 and h[peak] > 0:
            positions.append(centroid(h, peak))
            distances.append(float(mm))
    if len(positions) < MIN_FIT_ZONES:
        raise ValueError(f"only {len(positions)} zones to fit the bin scale")
    x, y = np.array(positions), np.array(distances)
    keep = np.ones(len(x), bool)
    for _ in range(6):
        slope, offset = np.polyfit(x[keep], y[keep], 1)
        error = np.abs(slope * x + offset - y)
        keep = error <= max(3 * np.median(error[keep]), 5.0)
    return BinScale(float(slope), float(-offset / slope))


def remove_stray(histograms: np.ndarray) -> np.ndarray:
    """(rows, cols, bins) minus what every zone shares, clipped at zero."""
    shared = np.median(histograms.reshape(-1, histograms.shape[-1]), axis=0)
    return np.clip(histograms - shared, 0.0, None)


def zone_peaks(cleaned: np.ndarray, raw: np.ndarray, scale: BinScale, rules: PeakRules,
               measurements: int = 1) -> list[tuple[float, float]]:
    """Up to `max_peaks` (radial mm, counts per measurement) of one zone, nearest first."""
    first_bin = int(np.ceil(scale.zero_bin + rules.min_range_mm / scale.mm_per_bin))
    if cleaned[first_bin:].max(initial=0.0) <= 0:
        return []
    floor = rules.min_counts * measurements
    tail = raw[first_bin:]
    noise = float(np.median(tail))
    height = max(floor, noise + rules.noise_sigmas * np.sqrt(max(noise, 1.0)))
    # Search the whole histogram and filter after: cutting at first_bin would turn the
    # falling edge of the crosstalk into a peak at the cut.
    bins, props = find_peaks(np.r_[cleaned, 0.0], height=height, distance=rules.min_separation_bins)
    heights = props["peak_heights"]
    inside = bins >= first_bin
    bins, heights = bins[inside], heights[inside]
    if len(bins) == 0:
        return []
    keep = heights >= rules.min_ratio * heights.max()
    bins, heights = bins[keep], heights[keep]
    strongest = np.argsort(heights)[::-1][:rules.max_peaks]
    found = [(float(scale.to_mm(centroid(cleaned, int(bins[i])))), float(heights[i]) / measurements)
             for i in strongest]
    return sorted(found)
