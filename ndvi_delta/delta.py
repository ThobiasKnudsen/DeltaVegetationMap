"""Compute a per-pixel NDVI delta (period B - period A) from the cached annual stack.

All the methodology lives here, as cheap reductions over the cached year slices:

* **Land vs. ocean by the union of both periods.** A pixel is land if it has >= 1 valid
  observation across A *or* B. Off-land stays NaN (transparent); on-land, fill is treated
  as 0 (default). Deciding land over the *union* keeps desert<->green transitions visible
  (the greening/desertification signal) while still blanking the sea.
* **Zero-fill vs. drop-fill mean.** Zero-fill (default) divides by all timesteps, so a
  lengthening growing season reads as greening and the spurious-greening bias from dropping
  dormant months is removed. Drop-fill averages greenness only when green.
* **Sparse mask on the *active* mean.** The <0.1 veg mask uses the drop-fill ("greenness
  when green") mean, so short-growing-season pixels (Sahel, boreal margins) are not
  over-masked by the zero-fill average.
* **QC reliability** = fraction of contributing half-months flagged "good" across both periods.

The annual stack has no within-year breakdown, so a month/season filter is intentionally
not supported here (it would need a heavier per-month stack).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

# Optional per-year cropland-fraction layer (uint8 percent) written by build_cache.build_cropland.
# Same string as landcover.STACK_VAR; duplicated here to keep this module import-light (no
# rasterio/requests just to name a layer).
CROPLAND_VAR = "cropland_pct"


@dataclass
class DeltaResult:
    delta: np.ndarray         # float32 NDVI units; NaN off-land / masked
    mean_a: np.ndarray        # period-A mean (per fill_mode)
    mean_b: np.ndarray
    reliability: np.ndarray   # 0..1 fraction "good" over both periods; NaN off-land
    obs_a: np.ndarray         # valid half-month counts (confidence)
    obs_b: np.ndarray
    years_a: tuple[int, int]
    years_b: tuple[int, int]
    fill_mode: str
    row_slice: slice
    col_slice: slice
    transform_origin: tuple[float, float, float]  # (west, north, pixel_deg) of the window
    # Mean cropland fraction over the two periods (0..1; NaN off-land), or None if the stack has
    # no cropland layer covering both periods. Drives both the gradual blue overlay alpha and the
    # fractional farmland-vs-rest stat weighting.
    cropland_alpha: np.ndarray | None = None


def _period_indices(years: list[int], lo: int, hi: int) -> list[int]:
    if lo > hi:
        lo, hi = hi, lo
    return [i for i, y in enumerate(years) if lo <= y <= hi]


def _bbox_to_slices(attrs, bbox) -> tuple[slice, slice]:
    """Map a (minlon, minlat, maxlon, maxlat) bbox to row/col slices; full grid if None."""
    nrows, ncols = attrs["n_rows"], attrs["n_cols"]
    if bbox is None:
        return slice(0, nrows), slice(0, ncols)
    west, north, px = attrs["west"], attrs["north"], attrs["pixel_deg"]
    minlon, minlat, maxlon, maxlat = bbox
    c0 = int(np.floor((minlon - west) / px))
    c1 = int(np.ceil((maxlon - west) / px))
    r0 = int(np.floor((north - maxlat) / px))
    r1 = int(np.ceil((north - minlat) / px))
    c0, c1 = max(0, c0), min(ncols, c1)
    r0, r1 = max(0, r0), min(nrows, r1)
    if r0 >= r1 or c0 >= c1:
        raise ValueError(f"bbox {bbox} does not intersect the grid.")
    return slice(r0, r1), slice(c0, c1)


def _safe_div(num: np.ndarray, den) -> np.ndarray:
    """num / den as float32, NaN where den <= 0. *den* may be a scalar or an array."""
    num = num.astype(np.float32)
    den = np.broadcast_to(np.asarray(den, dtype=np.float32), num.shape)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _accumulate(root, key: str, idxs: list[int], rows: slice, cols: slice) -> np.ndarray:
    """Sum a stack variable over the selected year indices, within the window."""
    shape = (rows.stop - rows.start, cols.stop - cols.start)
    acc = np.zeros(shape, dtype=np.uint32)
    for i in idxs:
        acc += root[key][i, rows, cols].astype(np.uint32)
    return acc


def _period_cropland_fraction(root, idxs: list[int], rows: slice, cols: slice) -> np.ndarray:
    """Mean cropland fraction (0..1) over a period's years, from the uint8-percent layer."""
    return _accumulate(root, CROPLAND_VAR, idxs, rows, cols).astype(np.float32) / (100.0 * len(idxs))


def compute_delta(
    stack_path: str | Path,
    period_a: tuple[int, int],
    period_b: tuple[int, int],
    fill_mode: str = "zero",
    mask_sparse: bool = True,
    sparse_threshold: float = 0.1,
    bbox: tuple[float, float, float, float] | None = None,
) -> DeltaResult:
    if fill_mode not in ("zero", "drop"):
        raise ValueError("fill_mode must be 'zero' or 'drop'.")
    root = zarr.open_group(str(stack_path), mode="r")
    attrs = root.attrs
    years = list(attrs["years"])
    scale = float(attrs["scale_factor"])
    hmpy = int(attrs["half_months_per_year"])

    ia = _period_indices(years, *period_a)
    ib = _period_indices(years, *period_b)
    if not ia or not ib:
        raise ValueError(
            f"A period has no years in the stack (stack covers {years[0]}-{years[-1]}; "
            f"got A={period_a}, B={period_b})."
        )

    rows, cols = _bbox_to_slices(attrs, bbox)

    sum_a = _accumulate(root, "ndvi_sum", ia, rows, cols)
    sum_b = _accumulate(root, "ndvi_sum", ib, rows, cols)
    obs_a = _accumulate(root, "valid_count", ia, rows, cols)
    obs_b = _accumulate(root, "valid_count", ib, rows, cols)

    # Land = valid at least once across A OR B.
    land = (obs_a + obs_b) > 0

    # Active ("greenness when green") means -> used only for the sparse-veg mask.
    active_a = _safe_div(sum_a * scale, obs_a)
    active_b = _safe_div(sum_b * scale, obs_b)

    # Reported means, per fill_mode.
    if fill_mode == "zero":
        mean_a = _safe_div(sum_a * scale, hmpy * len(ia))
        mean_b = _safe_div(sum_b * scale, hmpy * len(ib))
    else:  # drop
        mean_a, mean_b = active_a, active_b
    mean_a = np.where(land, mean_a, np.nan).astype(np.float32)
    mean_b = np.where(land, mean_b, np.nan).astype(np.float32)

    delta = (mean_b - mean_a).astype(np.float32)

    if mask_sparse:
        # NaN comparisons yield False, so a period with no obs doesn't trigger masking.
        sparse = (active_a < sparse_threshold) | (active_b < sparse_threshold)
        delta = np.where(sparse, np.nan, delta).astype(np.float32)

    # QC reliability over both periods: good / (good + interp + snow).
    good = _accumulate(root, "qc_good", ia, rows, cols) + _accumulate(root, "qc_good", ib, rows, cols)
    interp = _accumulate(root, "qc_interp", ia, rows, cols) + _accumulate(root, "qc_interp", ib, rows, cols)
    snow = _accumulate(root, "qc_snow", ia, rows, cols) + _accumulate(root, "qc_snow", ib, rows, cols)
    reliability = _safe_div(good, good + interp + snow)
    reliability = np.where(land, reliability, np.nan).astype(np.float32)

    # Optional cropland fraction — only if the layer exists and covers both periods. Per pixel we
    # take the mean cropland fraction over the two periods, 0.5*(fa+fb) — which is identically
    # min(fa,fb)+0.5*|fa-fb|, i.e. cropland present in both periods weighed full and gained/lost
    # cropland half. A 30%-cropland pixel is faint and a 90% one strong; the same fraction weights
    # the farmland stat split (a 30% pixel counts 0.3 farmland, 0.7 not).
    cropland_alpha = None
    if CROPLAND_VAR in root:
        built_c = set(root.attrs.get("built_cropland_indices", []))
        if built_c.issuperset(ia) and built_c.issuperset(ib):
            fa = _period_cropland_fraction(root, ia, rows, cols)
            fb = _period_cropland_fraction(root, ib, rows, cols)
            cropland_alpha = np.where(land, np.clip(0.5 * (fa + fb), 0.0, 1.0), np.nan).astype(np.float32)

    west = attrs["west"] + cols.start * attrs["pixel_deg"]
    north = attrs["north"] - rows.start * attrs["pixel_deg"]
    return DeltaResult(
        delta=delta, mean_a=mean_a, mean_b=mean_b, reliability=reliability,
        obs_a=obs_a, obs_b=obs_b, years_a=tuple(period_a), years_b=tuple(period_b),
        fill_mode=fill_mode, row_slice=rows, col_slice=cols,
        transform_origin=(west, north, attrs["pixel_deg"]),
        cropland_alpha=cropland_alpha,
    )


def window_bounds(result: DeltaResult) -> tuple[float, float, float, float]:
    """(south, west, north, east) lat/lon bounds of the result window, for map overlays."""
    west, north, px = result.transform_origin
    h, w = result.delta.shape
    return (north - h * px, west, north, west + w * px)


def row_latitudes(result: DeltaResult) -> np.ndarray:
    """Pixel-center latitude of each row in the result window."""
    _, north, px = result.transform_origin
    h = result.delta.shape[0]
    return north - (np.arange(h) + 0.5) * px


def summary_stats(result: DeltaResult, weights: np.ndarray | None = None) -> dict:
    """Greening/browning summary, both raw and cos(latitude) area-weighted.

    *weights* optionally multiplies each pixel's area weight (NaN treated as 0). Pass a 0..1
    cropland-fraction grid for fractional farmland stats: a 30%-cropland pixel then contributes
    0.3 of its weight to the farmland summary and 0.7 (= 1 − fraction) to the non-farmland one."""
    delta = result.delta
    valid = np.isfinite(delta)
    lat = row_latitudes(result)
    aw = np.cos(np.deg2rad(np.clip(lat, -89.999, 89.999))).astype(np.float32)
    w2d = np.broadcast_to(aw[:, None], delta.shape).astype(np.float32)
    if weights is not None:
        w2d = w2d * np.nan_to_num(np.asarray(weights, dtype=np.float32), nan=0.0)
    valid = valid & (w2d > 0)

    n = int(valid.sum())
    if n == 0:
        return {"valid_pixels": 0}

    d = delta[valid]
    w = w2d[valid]
    wsum = float(w.sum())
    greening = d > 0
    browning = d < 0
    wg = float(w[greening].sum())
    wb = float(w[browning].sum())
    # Magnitude-weighted: how much of the total (area-weighted) |ΔNDVI| is greening vs browning,
    # so a pixel that changed a lot counts more than one that barely moved.
    g_mag = float((d[greening] * w[greening]).sum())
    b_mag = float((-d[browning] * w[browning]).sum())
    tot_mag = g_mag + b_mag
    # Reliability-weighted: also scale each pixel by its QC reliability (0..1), so greening/
    # browning we are confident in counts more than noisy/gap-filled pixels. NaN -> 0 (distrust).
    rel = np.clip(np.nan_to_num(result.reliability[valid], nan=0.0), 0.0, 1.0)
    g_qc = float((d[greening] * w[greening] * rel[greening]).sum())
    b_qc = float((-d[browning] * w[browning] * rel[browning]).sum())
    tot_qc = g_qc + b_qc
    return {
        "valid_pixels": n,
        "mean_delta": float(d.mean()),
        "area_weighted_mean_delta": float((d * w).sum() / wsum),
        "pct_greening": 100.0 * float(greening.sum()) / n,
        "pct_browning": 100.0 * float(browning.sum()) / n,
        "area_weighted_pct_greening": 100.0 * wg / wsum,
        "area_weighted_pct_browning": 100.0 * wb / wsum,
        "greening_intensity_share": 100.0 * g_mag / tot_mag if tot_mag else 0.0,
        "browning_intensity_share": 100.0 * b_mag / tot_mag if tot_mag else 0.0,
        "qc_weighted_greening_share": 100.0 * g_qc / tot_qc if tot_qc else 0.0,
        "qc_weighted_browning_share": 100.0 * b_qc / tot_qc if tot_qc else 0.0,
        "mean_greening": g_mag / wg if wg else 0.0,
        "mean_browning": b_mag / wb if wb else 0.0,
        "median_reliability": float(np.nanmedian(result.reliability[valid])),
    }
