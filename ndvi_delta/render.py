"""Render delta / reliability grids to RGBA PNGs (Leaflet overlays), a labeled quicklook,
and a GeoTIFF. Kept free of any Streamlit/Folium import so it works headless in the CLI."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import colors
from rasterio.transform import from_bounds, from_origin
from rasterio.warp import Resampling, reproject

# Web Mercator (EPSG:3857) world extent: half-width in metres and the latitude it clips at.
_MERC_MAX = 20037508.342789244
_MERC_LAT = 85.051128779806604

# Diverging browning<->greening colormap: vivid red -> white -> vivid green. Endpoints are
# kept bright (not dark) so the strongest values stay easy to read at the edges of the scale.
DELTA_CMAP = colors.LinearSegmentedColormap.from_list(
    "browning_greening",
    ["#fa5252", "#ffc9c9", "#ffffff", "#b2f2bb", "#40c057"],
)


def robust_limit(delta: np.ndarray, pct: float = 99.0) -> float:
    """Symmetric color limit: the given percentile of |delta| (diverging maps centre at 0)."""
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        return 1e-6
    v = float(np.percentile(np.abs(finite), pct))
    return v if v > 0 else 1e-6


def reproject_to_web_mercator(values: np.ndarray, transform_origin, size: int = 3000):
    """Reproject an EPSG:4326 grid (NaN nodata) onto a square Web Mercator (EPSG:3857) grid
    covering the full world extent, so it aligns with a Leaflet/folium Mercator basemap.

    Returns ``(merc_values, bounds_latlon)`` where bounds_latlon is
    ``(south, west, north, east)`` for the image overlay (latitude clips at ±85.05°).
    """
    west, north, px = transform_origin
    src_transform = from_origin(west, north, px, px)
    dst_transform = from_bounds(-_MERC_MAX, -_MERC_MAX, _MERC_MAX, _MERC_MAX, size, size)
    dst = np.full((size, size), np.nan, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(values, dtype=np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:3857",
        src_nodata=float("nan"),
        dst_nodata=float("nan"),
        resampling=Resampling.nearest,
    )
    return dst, (-_MERC_LAT, -180.0, _MERC_LAT, 180.0)


def _to_rgba(values: np.ndarray, norm, cmap) -> np.ndarray:
    """Map a float grid through (norm, cmap) to an (H, W, 4) uint8 RGBA, NaN -> transparent.
    *cmap* may be a registered name or a Colormap instance."""
    cmap = (matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap).copy()
    cmap.set_bad((0, 0, 0, 0))  # NaN fully transparent
    return cmap(norm(np.ma.masked_invalid(values)), bytes=True)


def delta_to_rgba(delta: np.ndarray, vlim: float | None = None, cmap=DELTA_CMAP) -> tuple[np.ndarray, float]:
    """Diverging RGBA centred at 0 (red browning <-> green greening). Returns (rgba, vlim used)."""
    if vlim is None:
        vlim = robust_limit(delta)
    norm = colors.Normalize(vmin=-vlim, vmax=vlim)
    return _to_rgba(delta, norm, cmap), vlim


def fade_rgba_by_reliability(rgba: np.ndarray, reliability: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Fade an RGBA by QC reliability: multiply each pixel's alpha by its reliability (0..1) so
    low-confidence pixels dissolve toward transparent (the basemap shows through) instead of
    being shown at full strength. *strength* interpolates between no fade (0) and the full
    reliability multiply (1). NaN reliability (off-land) -> 0 (treated as fully unreliable)."""
    rel = np.nan_to_num(np.asarray(reliability, dtype=np.float32), nan=0.0)
    factor = np.clip((1.0 - strength) + strength * rel, 0.0, 1.0)
    out = rgba.copy()
    out[..., 3] = (out[..., 3].astype(np.float32) * factor).astype(np.uint8)
    return out


def save_png(rgba: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(path), rgba)
    return path


def write_geotiff(
    array: np.ndarray,
    path: str | Path,
    transform_origin: tuple[float, float, float],
    crs: str = "EPSG:4326",
    nodata: float = np.nan,
) -> Path:
    """Write a single-band float32 GeoTIFF; NaN is the nodata value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    west, north, px = transform_origin
    transform = from_origin(west, north, px, px)
    data = array.astype(np.float32)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=nodata,
        compress="deflate", predictor=3,
    ) as dst:
        dst.write(data, 1)
    return path


def quicklook(
    delta: np.ndarray,
    path: str | Path,
    title: str,
    vlim: float | None = None,
    cmap=DELTA_CMAP,
    bounds: tuple[float, float, float, float] | None = None,
) -> Path:
    """Labeled PNG with a diverging colorbar centred at 0 (the headless CLI deliverable).

    *bounds* is (south, west, north, east) for axis extent, if provided.
    """
    if vlim is None:
        vlim = robust_limit(delta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extent = None
    if bounds is not None:
        south, west, north, east = bounds
        extent = (west, east, south, north)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(delta, cmap=cmap, vmin=-vlim, vmax=vlim, extent=extent, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, extend="both")
    cbar.set_label("ΔNDVI (B − A)")
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    return path
