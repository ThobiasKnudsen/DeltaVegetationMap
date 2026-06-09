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
from rasterio.transform import from_origin


def robust_limit(delta: np.ndarray, pct: float = 99.0) -> float:
    """Symmetric color limit: the given percentile of |delta| (diverging maps centre at 0)."""
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        return 1e-6
    v = float(np.percentile(np.abs(finite), pct))
    return v if v > 0 else 1e-6


def _to_rgba(values: np.ndarray, norm, cmap_name: str) -> np.ndarray:
    """Map a float grid through (norm, cmap) to an (H, W, 4) uint8 RGBA, NaN -> transparent."""
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad((0, 0, 0, 0))  # NaN fully transparent
    rgba = cmap(norm(np.ma.masked_invalid(values)), bytes=True)
    return rgba


def delta_to_rgba(delta: np.ndarray, vlim: float | None = None, cmap: str = "BrBG") -> tuple[np.ndarray, float]:
    """Diverging RGBA centred at 0 (brown<->green). Returns (rgba, vlim used)."""
    if vlim is None:
        vlim = robust_limit(delta)
    norm = colors.Normalize(vmin=-vlim, vmax=vlim)
    return _to_rgba(delta, norm, cmap), vlim


def reliability_to_rgba(reliability: np.ndarray, cmap: str = "RdYlGn") -> np.ndarray:
    """Sequential RGBA for the 0..1 QC reliability overlay (red = low, green = high)."""
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    return _to_rgba(reliability, norm, cmap)


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
    cmap: str = "BrBG",
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
