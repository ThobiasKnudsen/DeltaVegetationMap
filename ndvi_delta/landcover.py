"""Per-year **cropland fraction** from ESA CCI / C3S Land Cover, on the NDVI grid.

The ΔNDVI greening signal is heavily confounded by agriculture: irrigated cropland and
cropland expansion read as strong "greening" that is nothing like natural recovery. This
module turns the ESA CCI / C3S annual land-cover maps into a per-pixel *cropland fraction*
on the same 1/12° grid as the NDVI stack, so the app can colour cropland separately and
split the greening/browning stats into farmland vs. non-farmland.

Source: **ESA CCI / C3S Land Cover** (300 m, annual 1992–2020), read as Cloud-Optimized
GeoTIFFs from the **Microsoft Planetary Computer** STAC. Access is anonymous: the public
``/sas/v1/sign`` endpoint signs each asset URL with no account, and rasterio streams the COG
over ``/vsicurl/``. Licence: ESA CCI / Copernicus C3S — free for use with attribution (see
README); we redistribute nothing, only the derived fraction.

Grid alignment is exact and is what keeps this cheap: the native grid is EPSG:4326 at
1/360°, origin (−180, +90) — precisely 30× finer than, and co-registered with, the NDVI grid
(1/12° = 30 × 1/360°). So each NDVI pixel maps to a clean 30×30 block of CCI pixels and the
cropland fraction is just the block-mean of the cropland mask — no reprojection. The globe
ships as 32 (45°×45°) tiles per year; we read each, aggregate, and drop it into the global
fraction grid.
"""

from __future__ import annotations

import time

import numpy as np
import rasterio
import requests
from rasterio.windows import Window

from .reader import N_COLS, N_ROWS, PIXEL_DEG  # 2160, 4320, 1/12

# CCI/C3S LCCS classes counted as cropland. 10 = rainfed cropland, 11/12 = its herbaceous /
# tree-shrub subclasses, 20 = irrigated or post-flooding, 30 = mosaic cropland (>50%) /
# natural vegetation. Class 40 (mosaic <50% cropland) is intentionally NOT counted.
CROPLAND_CLASSES = (10, 11, 12, 20, 30)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
COLLECTION = "esa-cci-lc"

CCI_PIXEL_DEG = 1.0 / 360.0
BLOCK = int(round(PIXEL_DEG / CCI_PIXEL_DEG))  # 30 CCI pixels per NDVI pixel, per axis
CCI_YEAR_MIN, CCI_YEAR_MAX = 1992, 2020

# Name of the per-year cropland-fraction layer this module populates in the NDVI stack,
# stored as uint8 percent (0–100). delta.py references the same string (kept import-light).
STACK_VAR = "cropland_pct"


def cropland_mask(lccs: np.ndarray) -> np.ndarray:
    """Boolean cropland mask from a raw LCCS class array."""
    return np.isin(lccs, CROPLAND_CLASSES)


def aggregate_fraction(mask: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Block-mean of a boolean mask over non-overlapping ``block × block`` windows → float32
    fraction in [0, 1]. Both dims must be exact multiples of *block* (CCI tiles are
    16200 = 540 × 30, so this always holds for full tiles)."""
    h, w = mask.shape
    if h % block or w % block:
        raise ValueError(f"mask shape {mask.shape} not divisible by block {block}")
    return mask.reshape(h // block, block, w // block, block).mean(axis=(1, 3), dtype=np.float32)


def _search_year(year: int, tries: int = 6, timeout: int = 90) -> list[dict]:
    """All CCI/C3S land-cover tiles (STAC items) for *year*. Retries the occasionally-flaky
    Planetary Computer API with backoff."""
    body = {
        "collections": [COLLECTION],
        "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
        "limit": 100,
    }
    last = None
    for i in range(tries):
        try:
            r = requests.post(f"{PC_STAC}/search", json=body, timeout=timeout)
            if r.ok:
                feats = r.json().get("features", [])
                if feats:
                    return feats
                last = "no features returned"
            else:
                last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(3 * (i + 1))
    raise IOError(f"Planetary Computer STAC search for {year} failed: {last}")


def _sign(href: str, tries: int = 4, timeout: int = 30) -> str:
    """Sign a Planetary Computer asset href for anonymous read (no account needed)."""
    last = None
    for i in range(tries):
        try:
            r = requests.get(PC_SIGN, params={"href": href}, timeout=timeout)
            if r.ok:
                return r.json()["href"]
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise IOError(f"Planetary Computer signing failed ({last}) for {href}")


def _place(frac: np.ndarray, sub: np.ndarray, bbox) -> None:
    """Drop a tile's aggregated fraction *sub* into the global grid *frac* at its bbox origin.
    The grid is north-up (row 0 = +90°), so the tile's top edge is its max latitude."""
    lon0, _lat0, _lon1, lat1 = bbox
    row0 = int(round((90.0 - lat1) / PIXEL_DEG))
    col0 = int(round((lon0 + 180.0) / PIXEL_DEG))
    frac[row0:row0 + sub.shape[0], col0:col0 + sub.shape[1]] = sub


def cropland_fraction_for_year(year: int, read_window: Window | None = None) -> np.ndarray:
    """Global cropland fraction grid ``(N_ROWS, N_COLS)`` float32 in [0, 1] for *year*.

    Years outside the CCI/C3S span (1992–2020) clamp to the nearest available year; the caller
    is responsible for flagging that to the user. *read_window* is for tests only (restricts
    the per-tile read to a sub-window to avoid pulling full 16200² tiles)."""
    src_year = min(max(int(year), CCI_YEAR_MIN), CCI_YEAR_MAX)
    frac = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
    for feat in _search_year(src_year):
        signed = _sign(feat["assets"]["lccs_class"]["href"])
        with rasterio.open("/vsicurl/" + signed) as ds:
            lccs = ds.read(1, window=read_window)
        _place(frac, aggregate_fraction(cropland_mask(lccs)), feat["bbox"])
    return frac
