"""Read PKU GIMMS NDVI half-monthly GeoTIFFs directly from the Zenodo decade zips.

Verified against ``Readme_for_PKU_GIMMS_NDVI_Product_updated_20230817.pdf`` (V1.2):

* Each member ``.tif`` is a **2-band** raster: band 1 = NDVI, band 2 = QC flag.
* NDVI is ``uint16``; valid range ``0-1000``; scale ``0.001`` (NDVI = DN * 0.001);
  fill value ``65535`` means "Non-veg **or** NDVI <= 0" (so fill is overloaded:
  permanent non-veg *and* land pixels that were <= 0 this half-month both map to it).
* Grid is full-globe ``4320 x 2160`` at 1/12 deg, geographic (EPSG:4326),
  origin (-180, +90).
* Member names: ``PKU_GIMMS_NDVI_<version>_<YYYYMMHH>.tif`` where the 8-digit token
  is ``year*10000 + month*100 + half`` and ``half`` is 01 (first) or 02 (second).
  Example: ``PKU_GIMMS_NDVI_V1.2_20010101.tif`` = first half of January 2001.

The reader stays faithful to the source: fill/out-of-range -> NaN. The decision to
treat fill-on-land as 0 (vs. dropping it) belongs to the cache builder, not here.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio import windows as rio_windows
from rasterio.crs import CRS
from rasterio.transform import from_origin

# --- Dataset constants (verified against the official Readme, V1.2) ---
FILL_VALUE = 65535
VALID_MIN = 0
VALID_MAX = 1000
SCALE_FACTOR = 0.001
N_ROWS = 2160
N_COLS = 4320
PIXEL_DEG = 1.0 / 12.0

# Full-globe georeferencing, used as a fallback when a member .tif carries no embedded
# transform/CRS (the product's TIFFs are not guaranteed to be georeferenced GeoTIFFs).
DEFAULT_TRANSFORM = from_origin(-180.0, 90.0, PIXEL_DEG, PIXEL_DEG)
DEFAULT_CRS = CRS.from_epsg(4326)

# 8-digit date token immediately before the .tif extension.
_DATE_TOKEN = re.compile(r"_(\d{8})\.tif$", re.IGNORECASE)


@dataclass(frozen=True, order=True)
class Timestep:
    """A half-month timestep. Ordering is chronological (year, month, half)."""

    year: int
    month: int
    half: int  # 1 = first half-month, 2 = second

    @property
    def index_in_year(self) -> int:
        """0-based index of this half-month within its year, 0..23."""
        return (self.month - 1) * 2 + (self.half - 1)

    def __str__(self) -> str:
        return f"{self.year}-{self.month:02d}-h{self.half}"


def parse_date(name: str) -> Timestep:
    """Parse a member filename into a :class:`Timestep`. Year is the only token the
    annual delta strictly needs; month/half enable the optional season filter."""
    m = _DATE_TOKEN.search(name)
    if not m:
        raise ValueError(f"Cannot parse date token from member name: {name!r}")
    token = m.group(1)
    year, month, half = int(token[:4]), int(token[4:6]), int(token[6:8])
    if not (1982 <= year <= 2022 and 1 <= month <= 12 and half in (1, 2)):
        raise ValueError(f"Implausible date {token!r} parsed from member name: {name!r}")
    return Timestep(year, month, half)


def list_ndvi_members(zip_path: str | Path) -> list[str]:
    """Internal paths of the NDVI ``.tif`` members in *zip_path*, chronologically sorted.

    Members are robust to being nested in a folder inside the archive; the returned
    paths are usable directly with :func:`vsizip_path`.
    """
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".tif")]
    if not members:
        raise ValueError(
            f"No .tif members found in {zip_path}. The archive layout may differ from "
            "the documented V1.2 scheme; re-check the Readme."
        )
    members.sort(key=parse_date)
    return members


def vsizip_path(zip_path: str | Path, member: str) -> str:
    """GDAL virtual path to read *member* straight from the zip with no extraction."""
    return f"/vsizip/{Path(zip_path).as_posix()}/{member}"


@dataclass
class NdviRead:
    ndvi: np.ndarray            # float32, NDVI in [0, 1]; NaN where fill/out-of-range
    qc: np.ndarray             # raw QC band (uint16), unmodified
    timestep: Timestep
    transform: Affine
    crs: CRS
    used_fallback_georef: bool  # True if the file lacked georef and DEFAULT_* was used


def _base_georef(ds) -> tuple[Affine, CRS, bool]:
    """Return (transform, crs, used_fallback) for the full grid of an open dataset."""
    transform, crs = ds.transform, ds.crs
    if crs is None or transform is None or transform == Affine.identity():
        return DEFAULT_TRANSFORM, DEFAULT_CRS, True
    return transform, crs, False


def scale_ndvi(raw: np.ndarray) -> np.ndarray:
    """Convert a raw NDVI band to float32 NDVI in [0, 1], with fill/out-of-range -> NaN."""
    ndvi = raw.astype(np.float32)
    # Anything outside 0..1000 (which includes the 65535 fill) is invalid.
    invalid = (raw < VALID_MIN) | (raw > VALID_MAX)
    ndvi[invalid] = np.nan
    ndvi *= SCALE_FACTOR
    return ndvi


def read_member(zip_path: str | Path, member: str, window=None) -> NdviRead:
    """Read one half-monthly member: scaled NDVI (band 1) and raw QC (band 2).

    *window* is an optional :class:`rasterio.windows.Window` for a regional crop,
    so a bbox study never reads the full globe.
    """
    with rasterio.open(vsizip_path(zip_path, member)) as ds:
        if ds.count < 2:
            raise ValueError(
                f"Expected >= 2 bands (NDVI + QC) in {member}, found {ds.count}. "
                "This contradicts the documented V1.2 layout; re-check the Readme."
            )
        raw_ndvi = ds.read(1, window=window)
        qc = ds.read(2, window=window)
        base_transform, crs, used_fallback = _base_georef(ds)
    transform = (
        base_transform if window is None
        else rio_windows.transform(window, base_transform)
    )
    return NdviRead(
        ndvi=scale_ndvi(raw_ndvi),
        qc=qc,
        timestep=parse_date(member),
        transform=transform,
        crs=crs,
        used_fallback_georef=used_fallback,
    )


def _smoke(zip_path: str) -> None:
    """Validate every dataset assumption against a real zip. Run:
    ``python -m ndvi_delta.reader <path-to-decade.zip>``"""
    members = list_ndvi_members(zip_path)
    first, last = parse_date(members[0]), parse_date(members[-1])
    print(f"zip                : {zip_path}")
    print(f"NDVI .tif members  : {len(members)}  (expect 24 * n_years)")
    print(f"first / last member: {members[0]}  ..  {members[-1]}")
    print(f"date span          : {first}  ..  {last}")

    r = read_member(zip_path, members[0])
    print(f"bands read         : NDVI {r.ndvi.shape} {r.ndvi.dtype} | QC {r.qc.shape} {r.qc.dtype}")
    print(f"grid == 2160x4320  : {r.ndvi.shape == (N_ROWS, N_COLS)}")
    print(f"CRS                : {r.crs}")
    print(f"transform          : {tuple(round(v, 4) for v in r.transform[:6])}")
    print(f"used georef fallbck: {r.used_fallback_georef}")

    valid = np.isfinite(r.ndvi)
    if valid.any():
        print(f"valid NDVI         : {100 * valid.mean():.1f}% of pixels, "
              f"range [{np.nanmin(r.ndvi):.3f}, {np.nanmax(r.ndvi):.3f}]")
    else:
        print("valid NDVI         : NONE — check fill handling")

    uniq, counts = np.unique(r.qc, return_counts=True)
    top = sorted(zip(counts.tolist(), uniq.tolist()), reverse=True)[:10]
    print("top QC codes (n)   : " + ", ".join(f"{v}:{c}" for c, v in top))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m ndvi_delta.reader <path-to-decade.zip>")
    _smoke(sys.argv[1])
