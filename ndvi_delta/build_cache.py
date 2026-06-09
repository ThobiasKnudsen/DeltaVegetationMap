"""Precompute the per-year annual NDVI stack that powers instant, interactive deltas.

The expensive primitive is reading + averaging ~24 half-monthly GeoTIFFs per year.
We do that once and persist, per year per pixel:

* ``ndvi_sum``    (uint16): sum of valid DN values (0..1000) over the year's
  half-months. Fill (NDVI <= 0 or non-veg) contributes nothing. Max 24*1000 < 65535.
* ``valid_count`` (uint8):  number of half-months with a valid NDVI (0..24).
* ``qc_good`` / ``qc_interp`` / ``qc_snow`` (uint8): per-tier counts for the
  reliability overlay.

Stored as a chunked, compressed Zarr group of shape ``(n_years, 2160, 4320)``, one
chunk per year-layer. Any period mean / delta / reliability is then a cheap reduction
over the selected year slices -- the UI never re-reads a GeoTIFF. Crucially we store
``sum`` and ``count`` *separately*, so the period mean can be formed either way at
query time:

* zero-fill (default): ``sum / (24 * n_years)``  -- counts dormant/snow half-months as 0
* drop-fill:           ``sum / valid_count``      -- averages greenness only when green
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from . import landcover, zenodo
from .qc import GOOD, INTERPOLATED, SNOW_CLOUD, classify
from .reader import (
    FILL_VALUE,
    N_COLS,
    N_ROWS,
    PIXEL_DEG,
    SCALE_FACTOR,
    list_ndvi_members,
    parse_date,
    read_member,
)

HALF_MONTHS_PER_YEAR = 24
DEFAULT_STACK_NAME = "annual_stack.zarr"

# Stack variable -> on-disk dtype.
_VARS = {
    "ndvi_sum": "uint16",
    "valid_count": "uint8",
    "qc_good": "uint8",
    "qc_interp": "uint8",
    "qc_snow": "uint8",
}


def _accumulate_year(zip_path, members: list[str], version: str) -> dict[str, np.ndarray]:
    """Reduce one year's half-monthly members into per-pixel sum/count/QC-tier grids."""
    ndvi_sum = np.zeros((N_ROWS, N_COLS), dtype=np.uint32)
    valid_count = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)
    qc_good = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)
    qc_interp = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)
    qc_snow = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)

    for member in members:
        r = read_member(zip_path, member)
        valid = np.isfinite(r.ndvi)
        # Recover integer DN exactly (NDVI = DN * 0.001) only where valid.
        dn = np.zeros((N_ROWS, N_COLS), dtype=np.uint32)
        dn[valid] = np.rint(r.ndvi[valid] / SCALE_FACTOR).astype(np.uint32)
        ndvi_sum += dn
        valid_count += valid.astype(np.uint8)

        tiers = classify(r.qc, version)
        qc_good += (tiers == GOOD).astype(np.uint8)
        qc_interp += (tiers == INTERPOLATED).astype(np.uint8)
        qc_snow += (tiers == SNOW_CLOUD).astype(np.uint8)

    return {
        "ndvi_sum": ndvi_sum.astype(np.uint16),
        "valid_count": valid_count,
        "qc_good": qc_good,
        "qc_interp": qc_interp,
        "qc_snow": qc_snow,
    }


def _create_store(path: Path, years: list[int], version: str) -> zarr.Group:
    root = zarr.open_group(str(path), mode="w")
    ny = len(years)
    for name, dtype in _VARS.items():
        root.create_array(
            name, shape=(ny, N_ROWS, N_COLS), chunks=(1, N_ROWS, N_COLS), dtype=dtype
        )
    root.attrs.update(
        {
            "version": version,
            "years": [int(y) for y in years],
            "n_rows": N_ROWS,
            "n_cols": N_COLS,
            "crs": "EPSG:4326",
            "west": -180.0,
            "north": 90.0,
            "pixel_deg": PIXEL_DEG,
            "scale_factor": SCALE_FACTOR,
            "half_months_per_year": HALF_MONTHS_PER_YEAR,
            "fill_value": FILL_VALUE,
            "qc_tiers": {"good": GOOD, "interpolated": INTERPOLATED, "snow_cloud": SNOW_CLOUD},
            "built_year_indices": [],
        }
    )
    return root


def _map_years_to_members(zips, years: set[int]):
    """year -> (zip_path, [members]) for every requested year present in the zips."""
    year_members: dict[int, list[str]] = {}
    year_zip: dict[int, Path] = {}
    for zp in zips:
        for member in list_ndvi_members(zp):
            y = parse_date(member).year
            if y in years:
                year_members.setdefault(y, []).append(member)
                year_zip[y] = zp
    return year_members, year_zip


def build(
    version: str = "consolidated",
    data_dir: str | Path = "data",
    years: list[int] | None = None,
    stack_path: str | Path | None = None,
    verify_md5: bool = False,
    use_api: bool = True,
    resume: bool = True,
) -> Path:
    """Download the needed decade zips and build/refresh the per-year stack.

    *years* defaults to the full span available for the product version. The build is
    resumable: completed year-layers are recorded and skipped on a re-run.
    """
    if years is None:
        decades = zenodo.decades_for(version)
        years = list(range(decades[0][0], decades[-1][1] + 1))
    years = sorted(set(int(y) for y in years))

    zips = zenodo.ensure_zips(min(years), max(years), version, data_dir, verify_md5, use_api)
    year_members, year_zip = _map_years_to_members(zips, set(years))
    missing = [y for y in years if y not in year_members]
    if missing:
        raise ValueError(f"No NDVI members found for years {missing} in the downloaded zips.")

    stack_path = Path(stack_path) if stack_path else Path(data_dir) / "cache" / DEFAULT_STACK_NAME
    stack_path.parent.mkdir(parents=True, exist_ok=True)

    expected_years = [int(y) for y in years]
    if resume and stack_path.exists():
        root = zarr.open_group(str(stack_path), mode="a")
        if list(root.attrs.get("years", [])) == expected_years and root.attrs.get("version") == version:
            built = set(root.attrs.get("built_year_indices", []))
        else:  # incompatible existing stack -> start fresh
            root = _create_store(stack_path, years, version)
            built = set()
    else:
        root = _create_store(stack_path, years, version)
        built = set()

    todo = [(yi, y) for yi, y in enumerate(years) if yi not in built]
    for yi, y in tqdm(todo, desc="building years", unit="yr"):
        members = sorted(year_members[y], key=parse_date)
        grids = _accumulate_year(year_zip[y], members, version)
        for name in _VARS:
            root[name][yi] = grids[name]
        built.add(yi)
        root.attrs["built_year_indices"] = sorted(built)

    return stack_path


def build_cropland(
    data_dir: str | Path = "data",
    stack_path: str | Path | None = None,
    years: list[int] | None = None,
    resume: bool = True,
) -> Path:
    """Add/refresh the per-year ``cropland_pct`` layer (uint8 percent) in an existing stack,
    from ESA CCI / C3S land cover via the Planetary Computer (see :mod:`ndvi_delta.landcover`).

    The NDVI stack must already exist (run :func:`build` first). Years are aligned to the
    stack's year axis; those outside the land-cover span (1992–2020) reuse the nearest
    available year (the app flags that). The build is resumable per year.
    """
    stack_path = Path(stack_path) if stack_path else Path(data_dir) / "cache" / DEFAULT_STACK_NAME
    if not stack_path.exists():
        raise FileNotFoundError(f"No stack at {stack_path}; run `build` first.")

    root = zarr.open_group(str(stack_path), mode="a")
    stack_years = [int(y) for y in root.attrs["years"]]
    if landcover.STACK_VAR not in root:
        root.create_array(
            landcover.STACK_VAR, shape=(len(stack_years), N_ROWS, N_COLS),
            chunks=(1, N_ROWS, N_COLS), dtype="uint8",
        )
        root.attrs["built_cropland_indices"] = []

    target = set(int(y) for y in years) if years else set(stack_years)
    built = set(root.attrs.get("built_cropland_indices", []))
    todo = [
        (yi, y) for yi, y in enumerate(stack_years)
        if y in target and (yi not in built or not resume)
    ]
    for yi, y in tqdm(todo, desc="cropland years", unit="yr"):
        frac = landcover.cropland_fraction_for_year(y)
        root[landcover.STACK_VAR][yi] = np.rint(frac * 100).astype(np.uint8)
        built.add(yi)
        root.attrs["built_cropland_indices"] = sorted(built)

    return stack_path


def _summary(stack_path: Path) -> None:
    root = zarr.open_group(str(stack_path), mode="r")
    years = list(root.attrs["years"])
    print(f"stack          : {stack_path}")
    print(f"variables      : {list(_VARS)}")
    print(f"shape per var  : {tuple(root['ndvi_sum'].shape)}  dtype ndvi_sum={root['ndvi_sum'].dtype}")
    print(f"years built    : {len(root.attrs['built_year_indices'])}/{len(years)}")

    # Sanity: drop-fill annual mean NDVI for the first year over land pixels.
    s = root["ndvi_sum"][0].astype(np.float32)
    vc = root["valid_count"][0].astype(np.float32)
    land = vc > 0
    mean_ndvi = (s[land] / vc[land]) * root.attrs["scale_factor"]
    print(
        f"year {years[0]} NDVI  : land pixels={int(land.sum()):,}  "
        f"mean={mean_ndvi.mean():.3f}  range=[{mean_ndvi.min():.3f}, {mean_ndvi.max():.3f}]"
    )
    g = int(root["qc_good"][0].sum())
    i = int(root["qc_interp"][0].sum())
    sn = int(root["qc_snow"][0].sum())
    tot = max(g + i + sn, 1)
    print(f"year {years[0]} QC   : good={100*g/tot:.0f}%  interp={100*i/tot:.0f}%  snow/cloud={100*sn/tot:.0f}%")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the per-year NDVI stack.")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--version", default="consolidated", choices=["consolidated", "avhrr"])
    ap.add_argument("--years", default=None, help="inclusive range, e.g. 1982-1990")
    ap.add_argument("--no-api", action="store_true", help="skip Zenodo API; use the offline file table")
    args = ap.parse_args()

    yrs = None
    if args.years:
        lo, hi = (int(v) for v in args.years.split("-"))
        yrs = list(range(lo, hi + 1))

    t0 = time.time()
    path = build(args.version, args.data_dir, years=yrs, use_api=not args.no_api)
    print(f"\nbuilt in {time.time() - t0:.1f}s")
    _summary(path)
