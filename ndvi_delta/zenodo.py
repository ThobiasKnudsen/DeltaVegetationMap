"""Discover, download, cache, and verify the PKU GIMMS NDVI decade zips from Zenodo.

Record 8253971 (https://doi.org/10.5281/zenodo.8253971), CC-BY-4.0. The product is
packaged as decade zips. We download only the zip(s) whose decade intersects the
requested year range, cache them under ``<data_dir>/zips``, skip what is already
present, and (optionally) verify md5.

The file table below is the offline source of truth (filename -> md5, byte size),
verified against the Zenodo REST API. When the network is up we still prefer the API
for discovery, falling back to this table if it is unreachable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

RECORD_ID = 8253971
BASE = "https://zenodo.org"

# (start_year, end_year, filename) per product version.
CONSOLIDATED_DECADES = [
    (1982, 1990, "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_1982_1990.zip"),
    (1991, 2000, "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_1991_2000.zip"),
    (2001, 2010, "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_2001_2010.zip"),
    (2011, 2022, "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_2011_2022.zip"),
]
AVHRR_DECADES = [
    (1982, 1990, "PKU_GIMMS_NDVI_AVHRR_solely_1982_1990.zip"),
    (1991, 2000, "PKU_GIMMS_NDVI_AVHRR_solely_1991_2000.zip"),
    (2001, 2010, "PKU_GIMMS_NDVI_AVHRR_solely_2001_2010.zip"),
    (2011, 2015, "PKU_GIMMS_NDVI_AVHRR_solely_2011_2015.zip"),
]

# filename -> (md5, size_bytes). Verified against the Zenodo REST API.
CHECKSUMS: dict[str, tuple[str, int]] = {
    "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_1982_1990.zip": ("a838b1be402938ad3a5ceac2c3707949", 595_833_567),
    "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_1991_2000.zip": ("ad646d67a89a10d3838844437260916e", 660_400_157),
    "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_2001_2010.zip": ("c8f95d9e7edba87cb4a3bdb59ba4a8be", 646_081_048),
    "PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_2011_2022.zip": ("225cf139bb1ed40c720c23c90dd659cc", 775_074_040),
    "PKU_GIMMS_NDVI_AVHRR_solely_1982_1990.zip": ("05e0e7744c644de87c8209f3e9029d09", 600_905_705),
    "PKU_GIMMS_NDVI_AVHRR_solely_1991_2000.zip": ("ad30d1d6baae9edcaa97e5829b5b57fe", 667_218_013),
    "PKU_GIMMS_NDVI_AVHRR_solely_2001_2010.zip": ("680866e7d0c191fc4c711249e9610d72", 671_023_050),
    "PKU_GIMMS_NDVI_AVHRR_solely_2011_2015.zip": ("f5a55180ba9d7298585715931d61470d", 340_138_221),
}


@dataclass(frozen=True)
class ZenodoFile:
    name: str
    url: str
    md5: str
    size: int


def file_url(name: str) -> str:
    return f"{BASE}/records/{RECORD_ID}/files/{name}?download=1"


def decades_for(version: str):
    if version == "consolidated":
        return CONSOLIDATED_DECADES
    if version == "avhrr":
        return AVHRR_DECADES
    raise ValueError(f"Unknown version {version!r}; expected 'consolidated' or 'avhrr'.")


def needed_zip_names(year_start: int, year_end: int, version: str) -> list[str]:
    """Names of the decade zips whose span intersects ``[year_start, year_end]``."""
    if year_start > year_end:
        year_start, year_end = year_end, year_start
    return [
        name
        for (s, e, name) in decades_for(version)
        if not (e < year_start or s > year_end)
    ]


def _api_catalog(timeout: int = 30) -> dict[str, ZenodoFile]:
    """Build name -> ZenodoFile from the Zenodo REST API. Raises on network failure."""
    r = requests.get(f"{BASE}/api/records/{RECORD_ID}", timeout=timeout)
    r.raise_for_status()
    out: dict[str, ZenodoFile] = {}
    for f in r.json().get("files", []):
        name = f["key"]
        checksum = f.get("checksum", "")
        md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else checksum
        out[name] = ZenodoFile(name=name, url=file_url(name), md5=md5, size=int(f["size"]))
    return out


def resolve_file(name: str, use_api: bool = True) -> ZenodoFile:
    """Resolve one file's url/md5/size, preferring the API and falling back to the table."""
    if use_api:
        try:
            return _api_catalog()[name]
        except Exception:
            pass  # offline / API hiccup -> fall back to the verified table
    md5, size = CHECKSUMS[name]
    return ZenodoFile(name=name, url=file_url(name), md5=md5, size=size)


def md5sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(file: ZenodoFile, dest: Path, chunk: int = 1 << 20, attempts: int = 3) -> None:
    """Stream a file to *dest* with a progress bar, resuming a partial file if present."""
    for attempt in range(1, attempts + 1):
        existing = dest.stat().st_size if dest.exists() else 0
        if existing == file.size:
            return
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(file.url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already have the whole thing
                    return
                # Server ignored our Range -> start over from byte 0.
                resume = existing > 0 and r.status_code == 206
                r.raise_for_status()
                start = existing if resume else 0
                total = file.size or (int(r.headers["Content-Length"]) + start
                                      if "Content-Length" in r.headers else None)
                mode = "ab" if resume else "wb"
                with open(dest, mode) as fh, tqdm(
                    total=total, initial=start, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=file.name,
                ) as bar:
                    for block in r.iter_content(chunk_size=chunk):
                        fh.write(block)
                        bar.update(len(block))
            if dest.stat().st_size == file.size:
                return
        except requests.RequestException:
            if attempt == attempts:
                raise
    raise IOError(f"Download of {file.name} did not reach the expected size {file.size}.")


def ensure_zips(
    year_start: int,
    year_end: int,
    version: str,
    data_dir: str | Path,
    verify_md5: bool = False,
    use_api: bool = True,
) -> list[Path]:
    """Ensure every decade zip intersecting the year range is present locally.

    Returns the local paths (downloading only what is missing/incomplete). With
    *verify_md5*, each resulting file's checksum is checked against Zenodo's.
    """
    zip_dir = Path(data_dir) / "zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in needed_zip_names(year_start, year_end, version):
        info = resolve_file(name, use_api=use_api)
        dest = zip_dir / name
        if not (dest.exists() and dest.stat().st_size == info.size):
            _download(info, dest)
        if verify_md5:
            actual = md5sum(dest)
            if actual != info.md5:
                raise ValueError(
                    f"md5 mismatch for {name}: got {actual}, expected {info.md5}. "
                    "Delete the file and re-download."
                )
        paths.append(dest)
    return paths
