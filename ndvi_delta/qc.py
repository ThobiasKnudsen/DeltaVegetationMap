"""Classify the PKU GIMMS NDVI quality-control band into reliability tiers.

QC is band 2 of each member ``.tif`` (band 1 is NDVI). Its meaning is era-dependent
(verified against the official Readme, V1.2):

Consolidated product (AVHRR+MODIS) — a 3-digit code ``d1 d2 d3``:
  * d1 consolidation method: 1 = Random Forests, 2-5 = linear/difference matching,
    9 = N/A (MODIS era 2003-2022).
  * d2 pre-consolidation GIMMS QC: 0 = good, 1 = spline interpolation,
    2 = possible snow/cloud, 9 = N/A (2003-2022).
  * d3 MODIS QC: 0 = good, 1 = marginal, 2 = snow/ice, 3 = cloudy,
    4 = estimated from MODIS history, 9 = N/A (1982-2002).
  Era is recoverable from the code itself: AVHRR-era pixels have ``d3 == 9``;
  MODIS-era pixels have ``d1 == 9 and d2 == 9``.

AVHRR-only product — a single digit: 0 = good, 1 = spline interpolation,
2 = possible snow/cloud.

We collapse this into three tiers for the per-pixel reliability overlay. ``65535``
(fill, shared with the NDVI band) and any undocumented code map to ``NODATA``.
"""

from __future__ import annotations

import numpy as np

from .reader import FILL_VALUE

# Reliability tiers.
GOOD = 0          # direct measurement (good AVHRR via Random Forests, or good MODIS)
INTERPOLATED = 1  # gap-filled / modelled / marginal / estimated
SNOW_CLOUD = 2    # snow / ice / cloud contaminated
NODATA = 255      # fill or undocumented code -> contributes to no tier

TIER_NAMES = {GOOD: "good", INTERPOLATED: "interpolated", SNOW_CLOUD: "snow/cloud"}


def classify(qc: np.ndarray, version: str) -> np.ndarray:
    """Map a raw QC band to a ``uint8`` tier array (GOOD / INTERPOLATED / SNOW_CLOUD / NODATA)."""
    qc = np.asarray(qc)
    tier = np.full(qc.shape, NODATA, dtype=np.uint8)
    valid = qc != FILL_VALUE

    if version == "avhrr":
        tier[valid & (qc == 0)] = GOOD
        tier[valid & (qc == 1)] = INTERPOLATED
        tier[valid & (qc == 2)] = SNOW_CLOUD
        return tier

    if version != "consolidated":
        raise ValueError(f"Unknown version {version!r}; expected 'consolidated' or 'avhrr'.")

    d1 = qc // 100
    d2 = (qc // 10) % 10
    d3 = qc % 10

    modis_era = valid & (d1 == 9) & (d2 == 9)
    avhrr_era = valid & ~modis_era

    # MODIS era (2003-2022): driven by the MODIS QC digit.
    tier[modis_era & (d3 == 0)] = GOOD
    tier[modis_era & np.isin(d3, (1, 4))] = INTERPOLATED
    tier[modis_era & np.isin(d3, (2, 3))] = SNOW_CLOUD

    # AVHRR era (1982-2002): snow/cloud first, then anything modelled or spline-filled,
    # then the clean case (measured GIMMS consolidated by Random Forests).
    snow = avhrr_era & (d2 == 2)
    interp = avhrr_era & ~snow & ((d2 == 1) | (d1 >= 2))
    good = avhrr_era & ~snow & ~interp & (d1 == 1) & (d2 == 0)
    tier[snow] = SNOW_CLOUD
    tier[interp] = INTERPOLATED
    tier[good] = GOOD
    return tier
