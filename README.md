# Global NDVI greenness delta-map tool

Map **per-pixel change in greenness** (greening ↔ browning) between any two multi-year
periods in **1982–2022**, anywhere on the globe — as an interactive, zoomable map or a
static GeoTIFF + quicklook PNG. Built for exploring greening vs. desertification
(Sahel, Australia, the Peru/Chile coast, subtropical drying, …).

The tool downloads the data it needs automatically and caches it locally.

> **Note on "greenness" vs. "height".** This maps **NDVI** (a greenness/productivity index),
> **not** vegetation height or structure — NDVI can't measure those. For canopy height you'd
> need a different product (e.g. GEDI / global canopy-height maps) and a different tool.

## Data

[**PKU GIMMS NDVI v1.2**](https://doi.org/10.5281/zenodo.8253971) (Li et al. 2023,
*Earth Syst. Sci. Data* 15:4181–4203) — the best long, **consistent** global NDVI record:
it explicitly corrects NOAA orbital drift and AVHRR sensor degradation and is consolidated
against MODIS, so naive AVHRR/MODIS splicing artefacts are already handled. Don't splice raw
sensors yourself; this product exists to avoid exactly that.

| | |
|---|---|
| Resolution | 1/12° (4320×2160, full globe), half-monthly (24/yr) |
| Encoding | `uint16`, valid 0–1000, scale ×0.001 → NDVI 0–1; fill `65535` = "non-veg **or** NDVI ≤ 0" |
| Bands | band 1 = NDVI, band 2 = QC (per-pixel quality flag) |
| Versions | `consolidated` AVHRR+MODIS 1982–2022 (default); `avhrr` only 1982–2015 |
| Licence | CC-BY-4.0 |

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
# 1) one-time global precompute: downloads ~2.7 GB and builds the per-year stack (~5 min CPU)
python -m ndvi_delta build

# 1b) OPTIONAL: add the ESA CCI/C3S cropland layer (per-year, 1992–2020) for the blue
#     farmland overlay + farmland-vs-natural stats. Heavy one-time download (several GB).
python -m ndvi_delta build-cropland

# 2) interactive map (Streamlit + Leaflet): pick periods, pan/zoom, toggle the QC overlay
python -m ndvi_delta app

# 3) or a headless static export (GeoTIFF + diverging PNG + stats)
python -m ndvi_delta delta --period-a 1983 1987 --period-b 2018 2022
```

`delta`/`app` auto-build the stack on first use if it's missing.

## How it works (and why)

* **Annual stack cache.** `build` reduces every half-monthly GeoTIFF to a per-year stack
  (NDVI sum, valid-count, QC-tier counts) stored as compressed Zarr. Any period delta is then
  a cheap reduction over year slices — so the UI updates **instantly** when you move the period
  sliders, with no re-download and no GeoTIFF re-reads.
* **Land vs. ocean by the *union* of both periods.** A pixel counts as land if it has ≥1 valid
  observation in *either* period. This keeps desert→green / green→desert transitions visible
  (the signal you care about) while still blanking the ocean.
* **Fill handling — zero-fill (default) vs. drop-fill.** Because `65535` means "non-veg **or**
  NDVI ≤ 0", simply dropping fill timesteps averages "greenness only when green" and biases the
  delta toward **spurious greening** when dormant/snow months shrink over time. Zero-fill counts
  those months as 0 (a time-average ≈ annual productivity), removing that bias and capturing
  growing-season-length change. `--fill-mode drop` reproduces the naive behaviour.
* **Sparse-vegetation mask** (`NDVI < 0.1`, recommended by the data authors) is applied on the
  *active* ("greenness when green") mean, so short-season pixels (Sahel, boreal margins) aren't
  over-masked.
* **QC reliability overlay.** QC is per-timestep; we aggregate it per pixel over your chosen
  periods into a "fraction good" score (good vs. interpolated/modelled vs. snow/cloud, decoded
  with the product's era-dependent scheme). Toggle it in the app to see how trustworthy each
  pixel is — e.g. whether a value is a real measurement or gap-filled/snow-contaminated.
* **AVHRR→MODIS seam.** The consolidated product is AVHRR (1982–2002) fused with MODIS
  (2003–2022). Deltas that straddle ~2002/2003 cross that seam; the app warns you. The product
  is engineered to be consistent across it, but residual inter-sensor bias is the dominant
  uncertainty for early-vs-late comparisons.
* **Cropland overlay (optional).** A lot of "greening" is just agriculture (irrigation, cropland
  expansion), which confounds the natural signal. `build-cropland` adds a per-year **cropland
  fraction** from [ESA CCI / C3S Land Cover](https://cds.climate.copernicus.eu/) (300 m, annual
  1992–2020, read as Cloud-Optimized GeoTIFFs from the Microsoft Planetary Computer — anonymous,
  no account). It aggregates to the NDVI grid (each pixel is a clean 30×30 block, no
  reprojection). The app draws cropland in **blue**, with each pixel's opacity scaling to its
  cropland *fraction* (and cropland present in *both* periods shown stronger than newly-farmed or
  abandoned land), plus its own opacity slider, and **splits the stats into farmland vs.
  non-farmland** so you can see how much of the change is agricultural. Periods before 1992 reuse
  the nearest available land-cover year (the app flags this).
* **Stats** are reported both raw and **area-weighted by cos(latitude)** so high-latitude pixels
  don't dominate.

## CLI reference

```
python -m ndvi_delta build  [--years 1982-2022] [--version consolidated|avhrr]
                            [--data-dir DIR] [--verify-md5]

python -m ndvi_delta build-cropland  [--years 1992-2020] [--data-dir DIR]
                            # adds the ESA CCI/C3S cropland layer to an existing stack

python -m ndvi_delta delta  --period-a START END  --period-b START END
                            [--bbox MINLON MINLAT MAXLON MAXLAT]
                            [--fill-mode zero|drop]
                            [--mask-sparse | --no-mask-sparse] [--sparse-threshold 0.1]
                            [--out-dir out] [--no-png] [--version ...] [--data-dir ...]

python -m ndvi_delta app    [--version ...] [--data-dir ...]
```

## Interpretation caveats (please read)

* An NDVI delta measures **change in greenness/productivity** — **not** land "health", soil
  condition, biodiversity, or vegetation height/structure. A **positive** delta can coexist with
  land degradation (the classic Sahel case: greening on sandy soils alongside woody-species
  impoverishment and erosion on shallow soils).
* A two-window delta is **endpoint-sensitive** (wet vs. dry years, ENSO phase). Prefer multi-year
  windows (**≥5 years**) to average out interannual variability.
* **Sparse-vegetation pixels (NDVI < 0.1) are noisy** — masking them is recommended by the data
  authors and is on by default.
* This product already corrects orbital drift and sensor degradation; that's why it's preferred
  over naive AVHRR/MODIS splices.
* **Coverage** is a full-globe grid; vegetated data stops in the high southern latitudes — there
  is **no Antarctica** (those pixels are fill).

## Limitations

* **v1 computes annual deltas.** A month/season filter would need a heavier per-month stack and
  is not yet implemented (the annual stack has no within-year breakdown).
* A Theil–Sen / Mann–Kendall **trend mode** (more robust to endpoint choice than a two-window
  delta) is a planned addition.
* **Cropland** uses ESA CCI/C3S Land Cover, which is annual **1992–2020** only; periods outside
  that reuse the nearest available year. Only *cropland* is distinguished (not pasture/rangeland,
  which global land-cover products don't separate reliably).

## Data attribution

* **NDVI** — PKU GIMMS NDVI v1.2 (Li et al. 2023, *Earth Syst. Sci. Data* 15:4181–4203), CC-BY-4.0.
* **Cropland (optional)** — ESA Climate Change Initiative / Copernicus Climate Change Service (C3S)
  Land Cover, accessed via the Microsoft Planetary Computer. © ESA CCI / Copernicus C3S; used with
  attribution.
