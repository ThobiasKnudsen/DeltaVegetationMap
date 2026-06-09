"""Interactive global ΔNDVI explorer: Streamlit shell + Leaflet (folium) map.

Pick Period A and Period B with the sliders; the delta recomputes instantly from the
cached annual stack (no re-download, no GeoTIFF re-read). Pan/zoom is native Leaflet.
Toggle the QC-reliability overlay to see how trustworthy each pixel is.

Launch with:  python -m ndvi_delta app    (wraps:  streamlit run ndvi_delta/app.py)
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys

# Make absolute package imports work when run via `streamlit run ndvi_delta/app.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import folium
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st
import zarr
from matplotlib import colors
from streamlit_folium import st_folium

from ndvi_delta import build_cache, render
from ndvi_delta.delta import compute_delta, summary_stats, window_bounds


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--version", default="consolidated")
    args, _ = ap.parse_known_args()
    return args


ARGS = _parse_args()
STACK_PATH = os.path.join(ARGS.data_dir, "cache", build_cache.DEFAULT_STACK_NAME)


@st.cache_resource
def stack_meta(stack_path: str):
    if not os.path.exists(stack_path):
        return None
    root = zarr.open_group(stack_path, mode="r")
    years = list(root.attrs["years"])
    built = sorted(root.attrs.get("built_year_indices", []))
    return {"years": years, "built_years": [years[i] for i in built], "version": root.attrs.get("version")}


@st.cache_data(show_spinner="Computing ΔNDVI…")
def compute(stack_path, pa, pb, fill_mode, mask_sparse, thr):
    res = compute_delta(stack_path, pa, pb, fill_mode=fill_mode, mask_sparse=mask_sparse, sparse_threshold=thr)
    vlim = render.robust_limit(res.delta)
    # Reproject to Web Mercator so the overlay aligns with the Leaflet (EPSG:3857) basemap.
    merc_delta, bounds = render.reproject_to_web_mercator(res.delta, res.transform_origin)
    merc_rel, _ = render.reproject_to_web_mercator(res.reliability, res.transform_origin)
    delta_uri = _data_uri(render.delta_to_rgba(merc_delta, vlim=vlim)[0])
    rel_uri = _data_uri(render.reliability_to_rgba(merc_rel))
    return delta_uri, rel_uri, vlim, bounds, summary_stats(res)


def _data_uri(rgba) -> str:
    """Encode an (H, W, 4) RGBA array as a base64 PNG data URI for a Leaflet overlay."""
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _crosses_seam(pa, pb) -> bool:
    def era(p):
        if p[1] <= 2002:
            return "avhrr"
        if p[0] >= 2003:
            return "modis"
        return "mixed"

    ea, eb = era(pa), era(pb)
    return "mixed" in (ea, eb) or ea != eb


def _colorbar_fig(vlim, label, cmap, from_zero=False):
    norm = colors.Normalize(0.0, vlim) if from_zero else colors.Normalize(-vlim, vlim)
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
    fig, ax = plt.subplots(figsize=(4, 0.45))
    fig.colorbar(sm, cax=ax, orientation="horizontal", label=label)
    ax.tick_params(labelsize=7)
    return fig


def main():
    st.set_page_config(page_title="Global ΔNDVI explorer", layout="wide")
    st.title("Global greenness ΔNDVI explorer")
    st.caption("PKU GIMMS NDVI V1.2 (1982–2022) · per-pixel greening ↔ browning between two periods.")

    meta = stack_meta(STACK_PATH)
    if meta is None or not meta["built_years"]:
        st.error(
            f"No cached stack at `{STACK_PATH}`.\n\nBuild it once (downloads ~2.7 GB):\n\n"
            "```\npython -m ndvi_delta build\n```"
        )
        st.stop()

    yr_lo, yr_hi = meta["built_years"][0], meta["built_years"][-1]
    sb = st.sidebar

    sb.header("Periods")
    pa = sb.slider("Period A", yr_lo, yr_hi, (yr_lo, min(yr_lo + 4, yr_hi)))
    pb = sb.slider("Period B", yr_lo, yr_hi, (max(yr_hi - 4, yr_lo), yr_hi))

    sb.header("Method")
    fill_mode = sb.radio(
        "Fill handling", ["zero", "drop"],
        help="zero: count dormant/snow half-months as 0 — captures growing-season change (default). "
             "drop: average greenness only when green.",
    )
    mask_sparse = sb.checkbox("Mask sparse vegetation", value=True)
    thr = sb.number_input("Sparse threshold (NDVI)", 0.0, 0.5, 0.1, 0.01, disabled=not mask_sparse)

    sb.header("Layers")
    delta_opacity = sb.slider("ΔNDVI opacity", 0.0, 1.0, 0.85, 0.05)
    show_qc = sb.checkbox(
        "Show QC reliability veil", value=False,
        help="Darkens pixels you should distrust (gap-filled / snow / cloud) so reliable data shows through.",
    )
    qc_opacity = sb.slider("QC veil strength", 0.0, 1.0, 0.7, 0.05, disabled=not show_qc)

    needed = set(range(min(pa[0], pb[0]), max(pa[1], pb[1]) + 1))
    if not needed.issubset(set(meta["built_years"])):
        st.warning(f"Stack covers {yr_lo}–{yr_hi}; some selected years aren't built yet.")
        st.stop()

    if _crosses_seam(pa, pb):
        st.warning(
            "⚠️ **AVHRR→MODIS seam.** These periods straddle the ~2002/2003 boundary where the product "
            "switches from AVHRR (1982–2002) to MODIS (2003–2022). It is engineered to be consistent "
            "across this seam, but residual inter-sensor bias is the dominant uncertainty for "
            "early-vs-late comparisons — interpret cross-seam deltas with care."
        )

    delta_uri, rel_uri, vlim, bounds, stats = compute(
        STACK_PATH, tuple(pa), tuple(pb), fill_mode, mask_sparse, float(thr)
    )
    south, west, north, east = bounds

    col_map, col_side = st.columns([3, 1])
    with col_map:
        m = folium.Map(location=[20, 10], zoom_start=2, tiles="CartoDB positron", world_copy_jump=True)
        folium.raster_layers.ImageOverlay(
            image=delta_uri, bounds=[[south, west], [north, east]],
            opacity=delta_opacity, name="ΔNDVI (B − A)",
        ).add_to(m)
        if show_qc:
            folium.raster_layers.ImageOverlay(
                image=rel_uri, bounds=[[south, west], [north, east]],
                opacity=qc_opacity, name="QC reliability veil",
            ).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, height=560, returned_objects=[])

    with col_side:
        st.subheader("ΔNDVI scale")
        st.pyplot(_colorbar_fig(vlim, "ΔNDVI (B − A)", render.DELTA_CMAP), clear_figure=True)
        st.caption("green = greening · red = browning")
        if show_qc:
            st.subheader("QC reliability veil")
            st.caption("Dark = low reliability (gap-filled / snow / cloud). "
                       "Clear = direct measurement — trust what shows through.")
        st.subheader("Summary")
        if stats.get("valid_pixels", 0):
            st.metric("Area-weighted mean Δ", f"{stats['area_weighted_mean_delta']:+.4f}")
            st.metric("Greening (area-wt)", f"{stats['area_weighted_pct_greening']:.1f}%")
            st.metric("Browning (area-wt)", f"{stats['area_weighted_pct_browning']:.1f}%")
            st.metric("Median QC reliability", f"{stats['median_reliability']:.2f}")
            st.caption(f"{stats['valid_pixels']:,} valid pixels")
        else:
            st.info("No valid pixels — adjust periods or thresholds.")

    st.caption(
        "An NDVI delta measures change in greenness/productivity — not land health, soil, or "
        "biodiversity; a positive delta can coexist with degradation. Two-window deltas are "
        "endpoint-sensitive — prefer ≥5-year windows. Coverage stops at the high southern latitudes "
        "(no Antarctica)."
    )


main()
