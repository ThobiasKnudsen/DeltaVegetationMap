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
from ndvi_delta.delta import compute_delta, summary_stats


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


MERC_SIZE = 3000  # Web Mercator overlay resolution (px)


@st.cache_data(show_spinner="Computing ΔNDVI…", max_entries=16)
def compute(stack_path, pa, pb, fade_qc):
    res = compute_delta(stack_path, pa, pb, fill_mode="zero", mask_sparse=False)
    vlim = render.robust_limit(res.delta)
    # Reproject to Web Mercator so the overlay aligns with the Leaflet (EPSG:3857) basemap.
    merc_delta, bounds = render.reproject_to_web_mercator(res.delta, res.transform_origin, size=MERC_SIZE)
    delta_rgba = render.delta_to_rgba(merc_delta, vlim=vlim)[0]
    if fade_qc:
        merc_rel, _ = render.reproject_to_web_mercator(res.reliability, res.transform_origin, size=MERC_SIZE)
        delta_rgba = render.fade_rgba_by_reliability(delta_rgba, merc_rel)
    return _data_uri(delta_rgba), vlim, bounds, summary_stats(res)


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
    st.set_page_config(
        page_title="Global ΔNDVI explorer", layout="wide", initial_sidebar_state="expanded"
    )
    # Map fills the screen; everything else lives in the collapsible left sidebar.
    st.markdown(
        """
        <style>
          header[data-testid="stHeader"] {height: 0; visibility: hidden;}
          [data-testid="stToolbar"] {display: none;}
          .block-container {padding: 0.4rem 0.6rem 0 0.6rem; max-width: 100%;}
          .stApp iframe {height: 93vh !important; width: 100% !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    meta = stack_meta(STACK_PATH)
    if meta is None or not meta["built_years"]:
        st.error(
            f"No cached stack at `{STACK_PATH}`.\n\nBuild it once (downloads ~2.7 GB):\n\n"
            "```\npython -m ndvi_delta build\n```"
        )
        st.stop()

    yr_lo, yr_hi = meta["built_years"][0], meta["built_years"][-1]
    sb = st.sidebar
    sb.title("Global ΔNDVI explorer")
    sb.caption("Where the land got greener or browner between two periods you choose.")

    with sb.expander("ℹ️ New here? What the words mean", expanded=True):
        st.markdown(
            "- **The map** shows the *change in greenness* between two time periods. "
            "🟢 green = the land got **greener** (more plant cover), 🔴 red = it got **browner**.\n"
            "- **NDVI** is a satellite measure of how green/leafy the land is — higher means more "
            "living vegetation. This tool maps the *change* in it, written **ΔNDVI** "
            "(\"delta-NDVI\") = period B − period A.\n"
            "- **Period A / Period B** are the two time windows you compare with the sliders below. "
            "Wider windows (≥5 years) are steadier than single years.\n"
            "- **PKU GIMMS NDVI** is just the name of the dataset behind the map — a global "
            "satellite greenness record (from Peking University), covering 1982–2022.\n"
            "- **QC (quality control)** is how trustworthy each pixel is: a clean satellite reading "
            "vs. one filled in through cloud or snow. The QC toggle fades the shakier pixels."
        )

    # ---- Settings ----
    pa = sb.slider("Period A", yr_lo, yr_hi, (yr_lo, min(yr_lo + 4, yr_hi)))
    pb = sb.slider("Period B", yr_lo, yr_hi, (max(yr_hi - 4, yr_lo), yr_hi))

    needed = set(range(min(pa[0], pb[0]), max(pa[1], pb[1]) + 1))
    if not needed.issubset(set(meta["built_years"])):
        sb.warning(f"Stack covers {yr_lo}–{yr_hi}; some selected years aren't built yet.")
        st.stop()
    if _crosses_seam(pa, pb):
        sb.warning(
            "⚠️ **AVHRR→MODIS seam.** These periods straddle ~2002/2003 (AVHRR→MODIS). "
            "Residual inter-sensor bias is the main uncertainty for cross-seam deltas."
        )

    delta_opacity = sb.slider("ΔNDVI opacity", 0.0, 1.0, 0.85, 0.05)
    fade_qc = sb.checkbox(
        "Fade unreliable data (QC)", value=False,
        help="Multiplies each pixel's opacity by its QC reliability, so gap-filled / snow / cloud "
             "pixels fade toward the basemap instead of showing at full strength.",
    )

    delta_uri, vlim, bounds, stats = compute(STACK_PATH, tuple(pa), tuple(pb), fade_qc)
    south, west, north, east = bounds

    # ---- Stats + legend (sidebar) ----
    sb.divider()
    if stats.get("valid_pixels", 0):
        sb.metric(
            "Area-weighted mean Δ",
            f"{stats['area_weighted_mean_delta']:+.4f}",
            help="Average greenness change across all visible land, with each pixel weighted by "
            "its true ground area (cos-latitude) so large high-latitude pixels don't count for "
            "more than they actually cover. Positive = net greening, negative = net browning; "
            "the value is in NDVI units (the index runs −1 to 1).",
        )
        sb.markdown(
            f"**Share of total change** 🟢 {stats['greening_intensity_share']:.0f}% · "
            f"🔴 {stats['browning_intensity_share']:.0f}%"
        )
        sb.caption(
            f"mean greening +{stats['mean_greening']:.3f} · mean browning −{stats['mean_browning']:.3f}  \n"
            f"{stats['valid_pixels']:,} px · median QC reliability {stats['median_reliability']:.2f}"
        )
        sb.caption(
            "Of all the greenness change — each pixel weighted by its land area and by how much it "
            "changed — this is the split between greening and browning."
        )
    else:
        sb.info("No valid pixels — adjust periods or thresholds.")

    sb.pyplot(_colorbar_fig(vlim, "ΔNDVI (B − A)", render.DELTA_CMAP), clear_figure=True)
    sb.caption("green = greening · red = browning")
    if fade_qc:
        sb.caption("**QC fade:** faint = low reliability (gap-filled / snow / cloud); solid = direct measurement.")

    with sb.expander("ℹ️ Interpretation notes"):
        st.caption(
            "An NDVI delta measures change in greenness/productivity — not land health, soil, or "
            "biodiversity; a positive delta can coexist with degradation. Two-window deltas are "
            "endpoint-sensitive — prefer ≥5-year windows. Coverage stops at the high southern "
            "latitudes (no Antarctica)."
        )

    # ---- Full-screen map ----
    # The ImageOverlay is baked into the map (feature_group_to_add can't render raster overlays),
    # so st_folium re-mounts the iframe whenever the map HTML changes. We exploit that: the HTML
    # only changes when the *data* changes (overlay/opacity), not when the user pans — so panning
    # re-runs the script but never re-mounts, staying flicker-free. To keep the view across a data
    # change we mount the map at wherever the user last left it (`map_mount`), updated only on a
    # data change. Mounting at the saved location (vs. feeding st_folium's center/zoom inputs) sets
    # the view at creation time, so there's no jump-to-default flash and no setView feedback loop.
    data_sig = (tuple(pa), tuple(pb), fade_qc, delta_opacity)
    if st.session_state.get("map_data_sig") != data_sig:
        if st.session_state.get("map_view"):  # carry the live view into the next mount
            st.session_state["map_mount"] = st.session_state["map_view"]
        st.session_state["map_data_sig"] = data_sig
    mount = st.session_state.get("map_mount")

    m = folium.Map(
        location=mount["center"] if mount else [25, 5],
        zoom_start=mount["zoom"] if mount else 3,
        tiles="CartoDB positron", world_copy_jump=True,
        zoomSnap=0.25, zoomDelta=0.25, wheelPxPerZoomLevel=120,
    )
    folium.raster_layers.ImageOverlay(
        image=delta_uri, bounds=[[south, west], [north, east]],
        opacity=delta_opacity, name="ΔNDVI (B − A)",
    ).add_to(m)
    out = st_folium(
        m, key="ndvi_map", height=900, use_container_width=True,
        returned_objects=["center", "zoom"],
    )
    # Remember where the user is now (true Leaflet center, so no Web-Mercator drift) so the next
    # data update can re-mount in place.
    if out:
        center, zoom = out.get("center"), out.get("zoom")
        if center and zoom is not None:
            st.session_state["map_view"] = {
                "center": [center["lat"], center["lng"]], "zoom": zoom
            }


main()
