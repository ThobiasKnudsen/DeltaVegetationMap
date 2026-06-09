"""Command-line interface: build the stack, export a static delta, or launch the app.

  python -m ndvi_delta build  --years 1982-2022          # one-time global precompute
  python -m ndvi_delta delta  --period-a 1983 1987 --period-b 2018 2022
  python -m ndvi_delta app                               # interactive map

``delta`` auto-builds the stack on first use if it is missing, so the headless one-shot
"just works".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import zarr

from . import build_cache, render
from .delta import compute_delta, summary_stats, window_bounds


def _stack_covers(stack_path: Path, required_years, version: str) -> bool:
    if not stack_path.exists():
        return False
    root = zarr.open_group(str(stack_path), mode="r")
    if root.attrs.get("version") != version:
        return False
    years = list(root.attrs.get("years", []))
    built = set(root.attrs.get("built_year_indices", []))
    return all(y in years and years.index(y) in built for y in required_years)


def _ensure_stack(data_dir: str, required_years, version: str, verify_md5: bool) -> Path:
    stack_path = Path(data_dir) / "cache" / build_cache.DEFAULT_STACK_NAME
    if _stack_covers(stack_path, required_years, version):
        return stack_path
    print(
        "No cached stack covers the requested years -> building (one-time; downloads data).",
        file=sys.stderr,
    )
    return build_cache.build(version, data_dir, years=None, verify_md5=verify_md5)


def cmd_build(args):
    years = None
    if args.years:
        lo, hi = (int(v) for v in args.years.split("-"))
        years = list(range(lo, hi + 1))
    path = build_cache.build(args.version, args.data_dir, years=years, verify_md5=args.verify_md5)
    build_cache._summary(path)
    print(f"\nstack ready: {path}")


def cmd_build_cropland(args):
    years = None
    if args.years:
        lo, hi = (int(v) for v in args.years.split("-"))
        years = list(range(lo, hi + 1))
    path = build_cache.build_cropland(args.data_dir, years=years)
    print(f"\ncropland layer ready: {path}")


def cmd_delta(args):
    pa, pb = tuple(args.period_a), tuple(args.period_b)
    required = list(range(min(pa[0], pb[0]), max(pa[1], pb[1]) + 1))
    stack_path = _ensure_stack(args.data_dir, required, args.version, args.verify_md5)

    res = compute_delta(
        stack_path, pa, pb,
        fill_mode=args.fill_mode,
        mask_sparse=args.mask_sparse,
        sparse_threshold=args.sparse_threshold,
        bbox=tuple(args.bbox) if args.bbox else None,
    )
    stats = summary_stats(res)

    out = Path(args.out_dir)
    tag = f"{pa[0]}-{pa[1]}_vs_{pb[0]}-{pb[1]}_{res.fill_mode}"
    outputs = [render.write_geotiff(res.delta, out / f"delta_{tag}.tif", res.transform_origin)]
    if not args.no_png:
        title = f"ΔNDVI: {pb[0]}–{pb[1]} minus {pa[0]}–{pa[1]}  ({res.fill_mode}-fill)"
        outputs.append(render.quicklook(res.delta, out / f"delta_{tag}.png", title, bounds=window_bounds(res)))

    print("\n=== ΔNDVI summary ===")
    if stats.get("valid_pixels", 0) == 0:
        print("No valid pixels (everything masked). Check periods / bbox / thresholds.")
    else:
        print(f"valid pixels         : {stats['valid_pixels']:,}")
        print(f"mean Δ              : {stats['mean_delta']:+.4f}")
        print(f"area-weighted mean Δ : {stats['area_weighted_mean_delta']:+.4f}")
        print(f"greening / browning  : {stats['pct_greening']:.1f}% / {stats['pct_browning']:.1f}%")
        print(f"  area-weighted      : {stats['area_weighted_pct_greening']:.1f}% / {stats['area_weighted_pct_browning']:.1f}%")
        print(f"median QC reliability: {stats['median_reliability']:.2f}")
    print("\noutputs:")
    for o in outputs:
        print(f"  {o}")


def cmd_app(args):
    import subprocess

    app_py = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_py),
         "--", "--data-dir", args.data_dir, "--version", args.version],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data", help="download + cache directory")
    common.add_argument("--version", default="consolidated", choices=["consolidated", "avhrr"])
    common.add_argument("--verify-md5", action="store_true", help="verify zip checksums against Zenodo")

    p = argparse.ArgumentParser(
        prog="ndvi_delta",
        description="Global NDVI greenness delta-map tool (PKU GIMMS NDVI V1.2).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("build", parents=[common], help="download data and build the per-year stack")
    pb.add_argument("--years", default=None, help="inclusive range, e.g. 1982-2022 (default: full record)")
    pb.set_defaults(func=cmd_build)

    pc = sub.add_parser(
        "build-cropland", parents=[common],
        help="add the ESA CCI/C3S cropland layer to an existing stack (downloads land cover)",
    )
    pc.add_argument("--years", default=None, help="inclusive range, e.g. 1992-2020 (default: full stack)")
    pc.set_defaults(func=cmd_build_cropland)

    pd = sub.add_parser("delta", parents=[common], help="export a static ΔNDVI GeoTIFF + PNG + stats")
    pd.add_argument("--period-a", type=int, nargs=2, required=True, metavar=("START", "END"))
    pd.add_argument("--period-b", type=int, nargs=2, required=True, metavar=("START", "END"))
    pd.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"))
    pd.add_argument("--fill-mode", default="zero", choices=["zero", "drop"],
                    help="zero: count dormant months as 0 (default); drop: average greenness only when green")
    pd.add_argument("--mask-sparse", action=argparse.BooleanOptionalAction, default=True)
    pd.add_argument("--sparse-threshold", type=float, default=0.1)
    pd.add_argument("--out-dir", default="out")
    pd.add_argument("--no-png", action="store_true")
    pd.set_defaults(func=cmd_delta)

    pa = sub.add_parser("app", parents=[common], help="launch the interactive Streamlit + Leaflet map")
    pa.set_defaults(func=cmd_app)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
