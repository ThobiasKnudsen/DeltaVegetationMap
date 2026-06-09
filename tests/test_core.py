"""Deterministic unit tests for the trickiest logic: QC era-decoding and the delta math
(union land test, zero-fill vs drop-fill, reliability). No network or real data needed.

Run:  .venv/bin/python -m unittest discover -s tests
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import zarr

from ndvi_delta import landcover, render, zenodo
from ndvi_delta.delta import CROPLAND_VAR, compute_delta, summary_stats
from ndvi_delta.qc import GOOD, INTERPOLATED, NODATA, SNOW_CLOUD, classify
from ndvi_delta.reader import FILL_VALUE, Timestep, parse_date, scale_ndvi


class TestReader(unittest.TestCase):
    def test_parse_date(self):
        t = parse_date("PKU_GIMMS_NDVI_V1.2_20010101.tif")
        self.assertEqual((t.year, t.month, t.half), (2001, 1, 1))
        nested = parse_date("consolidated_1982_1990/PKU_GIMMS_NDVI_V1.2_19821202.tif")
        self.assertEqual((nested.year, nested.month, nested.half), (1982, 12, 2))
        self.assertEqual(nested.index_in_year, 23)

    def test_parse_date_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parse_date("not_a_member.tif")
        with self.assertRaises(ValueError):
            parse_date("PKU_GIMMS_NDVI_V1.2_20011301.tif")  # month 13

    def test_scale_ndvi(self):
        raw = np.array([[0, 1000, 1001, FILL_VALUE]], dtype=np.uint16)
        s = scale_ndvi(raw)
        self.assertAlmostEqual(s[0, 0], 0.0)
        self.assertAlmostEqual(s[0, 1], 1.0)
        self.assertTrue(np.isnan(s[0, 2]))   # out of valid range
        self.assertTrue(np.isnan(s[0, 3]))   # fill

    def test_timestep_ordering(self):
        self.assertLess(Timestep(1982, 1, 1), Timestep(1982, 1, 2))
        self.assertLess(Timestep(1982, 12, 2), Timestep(1983, 1, 1))


class TestQC(unittest.TestCase):
    def test_consolidated_avhrr_era(self):
        # AVHRR-era codes end in 9 (MODIS digit N/A). 109 = RF+good; 2xx-5xx = modelled; 529 = snow.
        qc = np.array([[109, 209, 309, 409, 509, 519, 529, FILL_VALUE]], dtype=np.uint16)
        tier = classify(qc, "consolidated")
        self.assertEqual(tier[0, 0], GOOD)
        for i in range(1, 6):
            self.assertEqual(tier[0, i], INTERPOLATED)
        self.assertEqual(tier[0, 6], SNOW_CLOUD)
        self.assertEqual(tier[0, 7], NODATA)

    def test_consolidated_modis_era(self):
        # MODIS-era codes start with 99 (AVHRR digits N/A); 3rd digit = MODIS QC 0..4.
        qc = np.array([[990, 991, 992, 993, 994]], dtype=np.uint16)
        tier = classify(qc, "consolidated")
        self.assertEqual(list(tier[0]), [GOOD, INTERPOLATED, SNOW_CLOUD, SNOW_CLOUD, INTERPOLATED])

    def test_avhrr_only(self):
        qc = np.array([[0, 1, 2, FILL_VALUE]], dtype=np.uint16)
        tier = classify(qc, "avhrr")
        self.assertEqual(list(tier[0]), [GOOD, INTERPOLATED, SNOW_CLOUD, NODATA])


class TestZenodo(unittest.TestCase):
    def test_decade_selection(self):
        self.assertEqual(
            zenodo.needed_zip_names(1985, 1989, "consolidated"),
            ["PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_1982_1990.zip"],
        )
        self.assertEqual(
            zenodo.needed_zip_names(2015, 2019, "consolidated"),
            ["PKU_GIMMS_NDVI_AVHRR_MODIS_consolidated_2011_2022.zip"],
        )
        self.assertEqual(len(zenodo.needed_zip_names(1985, 2019, "consolidated")), 4)

    def test_checksums_consistent(self):
        for _, _, name in zenodo.CONSOLIDATED_DECADES + zenodo.AVHRR_DECADES:
            self.assertIn(name, zenodo.CHECKSUMS)


class TestDelta(unittest.TestCase):
    """Synthetic 2-year, 1x3 stack: pixel0 = ocean, pixel1 = stable land, pixel2 = barren -> green."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "s.zarr")
        root = zarr.open_group(self.path, mode="w")
        H, W = 1, 3
        for name, dt in {
            "ndvi_sum": "uint16", "valid_count": "uint8",
            "qc_good": "uint8", "qc_interp": "uint8", "qc_snow": "uint8",
        }.items():
            root.create_array(name, shape=(2, H, W), chunks=(1, H, W), dtype=dt)

        a = lambda *vals: np.array([list(vals)])  # 1x3 row helper
        root["ndvi_sum"][0] = a(0, 12000, 0)      # year A: ocean, stable(0.5), barren
        root["ndvi_sum"][1] = a(0, 12000, 6000)   # year B: ocean, stable(0.5), greened
        root["valid_count"][0] = a(0, 24, 0)
        root["valid_count"][1] = a(0, 24, 12)
        root["qc_good"][0] = a(0, 24, 0)
        root["qc_good"][1] = a(0, 12, 12)
        root["qc_interp"][0] = a(0, 0, 0)
        root["qc_interp"][1] = a(0, 12, 0)
        root["qc_snow"][0] = a(0, 0, 0)
        root["qc_snow"][1] = a(0, 0, 0)
        root.attrs.update({
            "version": "consolidated", "years": [2000, 2010], "n_rows": H, "n_cols": W,
            "west": -180.0, "north": 90.0, "pixel_deg": 1 / 12, "scale_factor": 0.001,
            "half_months_per_year": 24, "fill_value": FILL_VALUE, "built_year_indices": [0, 1],
        })

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_zero_fill_captures_transition_and_masks_ocean(self):
        res = compute_delta(self.path, (2000, 2000), (2010, 2010), fill_mode="zero")
        d = res.delta[0]
        self.assertTrue(np.isnan(d[0]))               # ocean -> masked (union land test)
        self.assertAlmostEqual(d[1], 0.0, places=5)    # stable land
        self.assertAlmostEqual(d[2], 0.25, places=5)   # barren -> green captured

    def test_drop_fill_hides_transition(self):
        res = compute_delta(self.path, (2000, 2000), (2010, 2010), fill_mode="drop")
        self.assertTrue(np.isnan(res.delta[0, 2]))     # no obs in A -> not comparable in drop mode

    def test_reliability(self):
        res = compute_delta(self.path, (2000, 2000), (2010, 2010))
        # stable pixel: good=24+12=36, interp=12, snow=0 -> 36/48 = 0.75
        self.assertAlmostEqual(res.reliability[0, 1], 0.75, places=5)
        self.assertTrue(np.isnan(res.reliability[0, 0]))  # ocean

    def test_summary_stats(self):
        res = compute_delta(self.path, (2000, 2000), (2010, 2010))
        stats = summary_stats(res)
        self.assertEqual(stats["valid_pixels"], 2)     # stable + greened (ocean masked)
        self.assertGreater(stats["mean_delta"], 0)      # net greening


class TestLandcover(unittest.TestCase):
    def test_cropland_mask(self):
        lccs = np.array([[10, 11, 12, 20, 30, 40, 130, 210]], dtype=np.uint8)
        self.assertEqual(landcover.cropland_mask(lccs).tolist(),
                         [[True, True, True, True, True, False, False, False]])

    def test_aggregate_fraction(self):
        lccs = np.full((60, 60), 200, np.uint8)  # all bare
        lccs[0:30, 0:30] = 10                     # block(0,0): all cropland -> 1.0
        lccs[30:45, 30:60] = 20                   # block(1,1): top half     -> 0.5
        frac = landcover.aggregate_fraction(landcover.cropland_mask(lccs))
        self.assertEqual(frac.shape, (2, 2))
        self.assertAlmostEqual(frac[0, 0], 1.0)
        self.assertAlmostEqual(frac[0, 1], 0.0)
        self.assertAlmostEqual(frac[1, 1], 0.5)

    def test_aggregate_fraction_rejects_misaligned(self):
        with self.assertRaises(ValueError):
            landcover.aggregate_fraction(np.zeros((31, 30), dtype=bool))

    def test_place_uses_bbox_origin(self):
        g = np.zeros((landcover.N_ROWS, landcover.N_COLS), np.float32)
        landcover._place(g, np.ones((540, 540), np.float32), [0.0, 45.0, 45.0, 90.0])  # N45E000
        self.assertEqual(g[0, 2160], 1.0)      # top edge +90°, lon 0° -> row 0, col 2160
        self.assertEqual(g[539, 2699], 1.0)
        self.assertEqual(g[540, 2160], 0.0)
        landcover._place(g, np.full((540, 540), 0.5, np.float32), [-180.0, -90.0, -135.0, -45.0])
        self.assertAlmostEqual(g[1620, 0], 0.5)  # S90W180 tile -> rows 1620.., col 0


class TestRenderCropland(unittest.TestCase):
    def test_cropland_to_rgba(self):
        rgba = render.cropland_to_rgba(np.array([[np.nan, 1.0, 2.0]], dtype=np.float32))
        self.assertEqual(rgba.shape, (1, 3, 4))
        self.assertEqual(rgba[0, 0, 3], 0)      # none -> transparent
        self.assertEqual(rgba[0, 1, 3], 110)    # one period -> half blue
        self.assertEqual(rgba[0, 2, 3], 210)    # both periods -> solid blue
        self.assertEqual(tuple(rgba[0, 2, :3]), render.CROPLAND_BLUE)


class TestCroplandDelta(unittest.TestCase):
    """1x4 stack: ocean | cropland-both | cropland-new(B only) | non-cropland."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "s.zarr")
        root = zarr.open_group(self.path, mode="w")
        H, W = 1, 4
        for name, dt in {
            "ndvi_sum": "uint16", "valid_count": "uint8",
            "qc_good": "uint8", "qc_interp": "uint8", "qc_snow": "uint8", CROPLAND_VAR: "uint8",
        }.items():
            root.create_array(name, shape=(2, H, W), chunks=(1, H, W), dtype=dt)
        a = lambda *v: np.array([list(v)])
        root["ndvi_sum"][0] = a(0, 12000, 12000, 12000)   # A means: -, 0.5, 0.5, 0.5
        root["ndvi_sum"][1] = a(0, 12000, 18000, 6000)    # B means: -, 0.5, 0.75, 0.25
        root["valid_count"][0] = a(0, 24, 24, 24)
        root["valid_count"][1] = a(0, 24, 24, 24)
        root["qc_good"][0] = a(0, 24, 24, 24)
        root["qc_good"][1] = a(0, 24, 24, 24)
        for v in ("qc_interp", "qc_snow"):
            root[v][0] = a(0, 0, 0, 0)
            root[v][1] = a(0, 0, 0, 0)
        root[CROPLAND_VAR][0] = a(0, 100, 0, 0)            # A cropland: -, yes, no,  no
        root[CROPLAND_VAR][1] = a(0, 100, 100, 0)          # B cropland: -, yes, yes, no
        root.attrs.update({
            "version": "consolidated", "years": [2000, 2010], "n_rows": H, "n_cols": W,
            "west": -180.0, "north": 90.0, "pixel_deg": 1 / 12, "scale_factor": 0.001,
            "half_months_per_year": 24, "fill_value": FILL_VALUE,
            "built_year_indices": [0, 1], "built_cropland_indices": [0, 1],
        })

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_classification(self):
        res = compute_delta(self.path, (2000, 2000), (2010, 2010), mask_sparse=False)
        self.assertEqual(res.cropland_both.tolist(), [[False, True, False, False]])
        self.assertEqual(res.cropland_one.tolist(), [[False, False, True, False]])  # new in B
        self.assertEqual(res.cropland_any.tolist(), [[False, True, True, False]])

    def test_split_stats(self):
        res = compute_delta(self.path, (2000, 2000), (2010, 2010), mask_sparse=False)
        farm = summary_stats(res, region=res.cropland_any)    # pixel1 (Δ0) + pixel2 (Δ+0.25)
        rest = summary_stats(res, region=~res.cropland_any)   # pixel3 (Δ-0.25); ocean excluded
        self.assertEqual(farm["valid_pixels"], 2)
        self.assertAlmostEqual(farm["greening_intensity_share"], 100.0)
        self.assertEqual(rest["valid_pixels"], 1)
        self.assertAlmostEqual(rest["browning_intensity_share"], 100.0)

    def test_absent_when_period_not_cropland_built(self):
        root = zarr.open_group(self.path, mode="a")
        root.attrs["built_cropland_indices"] = [0]   # 2010 not cropland-built -> can't classify
        res = compute_delta(self.path, (2000, 2000), (2010, 2010), mask_sparse=False)
        self.assertIsNone(res.cropland_any)


if __name__ == "__main__":
    unittest.main()
