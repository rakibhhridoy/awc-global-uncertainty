"""Paper 1 / M2 — build the stratified soil->water training table.

Reads reprojected COGs under {DATA_ROOT}/interim/reproj/<source>/... (produced by
src/reproject_paper1.py on the EPSG:6933 1 km grid) and assembles a
per-(pixel, depth) table joining predictors to the VWC hydraulic targets.

Targets: OpenLandMap/Zenodo-13837179 Global VWC at 33 kPa (field capacity),
1500 kPa (wilting point); AWC = VWC33 - VWC1500. Target depths: 0/30/60/100 cm.

Dev default = STRATIFIED SAMPLE (N_SAMPLE per region) for M1/M2 work on the Mac;
full-grid inference is a separate pass (M6). Untested until M1 reproj exists.

Run:  python Paper1/src/build_table.py [--n 200000]
Out:  Paper1/data/interim/train_table.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# self-contained grid + paths (vendored)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import INTERIM, DATA_ROOT  # noqa: E402  -> {DATA_ROOT}/interim

REPROJ = INTERIM / "reproj"
OUT = DATA_ROOT / "paper1_hydraulic/interim/train_table.parquet"
NODATA = -9999.0

# target depths (cm) and the SoilGrids interval whose support best matches each.
# SoilGrids intervals: 0-5,5-15,15-30,30-60,60-100,100-200.
DEPTH_MAP = {
    0:   "0-5cm",
    30:  "15-30cm",
    60:  "30-60cm",
    100: "60-100cm",
}
SOILGRIDS_VARS = ["clay", "sand", "silt", "bdod", "soc", "phh2o", "cec", "nitrogen", "cfvo"]

# static (depth-invariant) predictors: (source, glob relative to REPROJ/source)
STATIC_PREDICTORS = {
    "worldclim":       "**/*.tif",
    "chelsa":          "**/*.tif",
    "merit_hydro":     "elv/**/*.tif",   # elevation; slope/curvature derived later
    "copernicus_lulc": "**/*.tif",       # categorical -> treat as code
}


def continent(lat, lon):
    if -170 < lon < -30 and lat > 7:  return "N.America"
    if -90 < lon < -30 and lat <= 7:  return "S.America"
    if -20 < lon < 60 and -40 < lat < 37: return "Africa"
    if -15 < lon < 45 and lat >= 37:  return "Europe"
    if 45 <= lon < 180 and lat >= 5:  return "Asia"
    if 110 < lon < 180 and lat < 5:   return "Oceania"
    return "other"
# TODO(region): swap for a Köppen raster (plan §9 picked Köppen for spatial CV).


def vwc_path(suction: str, depth_cm: int) -> Path:
    return REPROJ / "openlandmap" / "zenodo_13837179" / f"Global_VWC_{suction}_{depth_cm}cm_V1.0.tif"


def soilgrids_path(var: str, depth_label: str) -> Path:
    return REPROJ / "soilgrids_isric" / var / f"{var}_{depth_label}_mean_1000.tif"


def read_at(path: Path, rows, cols) -> np.ndarray:
    """Read a single-band raster fully and gather values at (rows, cols)."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        nod = ds.nodata
    v = arr[rows, cols].astype("float32")
    if nod is not None:
        v[v == nod] = np.nan
    v[v == NODATA] = np.nan
    return v


def pixel_lonlat(ds, rows, cols):
    xs, ys = rasterio.transform.xy(ds.transform, rows, cols, offset="center")
    # grid is EPSG:6933 (metres) -> convert to lon/lat for region assignment
    from rasterio.warp import transform as warp_xy
    lon, lat = warp_xy(ds.crs, "EPSG:4326", np.asarray(xs), np.asarray(ys))
    return np.asarray(lon), np.asarray(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="total stratified sample size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # 1) valid-pixel mask from a reference target (33 kPa @ 0 cm)
    ref = vwc_path("33kPa", 0)
    if not ref.exists():
        sys.exit(f"missing reprojected target {ref} — run reproject_static.py first (M1)")
    with rasterio.open(ref) as ds:
        ref_arr = ds.read(1)
        ref_nod = ds.nodata
        transform, crs = ds.transform, ds.crs
    valid = np.isfinite(ref_arr) & (ref_arr != (ref_nod if ref_nod is not None else NODATA))
    vr, vc = np.where(valid)
    print(f"valid target pixels: {len(vr):,}")

    # 2) region label per valid pixel, then stratified sample
    lon, lat = pixel_lonlat(rasterio.open(ref), vr, vc)
    region = np.array([continent(a, o) for a, o in zip(lat, lon)])
    keep = region != "other"
    vr, vc, lon, lat, region = vr[keep], vc[keep], lon[keep], lat[keep], region[keep]

    per = max(1, args.n // len(set(region)))
    idx = []
    for rg in sorted(set(region)):
        cand = np.where(region == rg)[0]
        idx.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
    idx = np.concatenate(idx)
    rows, cols = vr[idx], vc[idx]
    print(f"sampled {len(idx):,} pixels across {len(set(region))} regions")

    df = pd.DataFrame({"row": rows, "col": cols,
                       "lon": lon[idx], "lat": lat[idx], "region": region[idx]})

    # 3) per-depth long table: targets + depth-matched SoilGrids + static predictors
    frames = []
    static_cache = {}  # static predictors read once, gathered per sample
    for src, pat in STATIC_PREDICTORS.items():
        for f in sorted((REPROJ / src).glob(pat)):
            if f.name.startswith("._"):
                continue  # skip macOS AppleDouble metadata files
            static_cache[f"{src}__{f.stem}"] = read_at(f, rows, cols)

    for depth_cm, sg_label in DEPTH_MAP.items():
        d = df.copy()
        d["depth_cm"] = depth_cm
        # targets
        d["vwc_33kPa"]   = read_at(vwc_path("33kPa", depth_cm), rows, cols)
        d["vwc_1500kPa"] = read_at(vwc_path("1500kPa", depth_cm), rows, cols)
        d["awc"] = d["vwc_33kPa"] - d["vwc_1500kPa"]
        # depth-matched soil predictors
        for var in SOILGRIDS_VARS:
            p = soilgrids_path(var, sg_label)
            d[f"sg_{var}"] = read_at(p, rows, cols) if p.exists() else np.nan
        # static predictors (broadcast across depths)
        for k, v in static_cache.items():
            d[k] = v
        frames.append(d)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["vwc_33kPa", "vwc_1500kPa"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out):,} rows x {out.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()
