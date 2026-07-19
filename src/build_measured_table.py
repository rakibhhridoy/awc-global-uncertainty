"""Paper 1 / M4b — build the MEASURED hydraulic validation table (breaks circularity).

Targets become independent lab measurements from WoSIS (water retention at 33 kPa =
field capacity, 1500 kPa = wilting point; AWC = FC - WP), NOT the OpenLandMap PTF
product. At each measured layer we sample:
  - our predictors (SoilGrids depth-matched + CHELSA bioclim + LULC), and
  - the OpenLandMap VWC product (the existing global PTF) as the benchmark to beat.

This lets M4b ask the non-circular question: can a model trained on global covariates
predict MEASURED retention, and does it beat the existing PTF product against the same
measurements?

Out: paper1_hydraulic/interim/measured_table.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_xy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import INTERIM, DATA_ROOT  # noqa: E402

WOSIS = Path("/Volumes/SSD Ex/PEDOFLUX_data/raw/wosis_isric/extracted/WoSIS_2023_December")
REPROJ = INTERIM / "reproj"
OUT = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"

# WoSIS layer midpoint (cm) -> our target depth + matching SoilGrids interval
DEPTH_BINS = [(0, 15, 0, "0-5cm"), (15, 45, 30, "15-30cm"),
              (45, 80, 60, "30-60cm"), (80, 150, 100, "60-100cm")]
SOILGRIDS_VARS = ["clay", "sand", "silt", "bdod", "soc", "phh2o", "cec", "nitrogen", "cfvo"]
STATIC = {  # source dir under reproj -> glob
    "chelsa": "CHELSA_bio*_1981-2010_V.2.1.tif",
    "copernicus_lulc": "*2018*.tif",
}


def load_measured():
    cols = ["profile_id", "layer_id", "upper_depth", "lower_depth",
            "value_avg", "longitude", "latitude", "continent"]
    fc = pd.read_csv(WOSIS / "wosis_202312_wv0033.tsv", sep="\t", usecols=cols)
    wp = pd.read_csv(WOSIS / "wosis_202312_wv1500.tsv", sep="\t",
                     usecols=["profile_id", "layer_id", "value_avg"])
    m = fc.merge(wp, on=["profile_id", "layer_id"], suffixes=("_fc", "_wp"))
    m = m.dropna(subset=["longitude", "latitude"])
    m["meas_fc"] = m.value_avg_fc / 100.0          # %v/v -> m3/m3
    m["meas_wp"] = m.value_avg_wp / 100.0
    m["meas_awc"] = m.meas_fc - m.meas_wp
    m["mid"] = (m.upper_depth + m.lower_depth) / 2
    # physical sanity
    m = m[(m.meas_fc > 0.01) & (m.meas_fc < 0.85) &
          (m.meas_wp > 0) & (m.meas_awc > 0) & (m.meas_awc < 0.7)]
    # bin to target depth
    def binit(mid):
        for lo, hi, d, lbl in DEPTH_BINS:
            if lo <= mid < hi:
                return d, lbl
        return np.nan, None
    dd = m["mid"].apply(binit)
    m["depth_cm"] = [x[0] for x in dd]; m["sg_label"] = [x[1] for x in dd]
    m = m.dropna(subset=["depth_cm"]).copy()
    m["depth_cm"] = m["depth_cm"].astype(int)
    return m


def sample_points(path, lon, lat):
    """sample a reproj raster (EPSG:6933) at lon/lat point arrays."""
    with rasterio.open(path) as ds:
        xs, ys = warp_xy("EPSG:4326", ds.crs, list(lon), list(lat))
        vals = np.array([v[0] for v in ds.sample(list(zip(xs, ys)))], dtype="float32")
        nod = ds.nodata
    if nod is not None:
        vals[vals == nod] = np.nan
    vals[vals <= -9000] = np.nan
    return vals


def main():
    m = load_measured()
    print(f"measured layers (clean, binned): {len(m):,} at {m.profile_id.nunique():,} sites")
    lon, lat = m.longitude.values, m.latitude.values

    # depth-matched SoilGrids predictors
    for var in SOILGRIDS_VARS:
        col = np.full(len(m), np.nan, dtype="float32")
        for d, lbl in m[["depth_cm", "sg_label"]].drop_duplicates().itertuples(index=False):
            p = REPROJ / "soilgrids_isric" / var / f"{var}_{lbl}_mean_1000.tif"
            if not p.exists():
                continue
            mask = m.sg_label.values == lbl
            col[mask] = sample_points(p, lon[mask], lat[mask])
        m[f"sg_{var}"] = col

    # depth-matched OpenLandMap VWC product (benchmark) -> ptf_fc / ptf_wp / ptf_awc
    vwc_depth = {0: 0, 30: 30, 60: 60, 100: 100}
    for suction, name in [("33kPa", "ptf_fc"), ("1500kPa", "ptf_wp")]:
        col = np.full(len(m), np.nan, dtype="float32")
        for d in sorted(m.depth_cm.unique()):
            p = REPROJ / "openlandmap" / "zenodo_13837179" / f"Global_VWC_{suction}_{vwc_depth[d]}cm_V1.0.tif"
            if not p.exists():
                continue
            mask = m.depth_cm.values == d
            col[mask] = sample_points(p, lon[mask], lat[mask])
        m[name] = col
    m["ptf_awc"] = m.ptf_fc - m.ptf_wp

    # static predictors (depth-invariant)
    for src, pat in STATIC.items():
        for f in sorted((REPROJ / src).glob(pat)):
            if f.name.startswith("._"):
                continue
            m[f"{src}__{f.stem}"] = sample_points(f, lon, lat)

    keep = ["profile_id", "longitude", "latitude", "continent", "depth_cm",
            "meas_fc", "meas_wp", "meas_awc", "ptf_fc", "ptf_wp", "ptf_awc"] + \
           [c for c in m.columns if c.startswith(("sg_", "chelsa__", "copernicus"))]
    out = m[keep].copy()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {len(out):,} rows x {out.shape[1]} cols -> {OUT}")
    print("  per-continent:", out.continent.value_counts().to_dict())
    print(f"  PTF-vs-measured FC corr: {np.corrcoef(out.meas_fc, out.ptf_fc.fillna(out.ptf_fc.median()))[0,1]:.3f}")


if __name__ == "__main__":
    main()
