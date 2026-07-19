"""Paper 1 / M4 — confront predicted soil hydraulics against observed ESA-CCI SM.

This is the study's independent-validation step. ESA-CCI surface soil moisture
(2010-2014 climatology) is observed and independent of both our predictors
(SoilGrids/CHELSA) and our hydraulic targets (OpenLandMap VWC) — so it breaks the
PTF circularity that limits the M3-only story.

Three confrontations, all at the SURFACE (0 cm) layer:

 1. PHYSICAL CONSISTENCY — does observed mean SM fall within the plant-available
    envelope [wilting point, field capacity]? Well-behaved soils should.

 2. COUPLING TEST (headline) — predict observed mean SM from:
       (a) climate only            (CHELSA bioclim)
       (b) climate + hydraulics    (+ vwc_33kPa, vwc_1500kPa, awc)
    The R^2 gain (b-a) = how much soil hydraulic composition explains REAL moisture
    beyond climate. Computed globally and per region (leave-one-region-out for honesty).

 3. WHERE coupling is strong — per-region gain table -> seeds the global coupling map (M6).

Out: paper1_hydraulic/eval/m4_confrontation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import INTERIM, DATA_ROOT  # noqa: E402

TABLE = DATA_ROOT / "paper1_hydraulic/interim/train_table.parquet"
ESA = INTERIM / "reproj" / "esacci"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m4_confrontation.json"
SEED = 0
MIN_OBS = 50          # require >=50 valid daily ESA-CCI obs over 2010-2014 per pixel


def sample_raster(path, rows, cols):
    with rasterio.open(path) as ds:
        a = ds.read(1); nod = ds.nodata
    v = a[rows, cols].astype("float32")
    if nod is not None:
        v[v == nod] = np.nan
    v[v <= -9000] = np.nan
    return v


def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=300, max_depth=8,
                                                       learning_rate=0.08, random_state=SEED))


def loro_r2(df, feats, tgt):
    """leave-one-region-out mean R^2 predicting tgt from feats."""
    r2s = {}
    for rg in sorted(df.region.unique()):
        tr, te = df.region != rg, df.region == rg
        m = gbt().fit(df.loc[tr, feats], df.loc[tr, tgt])
        r2s[rg] = float(r2_score(df.loc[te, tgt], m.predict(df.loc[te, feats])))
    return r2s


def main():
    df = pd.read_parquet(TABLE)
    df = df[df.depth_cm == 0].copy()                  # surface, to match satellite SM
    print(f"surface rows: {len(df):,}")

    # attach observed ESA-CCI climatology at each pixel
    rows, cols = df.row.values, df.col.values
    df["sm_obs_mean"] = sample_raster(ESA / "esacci_sm_mean.tif", rows, cols)
    df["sm_obs_std"]  = sample_raster(ESA / "esacci_sm_std.tif",  rows, cols)
    df["sm_obs_n"]    = sample_raster(ESA / "esacci_sm_n.tif",    rows, cols)
    df = df[(df.sm_obs_n >= MIN_OBS) & np.isfinite(df.sm_obs_mean)].copy()
    print(f"pixels with >= {MIN_OBS} ESA-CCI obs: {len(df):,} ({df.region.nunique()} regions)")

    out = {"n_surface_pixels": int(len(df)), "min_obs": MIN_OBS}

    # 1) PHYSICAL CONSISTENCY -------------------------------------------------
    fc, wp, sm = df.vwc_33kPa, df.vwc_1500kPa, df.sm_obs_mean
    within = ((sm >= wp) & (sm <= fc)).mean()
    above_fc = (sm > fc).mean(); below_wp = (sm < wp).mean()
    out["physical"] = {
        "frac_within_wilting_fieldcap": float(within),
        "frac_above_fieldcap": float(above_fc),
        "frac_below_wilting": float(below_wp),
        "corr_obsSM_fieldcap": float(np.corrcoef(sm, fc)[0, 1]),
        "corr_obsSM_AWC": float(np.corrcoef(sm, df.awc)[0, 1]),
    }
    print(f"1) physical: {within:.1%} of pixels have observed SM within [wilting,fieldcap];"
          f" corr(SM,FC)={out['physical']['corr_obsSM_fieldcap']:.2f}")

    # 2) COUPLING TEST --------------------------------------------------------
    # Test hydraulic control over multiple moisture descriptors. The mean is
    # climate-dominated; the soil-buffering hypothesis predicts stronger control
    # over moisture VARIABILITY (std) and relative variability (CV = std/mean).
    df["sm_obs_cv"] = df.sm_obs_std / df.sm_obs_mean.replace(0, np.nan)
    clim = [c for c in df.columns if c.startswith("chelsa__")]
    hydr = ["vwc_33kPa", "vwc_1500kPa", "awc"]

    out["coupling"] = {}
    for target, label in [("sm_obs_mean", "mean SM"),
                          ("sm_obs_std", "SM variability"),
                          ("sm_obs_cv", "SM rel. variability (CV)")]:
        d = df[np.isfinite(df[target])].copy()
        r2_clim = loro_r2(d, clim, target)
        r2_both = loro_r2(d, clim + hydr, target)
        gain = {rg: r2_both[rg] - r2_clim[rg] for rg in r2_clim}
        out["coupling"][target] = {
            "loro_r2_climate_only": {k: round(v, 4) for k, v in r2_clim.items()},
            "loro_r2_climate_plus_hydraulics": {k: round(v, 4) for k, v in r2_both.items()},
            "hydraulic_gain_by_region": {k: round(v, 4) for k, v in gain.items()},
            "mean_climate_only_r2": float(np.mean(list(r2_clim.values()))),
            "mean_with_hydraulics_r2": float(np.mean(list(r2_both.values()))),
            "mean_hydraulic_gain": float(np.mean(list(gain.values()))),
        }
        c = out["coupling"][target]
        print(f"2) coupling [{label}]: climate R2={c['mean_climate_only_r2']:.3f}"
              f" -> +hydraulics {c['mean_with_hydraulics_r2']:.3f}"
              f"  (gain {c['mean_hydraulic_gain']:+.3f})")
        print("     by region:", {k: round(v, 3) for k, v in gain.items()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
