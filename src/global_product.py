"""Paper 1 / M7 — global hydraulic product with reliability mask + agricultural significance.

Turns the benchmark into a usable, honest deliverable and quantifies its downstream
consequence. Trains final models on ALL measured WoSIS retention, predicts available
water capacity (AWC) and its 90% prediction interval over a global grid sample, flags
the reliable domain (area of applicability), and asks the "so what":

  How much of global CROPLAND has reliable AWC estimates, and how large is the AWC
  uncertainty there in plant-available-water (mm) terms?

This speaks directly to the crop-modelling, irrigation and land-surface communities
that consume global hydraulic products.

Out: paper1_hydraulic/eval/m7_global_product.json
     paper1_hydraulic/interim/global_awc_grid.parquet   (for maps)
     figures/data/{global_awc.csv, significance.csv}
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa

MEAS = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"
GRID = DATA_ROOT / "paper1_hydraulic/interim/train_table.parquet"
OUTJSON = DATA_ROOT / "paper1_hydraulic/eval/m7_global_product.json"
GRIDOUT = DATA_ROOT / "paper1_hydraulic/interim/global_awc_grid.parquet"
FIGDIR = Path(__file__).resolve().parents[1] / "figures/data"
SEED = 0
CROPLAND = 40           # Copernicus discrete class = cultivated/managed (cropland)
ROOTZONE_MM = 1000.0    # integrate AWC over a 1 m root zone -> plant-available water (mm)


def gbt(**kw):
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=400, max_depth=8,
                         learning_rate=0.06, random_state=SEED, **kw))


def main():
    m = pd.read_parquet(MEAS)
    g = pd.read_parquet(GRID)
    feats = [c for c in m.columns if c.startswith("sg_") and c != "sg_label"] + \
            sorted(c for c in m.columns if c.startswith("chelsa__CHELSA_bio")) + ["depth_cm"]
    feats = [c for c in feats if c in g.columns]
    print(f"features: {len(feats)} | train {len(m):,} measured | grid {len(g):,}")

    d = m[np.isfinite(m.meas_awc)].copy()
    y = d.meas_awc.values

    # final models: point + 90% interval
    point = gbt().fit(d[feats], y)
    q05 = gbt(loss="quantile", quantile=0.05).fit(d[feats], y)
    q95 = gbt(loss="quantile", quantile=0.95).fit(d[feats], y)

    g = g.copy()
    g["awc_pred"] = point.predict(g[feats])
    g["awc_lo"] = q05.predict(g[feats])
    g["awc_hi"] = q95.predict(g[feats])
    g["awc_width"] = (g.awc_hi - g.awc_lo).clip(lower=0)

    # --- reliability (area of applicability) ---
    sc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(d[feats])
    Xtr = sc.transform(d[feats])
    smp = Xtr[np.random.default_rng(SEED).choice(len(Xtr), size=2000, replace=False)]
    dbar = float(np.mean(np.linalg.norm(smp[:, None] - smp[None, :], axis=2)))
    # de-duplicated training NN distance (one row per site+depth) for a realistic threshold
    dd = d.assign(_k=d.profile_id.astype(str) + "_" + d.depth_cm.astype(str)).drop_duplicates("_k")
    Xdd = sc.transform(dd[feats])
    nn = cKDTree(Xdd).query(Xdd, k=2)[0][:, 1] / dbar
    q1, q3 = np.quantile(nn, [0.25, 0.75]); thr = float(q3 + 1.5 * (q3 - q1))
    g["di"] = cKDTree(Xtr).query(sc.transform(g[feats]), k=1)[0] / dbar
    g["reliable"] = (g.di <= thr).astype(int)

    # --- root-zone plant-available water (mm), per location (mean AWC over depths) ---
    loc = g.groupby(["row", "col"]).agg(
        lon=("lon", "first"), lat=("lat", "first"), lulc=("lulc_2018", "first"),
        awc=("awc_pred", "mean"), width=("awc_width", "mean"),
        reliable=("reliable", "min")).reset_index()       # reliable only if all depths reliable
    loc["paws_mm"] = loc.awc * ROOTZONE_MM                 # plant-available water storage
    loc["paws_pi_mm"] = loc.width * ROOTZONE_MM            # 90% interval width in mm
    crop = loc[loc.lulc == CROPLAND]

    def frac(mask): return float(np.mean(mask))
    sig = {
        "n_grid_locations": int(len(loc)),
        "n_cropland_locations": int(len(crop)),
        "reliable_frac_all_land": round(frac(loc.reliable == 1), 4),
        "reliable_frac_cropland": round(frac(crop.reliable == 1), 4),
        "cropland_OUTSIDE_reliable_pct": round(100 * frac(crop.reliable == 0), 1),
        "paws_mm_median_cropland": round(float(crop.paws_mm.median()), 1),
        "paws_pi_mm_median_cropland": round(float(crop.paws_pi_mm.median()), 1),
        "paws_pi_as_frac_of_estimate_cropland": round(float((crop.paws_pi_mm / crop.paws_mm).median()), 3),
        "aoa_threshold": round(thr, 4),
    }
    print(json.dumps(sig, indent=1))

    OUTJSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(sig, open(OUTJSON, "w"), indent=1)
    GRIDOUT.parent.mkdir(parents=True, exist_ok=True)
    g[["lon", "lat", "depth_cm", "awc_pred", "awc_width", "di", "reliable", "lulc_2018"]].to_parquet(GRIDOUT, index=False)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    # surface map data (one row per location): awc, reliability, paws
    loc[["lon", "lat", "awc", "paws_mm", "paws_pi_mm", "reliable", "lulc"]].round(4).to_csv(FIGDIR / "global_awc.csv", index=False)
    pd.DataFrame([sig]).to_csv(FIGDIR / "significance.csv", index=False)
    print(f"-> {OUTJSON}\n-> {GRIDOUT}\n-> {FIGDIR/'global_awc.csv'}")


if __name__ == "__main__":
    main()
