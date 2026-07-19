"""Paper 1 / M4b — validate predicted hydraulics against MEASURED WoSIS retention.

The non-circular headline. For field capacity / wilting point / AWC, compare three
predictors of the MEASURED value (held-out):
  1. PTF product  — the existing OpenLandMap VWC sampled at the measured point (no training)
  2. Ridge        — linear model on covariates (our baseline)
  3. GBT          — gradient-boosted trees on covariates (our model)

Skill is R^2 / RMSE against measured, under random 5-fold and leave-one-continent-out
(LOCO, honest spatial extrapolation). Question: does our model beat the existing global
PTF product at reproducing lab measurements?

Out: paper1_hydraulic/eval/m4b_measured_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa: E402

TABLE = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m4b_measured_validation.json"
SEED = 0
TARGETS = {"meas_fc": "ptf_fc", "meas_wp": "ptf_wp", "meas_awc": "ptf_awc"}


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))

def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=400, max_depth=8,
                                                       learning_rate=0.06, random_state=SEED))

def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    df = pd.read_parquet(TABLE)
    feats = [c for c in df.columns
             if c.startswith(("sg_", "chelsa__", "copernicus")) and c != "sg_label"
             and pd.api.types.is_numeric_dtype(df[c])] + ["depth_cm"]
    print(f"measured table {df.shape}, {len(feats)} covariates")
    out = {"n": int(len(df)), "n_sites": int(df.profile_id.nunique()),
           "n_features": len(feats), "results": []}

    for tgt, ptf in TARGETS.items():
        d = df[np.isfinite(df[tgt])].copy()
        y = d[tgt].values
        r = {"target": tgt, "n": int(len(d))}

        # 1) existing PTF product vs measured (benchmark; mask where PTF present)
        pm = np.isfinite(d[ptf].values)
        r["ptf_R2_vs_measured"] = float(r2_score(y[pm], d[ptf].values[pm]))
        r["ptf_RMSE"] = rmse(y[pm], d[ptf].values[pm])

        # 2/3) our models, random 5-fold
        for name, mk in [("ridge", ridge), ("gbt", gbt)]:
            r2s, rm = [], []
            for tr, te in KFold(5, shuffle=True, random_state=SEED).split(d):
                m = mk().fit(d.iloc[tr][feats], y[tr])
                p = m.predict(d.iloc[te][feats]); r2s.append(r2_score(y[te], p)); rm.append(rmse(y[te], p))
            r[f"{name}_random_R2"] = float(np.mean(r2s)); r[f"{name}_random_RMSE"] = float(np.mean(rm))

        # leave-one-continent-out (honest spatial)
        for name, mk in [("ridge", ridge), ("gbt", gbt)]:
            r2s, per = [], {}
            for c in sorted(d.continent.unique()):
                tr, te = d.continent != c, d.continent == c
                if te.sum() < 50:
                    continue
                m = mk().fit(d[tr][feats], y[tr.values])
                p = m.predict(d[te][feats]); sc = r2_score(y[te.values], p)
                r2s.append(sc); per[c] = round(float(sc), 4)
            r[f"{name}_loco_R2"] = float(np.mean(r2s)); r[f"{name}_loco_by_continent"] = per

        out["results"].append(r)
        print(f"-- {tgt} (n={r['n']})")
        print(f"   PTF product vs measured : R2={r['ptf_R2_vs_measured']:+.3f}  RMSE={r['ptf_RMSE']:.3f}")
        print(f"   ours random  ridge={r['ridge_random_R2']:.3f} gbt={r['gbt_random_R2']:.3f}")
        print(f"   ours LOCO    ridge={r['ridge_loco_R2']:.3f} gbt={r['gbt_loco_R2']:.3f}")
        print(f"   -> GBT beats PTF by R2 {r['gbt_random_R2']-r['ptf_R2_vs_measured']:+.3f} (random),"
              f" {r['gbt_loco_R2']-r['ptf_R2_vs_measured']:+.3f} (spatial)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
