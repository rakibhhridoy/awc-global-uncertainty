"""Paper 1 / M3 — baseline vs gradient-boosted models under spatial CV.

Predicts soil hydraulic targets (vwc_33kPa field capacity, vwc_1500kPa wilting point,
awc available water capacity) from soil + climate + land-cover predictors.

Two validation regimes:
  - random K-fold  : optimistic (in-distribution) skill
  - leave-one-region-out (LORO) : honest spatial-extrapolation skill (the headline)

Models: Ridge (linear baseline / PTF-style) vs HistGradientBoosting (rich, nonlinear).
"Headroom" = GBT - Ridge under LORO = where nonlinearity earns transfer skill.

Out: Paper1/data/eval/model_cv_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa: E402

TABLE = str(DATA_ROOT / "paper1_hydraulic/interim/train_table.parquet")
OUT = DATA_ROOT / "paper1_hydraulic/eval/model_cv_results.json"
TARGETS = ["vwc_33kPa", "vwc_1500kPa", "awc"]
SEED = 0


def features(df):
    return [c for c in df.columns if c.startswith(("sg_", "chelsa__", "lulc")) or c == "depth_cm"]


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))


def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=300, max_depth=8,
                                                       learning_rate=0.08, random_state=SEED))


def evaluate(df, feats, tgt):
    res = {"target": tgt}
    X, y = df[feats], df[tgt].values

    # random 5-fold (optimistic)
    for name, mk in [("ridge", ridge), ("gbt", gbt)]:
        r2s = []
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
            m = mk().fit(X.iloc[tr], y[tr])
            r2s.append(r2_score(y[te], m.predict(X.iloc[te])))
        res[f"random_{name}_r2"] = float(np.mean(r2s))

    # leave-one-region-out (honest spatial)
    loro = {}
    for name, mk in [("ridge", ridge), ("gbt", gbt)]:
        r2s, maes = [], []
        for rg in sorted(df.region.unique()):
            tr = df.region != rg
            te = df.region == rg
            m = mk().fit(X[tr], y[tr])
            p = m.predict(X[te])
            r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
        loro[name] = {"r2_mean": float(np.mean(r2s)), "mae_mean": float(np.mean(maes)),
                      "r2_by_region": {rg: float(r) for rg, r in
                                       zip(sorted(df.region.unique()), r2s)}}
    res["loro_ridge"] = loro["ridge"]
    res["loro_gbt"] = loro["gbt"]
    res["loro_headroom"] = loro["gbt"]["r2_mean"] - loro["ridge"]["r2_mean"]
    return res


def main():
    t0 = time.time()
    df = pd.read_parquet(TABLE)
    feats = features(df)
    print(f"table {df.shape}, {len(feats)} features, targets {TARGETS}")
    out = {"n_rows": int(len(df)), "n_features": len(feats), "features": feats, "results": []}
    for tgt in TARGETS:
        print(f"-- {tgt} ...", flush=True)
        r = evaluate(df, feats, tgt)
        out["results"].append(r)
        print(f"   random  ridge={r['random_ridge_r2']:.3f} gbt={r['random_gbt_r2']:.3f}")
        print(f"   LORO    ridge={r['loro_ridge']['r2_mean']:.3f} gbt={r['loro_gbt']['r2_mean']:.3f}"
              f"  headroom={r['loro_headroom']:+.3f}")
    out["runtime_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
