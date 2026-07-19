"""Paper 1 / M5 — Area of Applicability (AoA) and calibrated uncertainty.

Turns the negative transferability result into a USABLE one: it quantifies WHERE
predictions are trustworthy rather than concluding the map is globally unreliable.

Method (after Meyer & Pebesma, 2021): in standardised predictor space the
dissimilarity index (DI) of a location is its nearest-training-point distance divided
by the mean pairwise training distance. The AoA is the region with DI below a
threshold from the training DI distribution (Q3 + 1.5 IQR).

We report:
  - skill INSIDE vs OUTSIDE the AoA under random 5-fold CV (shows the AoA flags
    reliable predictions);
  - the fraction of global land inside the AoA;
  - the fraction of leave-one-CONTINENT-out test points inside the AoA (≈0, which
    explains why spatial hold-out skill collapses);
  - 90% prediction-interval coverage (PICP) from quantile gradient boosting.

Out: paper1_hydraulic/eval/m5_aoa.json + figures/data/aoa.csv
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
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa

MEAS = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"
GRID = DATA_ROOT / "paper1_hydraulic/interim/train_table.parquet"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m5_aoa.json"
FIGDATA = Path(__file__).resolve().parents[1] / "figures/data/aoa.csv"
SEED = 0
BLOCK_DEG = 5.0      # spatial block size (deg) for spatial-block PICP (matches M6/spatial_cv)
BLOCK_K = 10         # spatial-block folds
TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}


def gbt(**kw):
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=350, max_depth=8,
                         learning_rate=0.06, random_state=SEED, **kw))


def main():
    df = pd.read_parquet(MEAS)
    grid = pd.read_parquet(GRID)
    # AoA features = soil + bioclim + depth, present in BOTH tables (drops bio18/19)
    feats = [c for c in df.columns if c.startswith("sg_") and c != "sg_label"] + \
            sorted(c for c in df.columns if c.startswith("chelsa__CHELSA_bio")) + ["depth_cm"]
    feats = [c for c in feats if c in grid.columns]
    print(f"AoA features: {len(feats)}")

    out = {"targets": []}
    fig_rows = []
    rng = np.random.default_rng(SEED)

    for tgt, label in TARGETS.items():
        d = df[np.isfinite(df[tgt])].copy().reset_index(drop=True)
        y = d[tgt].values
        X = d[feats]

        # global standardisation + DI reference
        sc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
        Xs = sc.transform(X)
        smp = Xs[rng.choice(len(Xs), size=2000, replace=False)]
        dbar = float(np.mean(np.linalg.norm(smp[:, None] - smp[None, :], axis=2)))
        dnn = cKDTree(Xs).query(Xs, k=2)[0][:, 1]
        di_train = dnn / dbar
        q1, q3 = np.quantile(di_train, [0.25, 0.75])
        thr = float(q3 + 1.5 * (q3 - q1))

        # random 5-fold: OOF prediction + DI (test vs train fold) + 90% interval
        pred = np.full(len(d), np.nan); di = np.full(len(d), np.nan)
        lo = np.full(len(d), np.nan); hi = np.full(len(d), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
            scf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X.iloc[tr])
            di[te] = cKDTree(scf.transform(X.iloc[tr])).query(scf.transform(X.iloc[te]), k=1)[0] / dbar
            pred[te] = gbt().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            lo[te] = gbt(loss="quantile", quantile=0.05).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            hi[te] = gbt(loss="quantile", quantile=0.95).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])

        inside = di <= thr
        r2_in = r2_score(y[inside], pred[inside]) if inside.sum() > 30 else None
        r2_out = r2_score(y[~inside], pred[~inside]) if (~inside).sum() > 30 else None
        picp = float(np.mean((y >= lo) & (y <= hi)))

        # PICP is meaningless if calibrated only on random folds (test points sit next to
        # their training neighbours): the same autocorrelation that inflates point skill
        # inflates interval coverage. We therefore also calibrate the 90% interval under
        # spatial-block CV and under leave-one-continent-out, the regimes that matter for a
        # GLOBAL product. Coverage that falls below the nominal 0.90 out of domain means the
        # released intervals are too narrow exactly where the map is least trustworthy.
        bx = np.floor(d.longitude / BLOCK_DEG).astype(int).astype(str)
        by = np.floor(d.latitude / BLOCK_DEG).astype(int).astype(str)
        block = (bx + "_" + by).values
        ublk = np.array(sorted(set(block))); rng.shuffle(ublk)
        fold_of = {b: i % BLOCK_K for i, b in enumerate(ublk)}
        bfold = np.array([fold_of[b] for b in block])
        lo_s = np.full(len(d), np.nan); hi_s = np.full(len(d), np.nan)
        for k in range(BLOCK_K):
            tr = bfold != k; te = bfold == k
            if te.sum() == 0 or tr.sum() < 50:
                continue
            lo_s[te] = gbt(loss="quantile", quantile=0.05).fit(X[tr], y[tr]).predict(X[te])
            hi_s[te] = gbt(loss="quantile", quantile=0.95).fit(X[tr], y[tr]).predict(X[te])
        ok = np.isfinite(lo_s)
        picp_spatial = float(np.mean((y[ok] >= lo_s[ok]) & (y[ok] <= hi_s[ok])))

        lo_l = np.full(len(d), np.nan); hi_l = np.full(len(d), np.nan)
        per_cont_cov = {}
        for c in sorted(d.continent.unique()):
            tr = (d.continent != c).values; te = (d.continent == c).values
            if te.sum() < 30 or tr.sum() < 50:
                continue
            lo_l[te] = gbt(loss="quantile", quantile=0.05).fit(X[tr], y[tr]).predict(X[te])
            hi_l[te] = gbt(loss="quantile", quantile=0.95).fit(X[tr], y[tr]).predict(X[te])
            per_cont_cov[c] = round(float(np.mean((y[te] >= lo_l[te]) & (y[te] <= hi_l[te]))), 3)
        ok = np.isfinite(lo_l)
        picp_loco = float(np.mean((y[ok] >= lo_l[ok]) & (y[ok] <= hi_l[ok])))

        # fraction of global land inside AoA
        Xg = grid[feats].dropna(how="all")
        dg = cKDTree(Xs).query(sc.transform(Xg), k=1)[0] / dbar
        gfrac = float(np.mean(dg <= thr))

        # leave-one-continent-out: fraction of held-out points inside AoA (expect ~0)
        loco_inside = []
        for c in sorted(d.continent.unique()):
            tr = (d.continent != c).values; te = (d.continent == c).values
            if te.sum() < 30:
                continue
            scf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X[tr])
            dlo = cKDTree(scf.transform(X[tr])).query(scf.transform(X[te]), k=1)[0] / dbar
            loco_inside.append(float(np.mean(dlo <= thr)))
        loco_frac = float(np.mean(loco_inside))

        res = {"target": label, "n": int(len(d)), "aoa_threshold": round(thr, 4),
               "frac_measured_inside": round(float(inside.mean()), 3),
               "R2_inside_AoA": None if r2_in is None else round(float(r2_in), 3),
               "R2_outside_AoA": None if r2_out is None else round(float(r2_out), 3),
               "global_land_frac_inside_AoA": round(gfrac, 3),
               "loco_frac_inside_AoA": round(loco_frac, 3),
               "PICP_90pct": round(picp, 3),
               "PICP_90pct_spatial_block": round(picp_spatial, 3),
               "PICP_90pct_loco": round(picp_loco, 3),
               "PICP_90pct_loco_by_continent": per_cont_cov}
        out["targets"].append(res)
        print(f"-- {label}: R2 inside={res['R2_inside_AoA']} outside={res['R2_outside_AoA']} "
              f"| inside {res['frac_measured_inside']:.0%} meas, {gfrac:.0%} global land, "
              f"{loco_frac:.0%} of LOCO test")
        print(f"   PICP(nominal 0.90): random={picp:.2f}  spatial-block={picp_spatial:.2f}  "
              f"LOCO={picp_loco:.2f}  per-continent LOCO={per_cont_cov}")

        dd = pd.DataFrame({"di": di, "err2": (y - pred) ** 2})
        dd["bin"] = pd.qcut(dd.di, 6, labels=False, duplicates="drop")
        for b, g in dd.groupby("bin"):
            fig_rows.append({"target": label, "di_mid": round(float(g.di.median()), 3),
                             "rmse": round(float(np.sqrt(g.err2.mean())), 4), "n": int(len(g))})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    FIGDATA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fig_rows).to_csv(FIGDATA, index=False)
    print(f"-> {OUT}\n-> {FIGDATA}")


if __name__ == "__main__":
    main()
