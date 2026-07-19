"""Paper 1 / M6b — block-bootstrap CIs for the declustered-vs-unweighted spatial skill.

Closes the soft spot in M6: we claimed the unweighted (0.48) and density-declustered
(0.47) spatial-block R2 are "essentially identical", but reported only point estimates.
This puts a confidence interval on each AND on their difference, using a BLOCK bootstrap
(resample whole 5-degree blocks with replacement) so the spatial autocorrelation is
respected -- a point bootstrap would understate the uncertainty.

Logic:
  1. spatial-block 10-fold CV -> out-of-fold prediction for every measured point (once).
  2. block bootstrap B times: resample blocks with replacement, recompute unweighted and
     declustered R2 (and their difference) on the resampled points.
  3. report point estimate + 95% percentile CI for each, and the CI of the difference.

If the difference CI comfortably brackets 0, "imbalance does not bias the estimate" is
statistically supported, not just asserted.

Out: paper1_hydraulic/eval/m6b_bootstrap.json
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa

MEAS = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m6b_bootstrap.json"
SEED = 0
B = 5.0          # block size (deg) -- same as M6
K = 10           # CV folds
NBOOT = 2000     # bootstrap iterations
TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}


def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=350, max_depth=8,
                                      learning_rate=0.06, random_state=SEED))


def wr2(y, p, w):
    """weighted R^2."""
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - p) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return 1.0 - ss_res / ss_tot


def ci(a, lo=2.5, hi=97.5):
    return [round(float(np.percentile(a, lo)), 4), round(float(np.percentile(a, hi)), 4)]


def main():
    df = pd.read_parquet(MEAS)
    feats = [c for c in df.columns if c.startswith(("sg_", "chelsa__", "copernicus"))
             and c != "sg_label" and pd.api.types.is_numeric_dtype(df[c])] + ["depth_cm"]
    df = df.copy()
    df["bx"] = np.floor(df.longitude / B).astype(int)
    df["by"] = np.floor(df.latitude / B).astype(int)
    df["block"] = df.bx.astype(str) + "_" + df.by.astype(str)
    cell_n = df.groupby("block")["block"].transform("size")
    df["w"] = (1.0 / cell_n)
    df["w"] *= len(df) / df["w"].sum()       # declustering weights, mean 1
    rng = np.random.default_rng(SEED)

    out = {"n_boot": NBOOT, "block_deg": B, "k": K, "targets": []}
    for tgt, label in TARGETS.items():
        d = df[np.isfinite(df[tgt])].copy().reset_index(drop=True)
        y = d[tgt].values

        # 1) spatial-block CV -> out-of-fold predictions
        blocks = d.block.unique().copy(); rng.shuffle(blocks)
        fold = {b: i % K for i, b in enumerate(blocks)}
        d["fold"] = d.block.map(fold)
        pred = np.full(len(d), np.nan)
        for k in range(K):
            tr = (d.fold != k).values; te = (d.fold == k).values
            if te.sum() < 20 or tr.sum() < 100:
                continue
            pred[te] = gbt().fit(d.loc[tr, feats], y[tr]).predict(d.loc[te, feats])
        m = np.isfinite(pred)
        d, y, pred = d[m].reset_index(drop=True), y[m], pred[m]
        w = d.w.values

        r2_unw = wr2(y, pred, np.ones(len(y)))
        r2_dec = wr2(y, pred, w)

        # 2) block bootstrap: resample whole blocks with replacement
        blk_ids = d.block.values
        uniq = d.block.unique()
        # precompute point indices per block
        idx_by_block = {b: np.where(blk_ids == b)[0] for b in uniq}
        bu, bd, bdiff = [], [], []
        for _ in range(NBOOT):
            samp = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_block[b] for b in samp])
            yy, pp, ww = y[idx], pred[idx], w[idx]
            u = wr2(yy, pp, np.ones(len(yy)))
            dd = wr2(yy, pp, ww)
            bu.append(u); bd.append(dd); bdiff.append(dd - u)
        bu, bd, bdiff = map(np.array, (bu, bd, bdiff))

        res = {
            "target": label,
            "n": int(len(y)), "n_blocks": int(len(uniq)),
            "R2_unweighted": round(float(r2_unw), 4), "R2_unweighted_CI95": ci(bu),
            "R2_declustered": round(float(r2_dec), 4), "R2_declustered_CI95": ci(bd),
            "difference_declustered_minus_unweighted": round(float(r2_dec - r2_unw), 4),
            "difference_CI95": ci(bdiff),
            "CI_brackets_zero": bool(np.percentile(bdiff, 2.5) <= 0 <= np.percentile(bdiff, 97.5)),
        }
        out["targets"].append(res)
        print(f"-- {label}: unweighted {res['R2_unweighted']} {res['R2_unweighted_CI95']} | "
              f"declustered {res['R2_declustered']} {res['R2_declustered_CI95']}")
        print(f"   difference {res['difference_declustered_minus_unweighted']} "
              f"CI95 {res['difference_CI95']}  brackets 0: {res['CI_brackets_zero']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
