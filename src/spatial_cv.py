"""Paper 1 / M6 — robustness of the transferability result to sampling imbalance.

The measured sample is 70% Africa, so a reviewer may argue the leave-one-continent-out
(LOCO) collapse is an artefact of imbalance rather than a real limit of predictability.
We test this two ways, both standard and citable:

 1. SPATIAL-BLOCK CV (Roberts et al. 2017): assign sites to equal-area spatial blocks
    (B-degree grid) and do k-fold over whole blocks. Nearby points stay together
    (removing autocorrelation leakage) but every fold draws blocks from all continents
    (removing the continent confound). This is the fair middle ground between random CV
    (optimistic) and LOCO (confounded by imbalance).

 2. SPATIAL DECLUSTERING WEIGHTS (cell declustering, Deutsch 1989): weight each sample
    inversely to the sample density of its cell, so oversampled Africa does not dominate
    the global skill estimate. We report declustered (weighted) skill.

If the available-water collapse survives both, it is a real predictability limit, not a
sampling artefact -> the headline finding is robust.

Out: paper1_hydraulic/eval/m6_spatial_cv.json + figures/data/spatialcv.csv
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
OUT = DATA_ROOT / "paper1_hydraulic/eval/m6_spatial_cv.json"
FIGDATA = Path(__file__).resolve().parents[1] / "figures/data/spatialcv.csv"
SEED = 0
TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}
BLOCK_DEG = 5.0     # spatial block size (deg) for block CV and declustering cells
K = 10              # block-CV folds


def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=350, max_depth=8,
                         learning_rate=0.06, random_state=SEED))


def wr2(y, p, w):
    """weighted R^2."""
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - p) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return 1 - ss_res / ss_tot


def main():
    df = pd.read_parquet(MEAS)
    feats = [c for c in df.columns if c.startswith(("sg_", "chelsa__", "copernicus"))
             and c != "sg_label" and pd.api.types.is_numeric_dtype(df[c])] + ["depth_cm"]
    rng = np.random.default_rng(SEED)

    # spatial block id (lon,lat -> grid cell) shared by block-CV and declustering
    df["bx"] = np.floor(df.longitude / BLOCK_DEG).astype(int)
    df["by"] = np.floor(df.latitude / BLOCK_DEG).astype(int)
    df["block"] = df.bx.astype(str) + "_" + df.by.astype(str)

    # cell-declustering weights: inverse cell count, normalised to mean 1
    cell_n = df.groupby("block")["block"].transform("size")
    df["w_declus"] = (1.0 / cell_n)
    df["w_declus"] *= len(df) / df["w_declus"].sum()

    out = {"block_deg": BLOCK_DEG, "k": K, "n": int(len(df)),
           "n_blocks": int(df.block.nunique()), "targets": []}
    fig_rows = []

    for tgt, label in TARGETS.items():
        d = df[np.isfinite(df[tgt])].copy().reset_index(drop=True)
        y = d[tgt].values
        X = d[feats]

        # --- spatial block k-fold: assign whole blocks to folds ---
        blocks = d.block.unique()
        rng.shuffle(blocks)
        fold_of = {b: i % K for i, b in enumerate(blocks)}
        d["fold"] = d.block.map(fold_of)
        pred = np.full(len(d), np.nan)
        for k in range(K):
            tr = (d.fold != k).values; te = (d.fold == k).values
            if te.sum() < 20 or tr.sum() < 100:
                continue
            pred[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
        m = np.isfinite(pred)
        # unweighted and declustered (weighted) block-CV skill
        r2_block = wr2(y[m], pred[m], np.ones(m.sum()))
        r2_block_dc = wr2(y[m], pred[m], d.w_declus.values[m])

        res = {"target": label,
               "block_cv_R2": round(float(r2_block), 3),
               "block_cv_R2_declustered": round(float(r2_block_dc), 3)}
        out["targets"].append(res)
        print(f"-- {label}: block-CV R2={res['block_cv_R2']:.3f} | "
              f"declustered={res['block_cv_R2_declustered']:.3f}")
        fig_rows.append({"target": label, **{k: res[k] for k in res if k != "target"}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    FIGDATA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fig_rows).to_csv(FIGDATA, index=False)
    print(f"-> {OUT}\n-> {FIGDATA}")


if __name__ == "__main__":
    main()
