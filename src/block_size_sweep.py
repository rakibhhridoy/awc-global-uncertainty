"""Paper 1 / M16 — block-size sensitivity of the spatial-block skill.

The "fair" spatial-block skill (headline: 5-degree blocks, density-declustered) is the
paper's central baseline. Blocked-CV estimates are known to depend on block size, so this
sweep re-runs the *identical* M6 procedure (same GBT, same 10-fold whole-block assignment,
same cell-declustering weights) at a range of block sizes and reports the declustered R^2
for each target. Robustness = a stable plateau around 5 degrees rather than a value that
swings with the block-size choice.

Out: paper1_hydraulic/eval/m16_block_size_sweep.json (+ mirror to results/ + figures/data)
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
OUT = DATA_ROOT / "paper1_hydraulic/eval/m16_block_size_sweep.json"
RESULTS = Path(__file__).resolve().parents[1] / "results/m16_block_size_sweep.json"
FIGDATA = Path(__file__).resolve().parents[1] / "figures/data/block_size_sweep.csv"
SEED = 0
K = 10
TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}
BLOCK_SIZES = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0]


def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=350, max_depth=8,
                         learning_rate=0.06, random_state=SEED))


def wr2(y, p, w):
    ybar = np.average(y, weights=w)
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - ybar) ** 2)


def block_cv(d, feats, tgt, block_deg, rng):
    d = d.copy()
    d["block"] = (np.floor(d.longitude / block_deg).astype(int).astype(str) + "_" +
                  np.floor(d.latitude / block_deg).astype(int).astype(str))
    cell_n = d.groupby("block")["block"].transform("size")
    w = 1.0 / cell_n
    w *= len(d) / w.sum()
    y = d[tgt].values
    X = d[feats]
    blocks = d.block.unique().copy()
    rng.shuffle(blocks)
    fold_of = {b: i % K for i, b in enumerate(blocks)}
    fold = d.block.map(fold_of).values
    pred = np.full(len(d), np.nan)
    for k in range(K):
        tr = fold != k; te = fold == k
        if te.sum() < 20 or tr.sum() < 100:
            continue
        pred[te] = gbt().fit(X[tr], y[tr]).predict(X[te])
    m = np.isfinite(pred)
    return (round(float(wr2(y[m], pred[m], np.ones(m.sum()))), 3),
            round(float(wr2(y[m], pred[m], w.values[m])), 3),
            int(d.block.nunique()))


def main():
    df = pd.read_parquet(MEAS)
    feats = [c for c in df.columns if c.startswith(("sg_", "chelsa__", "copernicus"))
             and c != "sg_label" and pd.api.types.is_numeric_dtype(df[c])] + ["depth_cm"]

    out = {"k": K, "seed": SEED, "block_sizes_deg": BLOCK_SIZES, "targets": {}}
    fig_rows = []
    for tgt, label in TARGETS.items():
        d = df[np.isfinite(df[tgt])].reset_index(drop=True)
        curve = []
        for B in BLOCK_SIZES:
            rng = np.random.default_rng(SEED)
            r2, r2_dc, nb = block_cv(d, feats, tgt, B, rng)
            curve.append({"block_deg": B, "R2": r2, "R2_declustered": r2_dc, "n_blocks": nb})
            fig_rows.append({"target": label, "block_deg": B, "R2": r2, "R2_declustered": r2_dc})
            print(f"  {label:16} B={B:>4}deg  R2={r2:.3f}  declustered={r2_dc:.3f}  ({nb} blocks)")
        dc = [c["R2_declustered"] for c in curve]
        out["targets"][label] = {"curve": curve,
                                 "declustered_min": min(dc), "declustered_max": max(dc),
                                 "declustered_range": round(max(dc) - min(dc), 3),
                                 "declustered_at_5deg": next(c["R2_declustered"] for c in curve if c["block_deg"] == 5.0)}
        print(f"  -> {label}: declustered spans {min(dc):.3f}-{max(dc):.3f} "
              f"(range {max(dc)-min(dc):.3f}) across 1-15 deg\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    if RESULTS.parent.exists():
        RESULTS.write_text(json.dumps(out, indent=1))
    pd.DataFrame(fig_rows).to_csv(FIGDATA, index=False)
    print(f"-> {OUT}\n-> {FIGDATA}")


if __name__ == "__main__":
    main()
