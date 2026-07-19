"""Paper 1 / M11 — WHY transfer fails: demonstrated mechanism (not asserted).

The manuscript currently *asserts* the mechanism in the Discussion: wilting point
transfers best because it is clay-dominated (clay being the best-mapped textural
fraction), while available water capacity fails because it leans on soil structure and
organic-matter interactions that globally mapped covariates capture only weakly. This
module turns that assertion into evidence.

Three demonstrations, all on the measured table, reusing the M5/M6 model:

 1. PREDICTOR RELIANCE per property (permutation importance). Shows WP leans on clay,
    AWC/FC lean on organic carbon + bioclim (structure/moisture proxies).

 2. CROSS-CONTINENTAL COVARIATE SHIFT. For each predictor, how far its distribution
    moves between a held-out continent and the rest (standardised mean shift, averaged
    over the 6 LOCO folds). Clay-type texture shifts less / is better constrained than
    the SOC + bioclim block.

 3. TRANSFER-VULNERABILITY INDEX = sum_f (importance_f * shift_f): the importance-
    weighted covariate shift each property must survive to transfer. Prediction: the
    ranking AWC > FC > WP mirrors the observed LOCO skill collapse (AWC worst, WP best),
    i.e. a property fails to transfer *because* the covariates it relies on are the ones
    that shift most and are mapped least reliably. We also show LOCO squared error rises
    with local organic-carbon percentile for AWC/FC but not WP.

Out: paper1_hydraulic/eval/m11_mechanism.json + figures/data/mechanism.csv
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa

MEAS = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m11_mechanism.json"
FIGDATA = Path(__file__).resolve().parents[1] / "figures/data/mechanism.csv"
SEED = 0
TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}


def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=350, max_depth=8,
                         learning_rate=0.06, random_state=SEED))


def feature_list(df):
    sg = [c for c in df.columns if c.startswith("sg_") and c != "sg_label"]
    bio = sorted(c for c in df.columns if c.startswith("chelsa__CHELSA_bio"))
    return sg + bio + ["depth_cm"]


def short(f):
    return (f.replace("chelsa__CHELSA_", "").replace("sg_", ""))


def main():
    df = pd.read_parquet(MEAS)
    feats = feature_list(df)
    rng = np.random.default_rng(SEED)

    # --- (2) cross-continental covariate shift (property-independent) -----------------
    # standardise each feature globally, then for each held-out continent measure how far
    # its mean sits from the rest, in global-SD units; average |shift| over continents.
    Xall = df[feats].astype(float)
    med = Xall.median(); Xf = Xall.fillna(med)
    gstd = Xf.std().replace(0, np.nan)
    shift = {}
    conts = sorted(df.continent.dropna().unique())
    for f in feats:
        s = []
        z = (Xf[f] - Xf[f].mean()) / gstd[f]
        for c in conts:
            m = (df.continent == c).values
            if m.sum() < 30:
                continue
            s.append(abs(z[m].mean() - z[~m].mean()))
        shift[f] = float(np.mean(s))

    out = {"n": int(len(df)), "features": len(feats), "continents": conts, "targets": []}
    fig_rows = []

    for tgt, label in TARGETS.items():
        d = df[np.isfinite(df[tgt])].copy().reset_index(drop=True)
        y = d[tgt].values
        X = d[feats]

        # --- (1) predictor reliance: permutation importance on a held-out split -------
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
        mdl = gbt().fit(Xtr, ytr)
        pi = permutation_importance(mdl, Xte, yte, n_repeats=8, random_state=SEED,
                                    scoring="r2")
        imp = np.clip(pi.importances_mean, 0, None)
        imp = imp / imp.sum() if imp.sum() > 0 else imp   # normalise to shares
        imp_by_feat = dict(zip(feats, imp))

        clay_share = float(imp_by_feat.get("sg_clay", 0.0))
        soc_share = float(imp_by_feat.get("sg_soc", 0.0))
        texture_share = float(sum(imp_by_feat.get(f, 0.0)
                                  for f in ("sg_clay", "sg_sand", "sg_silt")))
        bio_share = float(sum(v for f, v in imp_by_feat.items()
                              if f.startswith("chelsa__")))

        # --- (3) transfer-vulnerability index = importance-weighted covariate shift ----
        tvi = float(sum(imp_by_feat[f] * shift[f] for f in feats))

        # LOCO squared error vs local organic-carbon percentile (structure proxy)
        loco_err = np.full(len(d), np.nan)
        for c in conts:
            tr = (d.continent != c).values; te = (d.continent == c).values
            if te.sum() < 30 or tr.sum() < 100:
                continue
            loco_err[te] = (y[te] - gbt().fit(X[tr], y[tr]).predict(X[te])) ** 2
        ok = np.isfinite(loco_err) & np.isfinite(d["sg_soc"].values)
        soc_pct = pd.Series(d["sg_soc"].values[ok]).rank(pct=True).values
        rho, p = spearmanr(soc_pct, loco_err[ok])

        # top features by importance for the fig
        top = sorted(imp_by_feat.items(), key=lambda kv: kv[1], reverse=True)[:6]

        res = {"target": label,
               "clay_importance_share": round(clay_share, 3),
               "soc_importance_share": round(soc_share, 3),
               "texture_importance_share": round(texture_share, 3),
               "bioclim_importance_share": round(bio_share, 3),
               "transfer_vulnerability_index": round(tvi, 4),
               "loco_err_vs_soc_spearman": round(float(rho), 3),
               "loco_err_vs_soc_p": float(f"{p:.2e}"),
               "top_features": [(short(f), round(v, 3)) for f, v in top]}
        out["targets"].append(res)
        print(f"-- {label}: clay share={clay_share:.2f} texture={texture_share:.2f} "
              f"SOC={soc_share:.2f} bioclim={bio_share:.2f} | TVI={tvi:.4f} | "
              f"err~SOC rho={rho:+.2f}")
        for f, v in top:
            fig_rows.append({"target": label, "feature": short(f),
                             "importance": round(float(v), 4),
                             "cont_shift": round(float(shift[f]), 4)})

    # headline check: does TVI ranking match the LOCO skill ranking (AWC>FC>WP vulnerable)?
    tvi_rank = [t["target"] for t in sorted(out["targets"],
                key=lambda t: t["transfer_vulnerability_index"], reverse=True)]
    out["tvi_ranking_most_to_least_vulnerable"] = tvi_rank
    print(f"\nTVI ranking (most->least vulnerable): {tvi_rank}")
    print("Expected from LOCO skill: ['Available water', 'Field capacity', 'Wilting point']")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    FIGDATA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fig_rows).to_csv(FIGDATA, index=False)
    print(f"-> {OUT}\n-> {FIGDATA}")


if __name__ == "__main__":
    main()
