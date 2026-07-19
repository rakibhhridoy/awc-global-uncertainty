"""Paper 1 / M12 — honest out-of-domain intervals via localized conformal prediction.

M5 diagnoses a real problem: the released 90% quantile-GBT intervals are calibrated only
in the sampled regime. Their coverage (PICP) is ~0.85 under random folds but falls to
~0.55-0.68 under leave-one-continent-out, i.e. they are too narrow exactly where the map
is least trustworthy. Shipping intervals we have shown to be miscalibrated OOD is an
internal tension. This module fixes it with a principled method.

We conformalize the existing quantile-GBT (Conformalized Quantile Regression, Romano,
Patterson & Candes 2019), which wraps the same q05/q95 models the paper already uses and
adds a finite-sample coverage guarantee. Vanilla (split) CQR gives only *marginal*
coverage, which — as we show — still under-covers out of domain. So we add a MONDRIAN /
LOCALIZED variant that conditions the conformal correction on the dissimilarity index
(DI, the same quantity that defines the Area of Applicability): calibration residuals are
binned by DI and each bin gets its own correction, so intervals widen automatically as a
prediction moves out of the sampled space.

We compare PICP (nominal 0.90) and mean interval width under three regimes — random,
spatial-block, leave-one-continent-out — for:
  (a) quantile GBT           (the paper's current intervals)
  (b) split CQR              (marginal conformal guarantee)
  (c) Mondrian/DI-CQR        (dissimilarity-conditional guarantee)
The claim to plant: only (c) holds ~0.90 coverage under LOCO, at the cost of honestly
wider intervals out of domain.

Out: paper1_hydraulic/eval/m12_conformal.json + figures/data/conformal.csv
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
OUT = DATA_ROOT / "paper1_hydraulic/eval/m12_conformal.json"
FIGDATA = Path(__file__).resolve().parents[1] / "figures/data/conformal.csv"
SEED = 0
ALPHA = 0.10          # 90% target
BLOCK_DEG = 5.0
BLOCK_K = 10
DI_BINS = 5           # Mondrian strata on the dissimilarity index
TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}


def gbt(**kw):
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=350, max_depth=8,
                         learning_rate=0.06, random_state=SEED, **kw))


def di_of(Xtr, Xq, dbar):
    """dissimilarity index of query rows against a training set (std feature space)."""
    sc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(Xtr)
    tree = cKDTree(sc.transform(Xtr))
    return tree.query(sc.transform(Xq), k=1)[0] / dbar


def conformal_eval(X, y, di_ref_dbar, feats, idx_tr, idx_ca, idx_te):
    """Fit quantile models on TRAIN, calibrate on CAL, score TEST.
    Returns coverage/width for quantile-GBT, split-CQR and Mondrian(DI)-CQR."""
    Xtr, ytr = X.iloc[idx_tr], y[idx_tr]
    Xca, yca = X.iloc[idx_ca], y[idx_ca]
    Xte, yte = X.iloc[idx_te], y[idx_te]

    qlo = gbt(loss="quantile", quantile=ALPHA / 2).fit(Xtr, ytr)
    qhi = gbt(loss="quantile", quantile=1 - ALPHA / 2).fit(Xtr, ytr)

    # (a) raw quantile GBT on TEST
    lo_t, hi_t = qlo.predict(Xte), qhi.predict(Xte)
    cov_q = float(np.mean((yte >= lo_t) & (yte <= hi_t)))
    wid_q = float(np.mean(hi_t - lo_t))

    # CQR nonconformity on CAL: E = max(qlo - y, y - qhi)
    lo_c, hi_c = qlo.predict(Xca), qhi.predict(Xca)
    E = np.maximum(lo_c - yca, yca - hi_c)
    n = len(E)
    k = int(np.ceil((1 - ALPHA) * (n + 1)))
    k = min(max(k, 1), n)

    # (b) split CQR: single marginal correction
    Q = np.sort(E)[k - 1]
    cov_s = float(np.mean((yte >= lo_t - Q) & (yte <= hi_t + Q)))
    wid_s = float(np.mean((hi_t + Q) - (lo_t - Q)))

    # (c) Mondrian / DI-localized CQR: correction per DI stratum
    di_ca = di_of(Xtr, Xca, di_ref_dbar)
    di_te = di_of(Xtr, Xte, di_ref_dbar)
    # bin edges from calibration DI
    edges = np.quantile(di_ca, np.linspace(0, 1, DI_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    Qbin = np.empty(DI_BINS)
    for b in range(DI_BINS):
        m = (di_ca >= edges[b]) & (di_ca < edges[b + 1])
        if m.sum() >= 20:
            Eb = np.sort(E[m]); nb = len(Eb)
            kb = min(max(int(np.ceil((1 - ALPHA) * (nb + 1))), 1), nb)
            Qbin[b] = Eb[kb - 1]
        else:                       # fall back to the global correction in sparse bins
            Qbin[b] = Q
    Qbin = np.maximum.accumulate(Qbin)     # enforce monotone: wider as DI grows
    bin_te = np.clip(np.digitize(di_te, edges[1:-1]), 0, DI_BINS - 1)
    Qm = Qbin[bin_te]
    cov_m = float(np.mean((yte >= lo_t - Qm) & (yte <= hi_t + Qm)))
    wid_m = float(np.mean((hi_t + Qm) - (lo_t - Qm)))

    # per-point detail (test DI + hit/miss per method) for conditional-coverage analysis
    detail = {"di": di_te,
              "hit_quantile": (yte >= lo_t) & (yte <= hi_t),
              "hit_split_cqr": (yte >= lo_t - Q) & (yte <= hi_t + Q),
              "hit_mondrian_cqr": (yte >= lo_t - Qm) & (yte <= hi_t + Qm)}
    return {"quantile": (cov_q, wid_q), "split_cqr": (cov_s, wid_s),
            "mondrian_cqr": (cov_m, wid_m), "_detail": detail}


def main():
    df = pd.read_parquet(MEAS)
    sg = [c for c in df.columns if c.startswith("sg_") and c != "sg_label"]
    bio = sorted(c for c in df.columns if c.startswith("chelsa__CHELSA_bio"))
    feats = sg + bio + ["depth_cm"]
    rng = np.random.default_rng(SEED)

    out = {"n": int(len(df)), "alpha": ALPHA, "di_bins": DI_BINS, "targets": []}
    fig_rows = []
    cond_fig_rows = []

    for tgt, label in TARGETS.items():
        d = df[np.isfinite(df[tgt])].copy().reset_index(drop=True)
        y = d[tgt].values
        X = d[feats]

        # global dbar reference (as in M5) for DI scaling
        sc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
        Xs = sc.transform(X)
        smp = Xs[rng.choice(len(Xs), size=2000, replace=False)]
        dbar = float(np.mean(np.linalg.norm(smp[:, None] - smp[None, :], axis=2)))

        n = len(d)
        idx = np.arange(n)

        # ---- RANDOM regime: 60/20/20 train/cal/test ----
        rng.shuffle(idx)
        a, b = int(0.6 * n), int(0.8 * n)
        r_rand = conformal_eval(X, y, dbar, feats, idx[:a], idx[a:b], idx[b:])

        # ---- SPATIAL-BLOCK regime: hold out whole 5-deg blocks as TEST ----
        bx = np.floor(d.longitude / BLOCK_DEG).astype(int).astype(str)
        by = np.floor(d.latitude / BLOCK_DEG).astype(int).astype(str)
        block = (bx + "_" + by).values
        ublk = np.array(sorted(set(block))); rng.shuffle(ublk)
        test_blocks = set(ublk[: max(1, len(ublk) // BLOCK_K)])
        te = np.array([bl in test_blocks for bl in block])
        rest = idx[~te]; rng.shuffle(rest)
        cut = int(0.75 * len(rest))
        r_block = conformal_eval(X, y, dbar, feats, rest[:cut], rest[cut:], idx[te])

        # ---- LOCO regime: pooled over the 6 held-out continents (test-point weighted) ----
        cov = {"quantile": 0.0, "split_cqr": 0.0, "mondrian_cqr": 0.0}
        wid = {"quantile": 0.0, "split_cqr": 0.0, "mondrian_cqr": 0.0}
        ntot = 0
        det = {"di": [], "hit_quantile": [], "hit_split_cqr": [], "hit_mondrian_cqr": []}
        for c in sorted(d.continent.unique()):
            te_c = (d.continent == c).values
            rest_c = idx[~te_c]
            nc = int(te_c.sum())
            if nc < 40 or len(rest_c) < 200:
                continue
            rng.shuffle(rest_c)
            cut = int(0.75 * len(rest_c))
            r = conformal_eval(X, y, dbar, feats, rest_c[:cut], rest_c[cut:], idx[te_c])
            for meth in cov:
                cov[meth] += r[meth][0] * nc
                wid[meth] += r[meth][1] * nc
            ntot += nc
            for kdet in det:
                det[kdet].append(r["_detail"][kdet])
        r_loco = {m: (float(cov[m] / ntot), float(wid[m] / ntot)) for m in cov}

        # ---- CONDITIONAL COVERAGE by dissimilarity: does the fix hold for the MOST
        # out-of-domain points, or only on average? Pool all LOCO test points, bin by DI,
        # and report per-bin coverage for split-CQR vs Mondrian. Split-CQR (one marginal
        # correction) is expected to under-cover the highest-DI bin; Mondrian should not.
        di_all = np.concatenate(det["di"])
        hits = {m: np.concatenate(det[f"hit_{m}"]) for m in cov}
        # rank-based quantile bins, dropping tied edges (a large DI~0 mass from
        # near-duplicate profiles otherwise leaves an empty bottom bin)
        binlab = pd.qcut(di_all, DI_BINS, labels=False, duplicates="drop")
        cond = []
        for b in sorted(pd.unique(binlab)):
            m = binlab == b
            row = {"target": label, "di_bin": int(b) + 1, "n": int(m.sum()),
                   "di_mid": round(float(np.median(di_all[m])), 3)}
            for meth in cov:
                row[f"cov_{meth}"] = round(float(hits[meth][m].mean()), 3)
            cond.append(row)
            cond_fig_rows.append(row)

        res = {"target": label, "loco_conditional_coverage_by_di": cond}
        for regime, r in [("random", r_rand), ("spatial_block", r_block), ("loco", r_loco)]:
            for meth in ("quantile", "split_cqr", "mondrian_cqr"):
                c_, w_ = r[meth]
                res[f"PICP_{regime}_{meth}"] = round(c_, 3)
                res[f"width_{regime}_{meth}"] = round(w_, 4)
                fig_rows.append({"target": label, "regime": regime, "method": meth,
                                 "PICP": round(c_, 3), "width": round(w_, 4)})
        out["targets"].append(res)
        print(f"-- {label}  PICP (nominal 0.90):")
        for regime in ("random", "spatial_block", "loco"):
            print(f"     {regime:14s} quantile={res[f'PICP_{regime}_quantile']:.2f}  "
                  f"split-CQR={res[f'PICP_{regime}_split_cqr']:.2f}  "
                  f"Mondrian-CQR={res[f'PICP_{regime}_mondrian_cqr']:.2f}")
        hi_bin = cond[-1]
        print(f"     conditional coverage, highest-DI bin (n={hi_bin['n']}): "
              f"split-CQR={hi_bin['cov_split_cqr']:.2f}  "
              f"Mondrian-CQR={hi_bin['cov_mondrian_cqr']:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    FIGDATA.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fig_rows).to_csv(FIGDATA, index=False)
    cond_path = FIGDATA.parent / "conformal_conditional.csv"
    pd.DataFrame(cond_fig_rows).to_csv(cond_path, index=False)
    print(f"-> {OUT}\n-> {FIGDATA}\n-> {cond_path}")


if __name__ == "__main__":
    main()
