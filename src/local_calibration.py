"""Paper 1 / M9 — local-calibration recovery curve (the constructive fix).

The diagnosis (M4b/M6) is that hydraulic maps do not transfer to unsampled continents.
The natural question for a practitioner is: how many LOCAL measurements does an
unsampled region need before predictions become reliable again? This experiment answers
it directly and turns the cautionary result into an actionable one.

Two split designs are run so the recovery is not itself inflated by the very spatial
autocorrelation this paper exposes:

  spatial (headline) -- within the held-out continent C, profiles are grouped into
      geographic blocks (a ~BLOCK_DEG grid). Whole blocks are assigned either to the
      fixed TEST set or to the CALIBRATION POOL, so an added calibration profile is
      never a point-neighbour of a test profile. This measures what a realistic, spatially
      distributed local sampling campaign recovers (and avoids point-level interpolation).

  random (upper bound) -- the original profile-level random split. Calibration and test
      profiles are spatially intermingled, so the curve interpolates between near
      neighbours; it is reported only as an optimistic upper bound on recovery.

Design (leakage-safe, profile-level), for each split mode:
  For each held-out continent C:
    - assign C's PROFILES to a fixed TEST set and a disjoint CALIBRATION POOL
      (by geographic block for 'spatial', at random for 'random');
    - base training = all layers from the OTHER continents (the LOCO setting);
    - for N in a sweep, add N calibration profiles (all their depth layers) from C to the
      training set, refit, and score on the fixed TEST set;
    - average over several random draws of the N calibration profiles.
  N = 0 reproduces pure leave-one-continent-out; increasing N traces the recovery curve.

Out: paper1_hydraulic/eval/m9_recovery.json + figures/data/recovery.csv
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa

MEAS = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m9_recovery.json"
FIG = Path(__file__).resolve().parents[1] / "figures/data/recovery.csv"
SEED = 0
TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}
N_SWEEP = [0, 5, 10, 15, 25, 50, 100]
N_DRAWS = 8                 # random draws of the N calibration profiles, averaged
MIN_SITES = 80             # skip continents too small to split fairly
BLOCK_DEG = 1.0            # geographic block size (deg) for the spatial test/pool split
MODES = ["spatial", "random"]


def gbt():
    return make_pipeline(SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_iter=300, max_depth=8,
                                      learning_rate=0.06, random_state=SEED))


def split_sites(inC, mode, rng):
    """Return (test_sites:set, pool_sites:np.ndarray) for a continent block `inC`.

    spatial -- assign whole BLOCK_DEG geographic cells to test vs pool, so calibration
               profiles are spatially separated from test profiles.
    random  -- shuffle profiles and split (spatially intermingled; upper bound).
    """
    sites = inC.profile_id.unique()
    if mode == "random":
        s = sites.copy(); rng.shuffle(s)
        n_test = min(200, len(s) // 2)
        return set(s[:n_test]), s[n_test:]

    # spatial: one geographic block per profile (use the profile's median coordinate)
    pc = (inC.groupby("profile_id")[["longitude", "latitude"]].median())
    blk = (np.floor(pc["longitude"] / BLOCK_DEG).astype(int).astype(str) + "_"
           + np.floor(pc["latitude"] / BLOCK_DEG).astype(int).astype(str))
    blocks = blk.unique().copy(); rng.shuffle(blocks)
    # walk blocks until the test set holds ~half the profiles (capped at 200)
    target_test = min(200, len(sites) // 2)
    test_blocks, n = set(), 0
    for b in blocks:
        if n >= target_test:
            break
        test_blocks.add(b); n += int((blk == b).sum())
    test_sites = set(blk.index[blk.isin(test_blocks)])
    pool_sites = np.array([s for s in sites if s not in test_sites])
    return test_sites, pool_sites


def recovery_for_mode(df, feats, mode, rng):
    """Compute per-continent recovery curves and the mean curve for one split mode."""
    fig_rows, results = [], []
    for tgt, label in TARGETS.items():
        d = df[np.isfinite(df[tgt])].copy()
        per_continent = {}
        for c in sorted(d.continent.unique()):
            inC = d[d.continent == c]
            if inC.profile_id.nunique() < MIN_SITES:
                continue
            other = d[d.continent != c]                      # base training (LOCO)
            test_sites, pool_sites = split_sites(inC, mode, rng)
            test = inC[inC.profile_id.isin(test_sites)]
            yte = test[tgt].values
            if len(pool_sites) == 0 or len(yte) == 0:
                continue

            curve = []
            for N in N_SWEEP:
                if N > len(pool_sites):
                    curve.append(None); continue
                scores = []
                for dseed in range(N_DRAWS):
                    rg = np.random.default_rng(1000 * dseed + N)
                    add_sites = rg.choice(pool_sites, size=N, replace=False) if N > 0 else []
                    add = inC[inC.profile_id.isin(set(add_sites))]
                    tr = pd.concat([other, add], ignore_index=True)
                    m = gbt().fit(tr[feats], tr[tgt].values)
                    scores.append(r2_score(yte, m.predict(test[feats])))
                    if N == 0:                                # deterministic; one draw enough
                        break
                curve.append(float(np.mean(scores)))
            per_continent[c] = curve
            for N, r2 in zip(N_SWEEP, curve):
                if r2 is not None:
                    fig_rows.append({"split": mode, "target": label, "continent": c,
                                     "n_local": N, "r2": round(r2, 4)})

        mat = np.array([[v if v is not None else np.nan for v in cur] for cur in per_continent.values()])
        mean_curve = np.nanmean(mat, axis=0) if len(mat) else np.full(len(N_SWEEP), np.nan)
        res = {"split": mode, "target": label,
               "mean_recovery_curve": {str(N): (None if np.isnan(v) else round(float(v), 3))
                                       for N, v in zip(N_SWEEP, mean_curve)},
               "per_continent": {c: [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(v, 3) for v in cur]
                                 for c, cur in per_continent.items()}}
        n_to_04 = next((N for N, v in zip(N_SWEEP, mean_curve) if not np.isnan(v) and v >= 0.4), None)
        res["n_local_to_reach_R2_0p4"] = n_to_04
        res["loco_R2_at_N0"] = None if np.isnan(mean_curve[0]) else round(float(mean_curve[0]), 3)
        res["recovered_R2_at_N25"] = None if np.isnan(mean_curve[N_SWEEP.index(25)]) else round(float(mean_curve[N_SWEEP.index(25)]), 3)
        results.append(res)
        print(f"[{mode:7s}] {label}: mean curve {[None if np.isnan(v) else round(float(v),2) for v in mean_curve]}")
        print(f"          N(local) to R2>=0.4: {n_to_04} | R2 N=0 {res['loco_R2_at_N0']} -> N=25 {res['recovered_R2_at_N25']}")
    return results, fig_rows


def main():
    df = pd.read_parquet(MEAS)
    feats = [c for c in df.columns if c.startswith(("sg_", "chelsa__", "copernicus"))
             and c != "sg_label" and pd.api.types.is_numeric_dtype(df[c])] + ["depth_cm"]

    out = {"n_sweep": N_SWEEP, "n_draws": N_DRAWS, "block_deg": BLOCK_DEG,
           "headline_split": "spatial",
           "note": ("Recovery under a spatial-block test/pool split (calibration profiles "
                    "spatially separated from test profiles) is the headline; the random "
                    "intermingled split is reported as an optimistic upper bound."),
           "targets": [], "targets_upper_bound": []}
    all_fig_rows = []
    for mode in MODES:
        rng = np.random.default_rng(SEED)            # same seed per mode for comparability
        results, fig_rows = recovery_for_mode(df, feats, mode, rng)
        out["targets" if mode == "spatial" else "targets_upper_bound"] = results
        all_fig_rows += fig_rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_fig_rows).to_csv(FIG, index=False)
    print(f"-> {OUT}\n-> {FIG}")


if __name__ == "__main__":
    main()
