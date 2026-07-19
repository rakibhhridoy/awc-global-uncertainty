"""Paper 1 / M15 — measurement-noise ceiling on achievable R^2.

WoSIS retention values are pooled across laboratories, methods and eras, so they carry
measurement error; and at the 1 km covariate resolution, sub-pixel soil variability is
also irreducible. Both are captured by CO-LOCATED profiles: independent WoSIS layers that
fall in the same ~1 km cell and depth bin share an identical covariate vector, so the
variance *between* them is variance no covariate-based model can explain.

Ceiling: R^2_max = 1 - sigma^2_within / sigma^2_total, where sigma^2_within is the pooled
within-(cell,depth) variance over groups with >=2 co-located measurements. This is the
best R^2 attainable by ANY model built on these 1 km covariates; observed skill compared
against it separates genuine model limitation from irreducible reference-data noise.

Out: paper1_hydraulic/eval/m15_noise_ceiling.json  (+ mirror to results/ if present)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "interim" / "measured_table.parquet"
OUT = ROOT / "data" / "eval" / "m15_noise_ceiling.json"
RESULTS = ROOT / "results" / "m15_noise_ceiling.json"

TARGETS = {"meas_fc": "Field capacity", "meas_wp": "Wilting point", "meas_awc": "Available water"}
# observed skill (from m4b / m6) for the comparison, m3/m3 targets
OBSERVED = {
    "meas_fc":  {"ptf": 0.28, "random": 0.80, "spatial_declustered": 0.50, "loco": 0.16},
    "meas_wp":  {"ptf": 0.20, "random": 0.79, "spatial_declustered": 0.47, "loco": 0.29},
    "meas_awc": {"ptf": 0.35, "random": 0.72, "spatial_declustered": 0.47, "loco": 0.00},
}


def ceiling_at(df, rnd):
    d = df.copy()
    d["_g"] = (d.longitude.round(rnd).astype(str) + "_" +
               d.latitude.round(rnd).astype(str) + "_" + d.depth_cm.astype(str))
    out = {}
    for t in TARGETS:
        n = d.groupby("_g")[t].transform("size")
        rep = d[n >= 2]
        gm = rep.groupby("_g")[t].transform("mean")
        ss = ((rep[t] - gm) ** 2).groupby(rep["_g"]).sum() if len(rep) else pd.Series(dtype=float)
        dof = rep.groupby("_g")[t].size() - 1
        s2_within = ss.sum() / dof.sum()
        s2_total = d[t].var(ddof=1)
        out[t] = {
            "ceiling_R2": round(float(1 - s2_within / s2_total), 3),
            "noise_RMSE": round(float(np.sqrt(s2_within)), 4),
            "sigma2_within": round(float(s2_within), 6),
            "sigma2_total": round(float(s2_total), 6),
            "n_groups": int((dof > 0).sum()),
            "n_rows": int(len(rep)),
        }
    return out


def main():
    df = pd.read_parquet(TABLE)[["longitude", "latitude", "depth_cm", *TARGETS]]
    primary = ceiling_at(df, 2)     # 0.01 deg ~ 1.1 km, matches the 1 km grid
    strict = ceiling_at(df, 3)      # 0.001 deg ~ 110 m, robustness

    res = {"colocation_deg": 0.01, "note": "co-located WoSIS profiles, same ~1km cell + depth bin",
           "targets": {}}
    for t, name in TARGETS.items():
        c = primary[t]
        obs = OBSERVED[t]
        res["targets"][name] = {
            **c,
            "ceiling_R2_strict_110m": strict[t]["ceiling_R2"],
            "observed": obs,
            "spatial_as_frac_of_ceiling": round(obs["spatial_declustered"] / c["ceiling_R2"], 2),
            "random_as_frac_of_ceiling": round(obs["random"] / c["ceiling_R2"], 2),
        }
    OUT.write_text(json.dumps(res, indent=1))
    if RESULTS.parent.exists():
        RESULTS.write_text(json.dumps(res, indent=1))
    for name, r in res["targets"].items():
        print(f"{name:16} ceiling R2={r['ceiling_R2']:.3f}  "
              f"spatial {r['observed']['spatial_declustered']:.2f} "
              f"({r['spatial_as_frac_of_ceiling']*100:.0f}% of ceiling)  "
              f"random {r['observed']['random']:.2f} "
              f"({r['random_as_frac_of_ceiling']*100:.0f}% of ceiling)")
    print("->", OUT)


if __name__ == "__main__":
    main()
