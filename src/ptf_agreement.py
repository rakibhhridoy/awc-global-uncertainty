"""Paper 1 / M17 — bias/variance decomposition of the existing PTF product's disagreement.

R^2 alone conflates pattern disagreement, calibration offset and range compression. We
decompose the OpenLandMap product vs measured WoSIS retention into Pearson r (pattern),
Lin's concordance (CCC, agreement incl. bias), mean bias, and the regression slope of
product-on-measured (attenuation / range compression). Shows the low R^2 is genuine
scatter + compression, not a removable offset.

Out: paper1_hydraulic/eval/m17_ptf_agreement.json (+ mirror to results/)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "interim" / "measured_table.parquet"
OUT = ROOT / "data" / "eval" / "m17_ptf_agreement.json"
RESULTS = ROOT / "results" / "m17_ptf_agreement.json"
TARGETS = [("fc", "Field capacity"), ("wp", "Wilting point"), ("awc", "Available water")]


def ccc(o, p):
    mo, mp = o.mean(), p.mean()
    cov = ((o - mo) * (p - mp)).mean()
    return float(2 * cov / (o.var() + p.var() + (mo - mp) ** 2))


def main():
    df = pd.read_parquet(TABLE)
    res = {"targets": {}}
    for t, name in TARGETS:
        o, p = df[f"meas_{t}"].values, df[f"ptf_{t}"].values
        m = np.isfinite(o) & np.isfinite(p)
        o, p = o[m], p[m]
        r = float(stats.pearsonr(o, p)[0])
        R2 = float(1 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))
        res["targets"][name] = {
            "n": int(m.sum()),
            "pearson_r": round(r, 3),
            "r2": round(r ** 2, 3),
            "R2": round(R2, 3),
            "CCC": round(ccc(o, p), 3),
            "mean_bias": round(float((p - o).mean()), 4),
            "slope_product_on_measured": round(float(np.polyfit(o, p, 1)[0]), 3),
        }
    OUT.write_text(json.dumps(res, indent=1))
    if RESULTS.parent.exists():
        RESULTS.write_text(json.dumps(res, indent=1))
    for name, r in res["targets"].items():
        print(f"{name:16} r={r['pearson_r']:.2f} R2={r['R2']:.2f} CCC={r['CCC']:.2f} "
              f"bias={r['mean_bias']:+.3f} slope={r['slope_product_on_measured']:.2f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
