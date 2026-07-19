"""Paper 1 / M8 — propagate AWC uncertainty into an agronomic drought-buffering metric.

Demonstrates (not merely asserts) the environmental significance of the hydraulic-map
uncertainty. A transparent soil-water bucket gives the plant-available water storage
PAWS = AWC x root-zone depth (mm). During a rain-free spell, the time to deplete that
store to wilting -- the drought-buffering time -- is PAWS / ETc, where ETc is crop
evapotranspiration. The 90% prediction interval on AWC therefore maps linearly to a
90% interval on buffering days.

The ROBUST, threshold-free headline is the buffering-time uncertainty itself: the 90%
interval is nearly as wide as the central estimate. We then ask how much cropland the
uncertainty leaves classification-ambiguous (buffer interval straddles a vulnerability
threshold) -- reported as a RANGE over a 7-21 day threshold sweep and a 3-7 mm/day crop
demand band, because the exact fraction is threshold-dependent and a single value would
be misleading.

ETc is illustrative (FAO-56 typical mid-season crop demand); the conclusion is about the
*propagated uncertainty*, not a specific drought forecast.

Out: paper1_hydraulic/eval/m8_downstream.json + figures/data/downstream.csv
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa

GRID = DATA_ROOT / "paper1_hydraulic/interim/global_awc_grid.parquet"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m8_downstream.json"
FIG = Path(__file__).resolve().parents[1] / "figures/data/downstream.csv"
ROOT_MM = 1000.0          # 1 m root zone
ETC = 5.0                 # mm/day, illustrative FAO-56 mid-season crop ET
ETC_LO, ETC_HI = 3.0, 7.0 # sensitivity band
THRESH_DAYS = 14.0        # drought-vulnerability threshold (~2 weeks of buffer)
CROPLAND = 40


def main():
    g = pd.read_parquet(GRID)
    # per-location: mean AWC over depths, mean 90% interval width
    loc = (g.groupby(["lon", "lat"])
             .agg(awc=("awc_pred", "mean"), width=("awc_width", "mean"),
                  reliable=("reliable", "min"), lulc=("lulc_2018", "first"))
             .reset_index())
    crop = loc[loc.lulc == CROPLAND].copy()
    crop = crop[crop.awc > 0]                         # physical
    print(f"cropland locations: {len(crop):,}")

    # plant-available water storage (mm) and its 90% interval
    crop["paws"] = crop.awc * ROOT_MM
    crop["paws_lo"] = (crop.awc - crop.width / 2).clip(lower=0) * ROOT_MM
    crop["paws_hi"] = (crop.awc + crop.width / 2) * ROOT_MM

    out = {"etc_mm_day": ETC, "etc_band": [ETC_LO, ETC_HI],
           "root_mm": ROOT_MM, "threshold_days": THRESH_DAYS,
           "n_cropland": int(len(crop))}

    # buffering days at central ETc
    buf = crop.paws / ETC
    buf_lo = crop.paws_lo / ETC          # high-ET... note low PAWS -> low buffer
    buf_hi = crop.paws_hi / ETC
    out["buffer_days_median"] = round(float(buf.median()), 1)
    out["buffer_days_90pi_median"] = round(float((buf_hi - buf_lo).median()), 1)
    out["buffer_days_iqr"] = [round(float(buf.quantile(.25)), 1), round(float(buf.quantile(.75)), 1)]

    # classification ambiguity is THRESHOLD-DEPENDENT, so we sweep thresholds (and ET)
    # rather than report a single cherry-picked value. 'ambiguous' = the 90% buffer
    # interval straddles the vulnerability threshold (cannot confidently classify).
    thr_sweep = [7, 10, 14, 18, 21]
    sweep = {}
    for thr in thr_sweep:
        sweep[thr] = round(100 * float(((buf_lo < thr) & (buf_hi > thr)).mean()), 1)
    out["ambiguous_by_threshold_days_et5"] = sweep

    # full range across the threshold sweep AND the ET band (the honest envelope)
    amb_all = []
    for et in [ETC_LO, ETC, ETC_HI]:
        blo = crop.paws_lo / et; bhi = crop.paws_hi / et
        for thr in thr_sweep:
            amb_all.append(100 * float(((blo < thr) & (bhi > thr)).mean()))
    out["ambiguous_range_pct"] = [round(min(amb_all), 0), round(max(amb_all), 0)]
    # central illustrative point (14 d, 5 mm/day)
    out["ambiguous_central_14d_et5_pct"] = sweep[14]
    out["central_drought_vulnerable_14d_et5_pct"] = round(100 * float((buf < THRESH_DAYS).mean()), 1)

    # within the RELIABLE domain (AoA), at the central choice -- shows it is not just
    # an extrapolation artefact
    rc = crop[crop.reliable == 1]
    if len(rc) > 50:
        b = rc.paws / ETC; blo = rc.paws_lo / ETC; bhi = rc.paws_hi / ETC
        out["reliable_cropland_n"] = int(len(rc))
        out["reliable_cropland_ambiguous_14d_et5_pct"] = round(
            100 * float(((blo < THRESH_DAYS) & (bhi > THRESH_DAYS)).mean()), 1)

    print(json.dumps(out, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    # figure data: distribution of buffer days + interval, sampled
    s = crop.sample(min(4000, len(crop)), random_state=0)
    pd.DataFrame({
        "buffer": (s.paws / ETC).round(2),
        "buffer_lo": (s.paws_lo / ETC).round(2),
        "buffer_hi": (s.paws_hi / ETC).round(2),
        "reliable": s.reliable.values,
    }).to_csv(FIG, index=False)
    # threshold-sweep table for the figure (ambiguous % vs threshold, per ET)
    rows = []
    for et in [ETC_LO, ETC, ETC_HI]:
        blo = crop.paws_lo / et; bhi = crop.paws_hi / et
        for thr in range(5, 26):
            rows.append({"et": et, "thr": thr,
                         "ambiguous": round(100 * float(((blo < thr) & (bhi > thr)).mean()), 2)})
    pd.DataFrame(rows).to_csv(FIG.with_name("downstream_sweep.csv"), index=False)
    # headline scalars
    flat = {k: v for k, v in out.items() if not isinstance(v, dict)}
    pd.DataFrame([flat]).to_csv(FIG.with_name("downstream_summary.csv"), index=False)
    print(f"-> {OUT}\n-> {FIG}")


if __name__ == "__main__":
    main()
