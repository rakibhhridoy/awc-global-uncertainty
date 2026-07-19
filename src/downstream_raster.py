"""Paper 1 / M14 -- downstream agronomic significance recomputed on the SEAMLESS raster.

Replaces the sample-based M8 (drought-buffering) and M10 (climatic water balance) with
full-grid statistics from the M13 product, so the manuscript's downstream numbers are
consistent with the released raster and its cross-conformal intervals. Uses the direct
asymmetric conformal bounds (awc_lo / awc_hi) rather than a symmetric width/2.

Cropland = Copernicus discrete class 40. Reads product + CHELSA (bio2/bio9/bio17) +
pixel latitude, block by block, accumulates cropland pixels, then computes:
  M8  : drought-buffering time PAWS/ETc, its 90% interval, threshold-sweep ambiguity.
  M10 : Hargreaves dry-season water balance, % water-limited, % flipped by AWC uncertainty.

Out: paper1_hydraulic/eval/m14_downstream_raster.json + refreshed figures/data/{downstream,
waterbalance}*.csv
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import rasterio
from rasterio.warp import transform as warp_xy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT, INTERIM  # noqa

P = DATA_ROOT / "paper1_hydraulic/product"
REPROJ = INTERIM / "reproj"
LULC = REPROJ / "copernicus_lulc" / "2018" / "PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif"
CHELSA = lambda b: REPROJ / "chelsa" / f"CHELSA_bio{b}_1981-2010_V.2.1.tif"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m14_downstream_raster.json"
FIGDIR = Path(__file__).resolve().parents[1] / "figures/data"
CROPLAND, ROOT_MM, DRY_DAYS, HARG_K = 40, 1000.0, 90.0, 0.0023
ETC, ETC_LO, ETC_HI, THRESH = 5.0, 3.0, 7.0, 14.0
BLOCK = 512


def annual_mean_Ra(lat_deg):
    phi = np.radians(np.clip(lat_deg, -66.5, 66.5)); Gsc = 0.0820
    tot = np.zeros_like(phi, dtype=float)
    for J in range(1, 366):
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
        dec = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
        ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
        Ra = (24 * 60 / np.pi) * Gsc * dr * (ws * np.sin(phi) * np.sin(dec)
              + np.cos(phi) * np.cos(dec) * np.sin(ws))
        tot += Ra
    return (tot / 365) * 0.408


def main():
    ds = {n: rasterio.open(P / f"{n}.tif") for n in ["awc_mean", "awc_lo", "awc_hi", "reliable"]}
    lulc_ds = rasterio.open(LULC)
    b2, b9, b17 = rasterio.open(CHELSA(2)), rasterio.open(CHELSA(9)), rasterio.open(CHELSA(17))
    a = ds["awc_mean"]; H, W, T, crs, nd = a.height, a.width, a.transform, a.crs, a.nodata

    cols = {k: [] for k in ["paws", "paws_lo", "paws_hi", "deficit", "reliable",
                            "pet_ann", "def_k20", "def_k26"]}
    for r0 in range(0, H, BLOCK):
        nr = min(BLOCK, H - r0); win = rasterio.windows.Window(0, r0, W, nr)
        awc = ds["awc_mean"].read(1, window=win)
        T2 = b2.read(1, window=win).astype("float64"); T9 = b9.read(1, window=win).astype("float64")
        P17 = b17.read(1, window=win).astype("float64")
        CND = -9999.0                       # CHELSA nodata: exclude, don't let it dilute
        clim_ok = (T2 != CND) & (T9 != CND) & (P17 != CND)
        crop = (lulc_ds.read(1, window=win) == CROPLAND) & (awc != nd) & (awc > 0) & clim_ok
        if not crop.any():
            continue
        lo = ds["awc_lo"].read(1, window=win); hi = ds["awc_hi"].read(1, window=win)
        rel = ds["reliable"].read(1, window=win)
        rr, cc = np.where(crop)
        ys = T.f + (r0 + rr + 0.5) * T.e; xs = T.c + (cc + 0.5) * T.a
        _, lat = warp_xy(crs, "EPSG:4326", xs.tolist(), ys.tolist()); lat = np.asarray(lat)
        awc_c = awc[rr, cc]; lo_c = np.maximum(lo[rr, cc], 0); hi_c = hi[rr, cc]
        # CHELSA v2.1 units: temp = v*0.1-273.15 degC ; precip = v*0.1 mm
        Tdry = T9[rr, cc] * 0.1 - 273.15
        TD = np.clip(T2[rr, cc] * 0.1, 0.1, None)
        Pdry = np.clip(P17[rr, cc] * 0.1, 0, None)
        Ra = annual_mean_Ra(lat)
        pet = np.clip(HARG_K * Ra * (Tdry + 17.8) * np.sqrt(TD), 0, None) * DRY_DAYS
        pet_ann = np.clip(HARG_K * Ra * (Tdry + 17.8) * np.sqrt(TD), 0, None) * 365
        pet_k = {k: np.clip(k * Ra * (Tdry + 17.8) * np.sqrt(TD), 0, None) * DRY_DAYS
                 for k in (0.0020, 0.0026)}
        cols["paws"].append(awc_c * ROOT_MM)
        cols["paws_lo"].append(lo_c * ROOT_MM); cols["paws_hi"].append(hi_c * ROOT_MM)
        cols["deficit"].append(np.clip(pet - Pdry, 0, None))
        cols["pet_ann"].append(pet_ann)
        cols["def_k20"].append(np.clip(pet_k[0.0020] - Pdry, 0, None))
        cols["def_k26"].append(np.clip(pet_k[0.0026] - Pdry, 0, None))
        cols["reliable"].append((rel[rr, cc] == 1).astype("uint8"))
        if (r0 // BLOCK) % 5 == 0:
            print(f"  rows {r0}/{H} ({100*r0/H:.0f}%)  cropland px so far={sum(len(x) for x in cols['paws']):,}")

    for k in cols: cols[k] = np.concatenate(cols[k])
    paws, plo, phi_, deficit, rel = (cols["paws"], cols["paws_lo"], cols["paws_hi"],
                                     cols["deficit"], cols["reliable"])
    pet_ann, def_k20, def_k26 = cols["pet_ann"], cols["def_k20"], cols["def_k26"]
    n = len(paws); print(f"cropland pixels: {n:,}")

    # --- M8: drought-buffering ---
    buf = paws / ETC; blo = plo / ETC; bhi = phi_ / ETC
    sweep = {int(thr): round(100 * float(((blo < thr) & (bhi > thr)).mean()), 1)
             for thr in [7, 10, 14, 18, 21]}
    # extra manuscript-cited scalars (annual PET, buffer range, interval mm, Hargreaves band)
    extra = {
        "buffer_lo_median": round(float(np.median(blo)), 1),
        "buffer_hi_median": round(float(np.median(bhi)), 1),
        "paws_interval_mm_median": round(float(np.median(phi_ - plo)), 0),
        "paws_half_interval_mm_median": round(float(np.median((phi_ - plo) / 2)), 0),
    }
    amb_all = [100 * float((( (plo/et) < thr) & ((phi_/et) > thr)).mean())
               for et in [ETC_LO, ETC, ETC_HI] for thr in [7, 10, 14, 18, 21]]
    rc = rel == 1
    out = {"n_cropland": int(n), "etc_mm_day": ETC, "root_mm": ROOT_MM,
           "buffer_days_median": round(float(np.median(buf)), 1),
           "buffer_days_90pi_median": round(float(np.median(bhi - blo)), 1),
           "ambiguous_by_threshold_days_et5": sweep,
           "ambiguous_range_pct": [round(min(amb_all)), round(max(amb_all))],
           "ambiguous_central_14d_et5_pct": sweep[14],
           "central_drought_vulnerable_14d_et5_pct": round(100 * float((buf < THRESH).mean()), 1),
           "reliable_cropland_ambiguous_14d_et5_pct":
               round(100 * float(((blo[rc] < THRESH) & (bhi[rc] > THRESH)).mean()), 1),
           # --- M10: climatic water balance ---
           "PAWS_mm_median": round(float(np.median(paws)), 0),
           "PET_annual_mm_median": round(float(np.median(pet_ann)), 0),
           "dry_season_deficit_mm_median": round(float(np.median(deficit)), 0),
           "cropland_water_limited_central_pct": round(100 * float((deficit > paws).mean()), 1),
           "cropland_flipped_by_AWC_uncertainty_pct":
               round(100 * float(((deficit > plo) & (deficit < phi_)).mean()), 1),
           "reliable_cropland_flipped_pct":
               round(100 * float(((deficit[rc] > plo[rc]) & (deficit[rc] < phi_[rc])).mean()), 1),
           "flipped_pct_k0020": round(100 * float(((def_k20 > plo) & (def_k20 < phi_)).mean()), 1),
           "flipped_pct_k0026": round(100 * float(((def_k26 > plo) & (def_k26 < phi_)).mean()), 1),
           "paws_interval_as_frac_of_estimate":
               round(float(np.median((phi_ - plo) / np.maximum(paws, 1e-6))), 3),
           **extra}
    print(json.dumps(out, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    # refresh figure CSVs (sampled) so Fig 7 / waterbalance reflect the raster
    rng = np.random.default_rng(0); idx = rng.choice(n, min(4000, n), replace=False)
    pd.DataFrame({"buffer": np.round(buf[idx], 2), "buffer_lo": np.round(blo[idx], 2),
                  "buffer_hi": np.round(bhi[idx], 2), "reliable": rel[idx]}
                 ).to_csv(FIGDIR / "downstream.csv", index=False)
    rows = [{"et": et, "thr": thr,
             "ambiguous": round(100 * float((((plo/et) < thr) & ((phi_/et) > thr)).mean()), 2)}
            for et in [ETC_LO, ETC, ETC_HI] for thr in range(5, 26)]
    pd.DataFrame(rows).to_csv(FIGDIR / "downstream_sweep.csv", index=False)
    pd.DataFrame({"deficit": np.round(deficit[idx], 1), "paws": np.round(paws[idx], 1),
                  "paws_lo": np.round(plo[idx], 1), "paws_hi": np.round(phi_[idx], 1),
                  "reliable": rel[idx]}).to_csv(FIGDIR / "waterbalance.csv", index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
