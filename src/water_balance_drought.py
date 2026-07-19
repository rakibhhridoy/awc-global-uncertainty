"""Paper 1 / M10 — climatic water-balance dry-season drought index (replaces the toy bucket).

The earlier downstream metric assumed constant crop demand and no rain. Here we use a real
climatic water balance over the dry season, with spatially varying demand and supply:

  PET (Hargreaves):  PET_day = 0.0023 * Ra * (T + 17.8) * sqrt(TD)        [mm/day]
     Ra  = annual-mean extraterrestrial radiation from latitude (FAO-56), in mm/day
     T   = mean temperature of the driest quarter (CHELSA bio9)
     TD  = mean diurnal temperature range (CHELSA bio2), proxy for Tmax-Tmin
  Dry season  = driest quarter (~90 days).
  Supply      = precipitation of the driest quarter (CHELSA bio17).
  Climatic water deficit the soil must buffer:  D = max(0, PET_dry - P_dry)   [mm]
  Plant-available water storage:                PAWS = AWC * root_depth (1 m) [mm]
  A cropland soil is dry-season water-limited if D > PAWS.

We then propagate the 90% prediction interval on AWC through PAWS and ask: for how much
cropland does the AWC uncertainty alone change whether the soil is classified as
dry-season water-limited? Because AWC enters linearly via PAWS, the propagation is exact.

CHELSA v2.1 unit conventions (verified physical): temperature = value*0.1 - 273.15 (deg C);
precipitation = value*0.1 (mm). PET estimation is approximate (Hargreaves, annual-mean Ra);
we report a sensitivity band on the PET coefficient.

Out: paper1_hydraulic/eval/m10_waterbalance.json + figures/data/waterbalance.csv
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT  # noqa

GRID = DATA_ROOT / "paper1_hydraulic/interim/global_awc_grid.parquet"
TRAIN = DATA_ROOT / "paper1_hydraulic/interim/train_table.parquet"
OUT = DATA_ROOT / "paper1_hydraulic/eval/m10_waterbalance.json"
FIG = Path(__file__).resolve().parents[1] / "figures/data/waterbalance.csv"
CROPLAND = 40
ROOT_MM = 1000.0
DRY_DAYS = 90.0
HARG_K = 0.0023            # Hargreaves coefficient (central); sensitivity 0.0020-0.0026


def annual_mean_Ra(lat_deg):
    """FAO-56 extraterrestrial radiation, averaged over the year, per latitude (mm/day)."""
    phi = np.radians(np.clip(lat_deg, -66.5, 66.5))
    Gsc = 0.0820
    days = np.arange(1, 366)
    tot = np.zeros_like(phi, dtype=float)
    for J in days:
        dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
        dec = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
        x = -np.tan(phi) * np.tan(dec)
        x = np.clip(x, -1, 1)
        ws = np.arccos(x)
        Ra = (24 * 60 / np.pi) * Gsc * dr * (
            ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
        tot += Ra
    Ra_MJ = tot / len(days)            # MJ m-2 day-1
    return Ra_MJ * 0.408               # -> mm/day equivalent


def chelsa_col(df, b):
    c = [x for x in df.columns if f"CHELSA_{b}_" in x]
    return df[c[0]] if c else None


def main():
    g = pd.read_parquet(GRID)
    t = pd.read_parquet(TRAIN)
    # climate per location from the train table (same sample); convert CHELSA units
    clim = pd.DataFrame({"lon": t.lon, "lat": t.lat})
    clim["T_dry"]  = chelsa_col(t, "bio9")  * 0.1 - 273.15     # deg C, driest quarter
    clim["TD"]     = (chelsa_col(t, "bio2") * 0.1).clip(lower=0.1)  # deg C diurnal range
    clim["P_dry"]  = (chelsa_col(t, "bio17") * 0.1).clip(lower=0)   # mm, driest quarter
    clim = clim[(chelsa_col(t, "bio12") > 0) & (chelsa_col(t, "bio12") < 65000)]
    clim = clim.dropna().drop_duplicates(["lon", "lat"])

    # AWC + interval per location (mean over depths), cropland only
    loc = (g.groupby(["lon", "lat"]).agg(
            awc=("awc_pred", "mean"), w=("awc_width", "mean"),
            reliable=("reliable", "min"), lulc=("lulc_2018", "first")).reset_index())
    d = loc.merge(clim, on=["lon", "lat"], how="inner")
    crop = d[(d.lulc == CROPLAND) & (d.awc > 0)].copy()
    print(f"cropland locations with climate: {len(crop):,}")

    crop["Ra"] = annual_mean_Ra(crop.lat.values)
    def pet_dry(k):
        pet_day = k * crop.Ra * (crop.T_dry + 17.8) * np.sqrt(crop.TD)
        return (pet_day.clip(lower=0) * DRY_DAYS)
    crop["PET_dry"] = pet_dry(HARG_K)
    crop["deficit"] = (crop.PET_dry - crop.P_dry).clip(lower=0)
    crop["PAWS"]    = crop.awc * ROOT_MM
    crop["PAWS_lo"] = (crop.awc - crop.w / 2).clip(lower=0) * ROOT_MM
    crop["PAWS_hi"] = (crop.awc + crop.w / 2) * ROOT_MM

    out = {"root_mm": ROOT_MM, "dry_days": DRY_DAYS, "harg_k": HARG_K,
           "n_cropland": int(len(crop))}
    # sanity: PET in physically plausible global range
    pet_ann = (HARG_K * crop.Ra * (crop.T_dry + 17.8) * np.sqrt(crop.TD) * 365).clip(lower=0)
    out["PET_annual_mm_median"] = round(float(pet_ann.median()), 0)
    out["dry_season_PET_mm_median"] = round(float(crop.PET_dry.median()), 0)
    out["dry_season_precip_mm_median"] = round(float(crop.P_dry.median()), 0)
    out["dry_season_deficit_mm_median"] = round(float(crop.deficit.median()), 0)
    out["PAWS_mm_median"] = round(float(crop.PAWS.median()), 0)

    # classification: is the soil dry-season water-limited (deficit exceeds storage)?
    central_limited = crop.deficit > crop.PAWS
    out["cropland_water_limited_central_pct"] = round(100 * float(central_limited.mean()), 1)
    # AWC uncertainty flips the classification where the deficit sits between PAWS_lo and PAWS_hi
    ambiguous = (crop.deficit > crop.PAWS_lo) & (crop.deficit < crop.PAWS_hi)
    out["cropland_classification_flipped_by_AWC_uncertainty_pct"] = round(100 * float(ambiguous.mean()), 1)
    # within reliable domain
    rc = crop[crop.reliable == 1]
    if len(rc) > 50:
        amb_r = (rc.deficit > rc.PAWS_lo) & (rc.deficit < rc.PAWS_hi)
        out["reliable_cropland_flipped_pct"] = round(100 * float(amb_r.mean()), 1)
    # PET-coefficient sensitivity band (k = 0.0020 and 0.0026)
    for k, tag in [(0.0020, "k0020"), (0.0026, "k0026")]:
        pet = pet_dry(k); defc = (pet - crop.P_dry).clip(lower=0)
        amb = (defc > crop.PAWS_lo) & (defc < crop.PAWS_hi)
        out[f"flipped_pct_{tag}"] = round(100 * float(amb.mean()), 1)

    print(json.dumps(out, indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    s = crop.sample(min(4000, len(crop)), random_state=0)
    pd.DataFrame({
        "deficit": s.deficit.round(1), "paws": s.PAWS.round(1),
        "paws_lo": s.PAWS_lo.round(1), "paws_hi": s.PAWS_hi.round(1),
        "reliable": s.reliable.values,
    }).to_csv(FIG, index=False)
    print(f"-> {OUT}\n-> {FIG}")


if __name__ == "__main__":
    main()
