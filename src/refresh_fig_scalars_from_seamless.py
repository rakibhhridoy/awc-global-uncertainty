"""Regenerate the two scalar figure CSVs (significance.csv, downstream_summary.csv) from
the SEAMLESS-raster eval JSONs (M13 global raster, M14 downstream), which the seamless
pipeline (global_raster.py / downstream_raster.py) does not itself emit.

Root cause of the drift these CSVs had: only the POINT-sample scripts (global_product.py
=m7, downstream_impact.py=m8) write these two CSVs, so after the seamless recompute they
were left at point values (33% land / 56% cropland / 45.7% ambiguous). fig7/fig8 read
them, so the figures rendered stale numbers. This script brings them to M13/M14 without
re-running the ~5 h raster jobs, since the JSONs already hold every needed scalar.

Run:  python3 src/refresh_fig_scalars_from_seamless.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
FIG = ROOT / "figures" / "data"

m13 = json.loads((EVAL / "m13_global_raster.json").read_text())
m14 = json.loads((EVAL / "m14_downstream_raster.json").read_text())

# --- significance.csv (fig7): seamless coverage + PAWS uncertainty over cropland ---
n_land = m13["n_land_px"]
n_crop = m13["n_cropland_px"]
sig_cols = ["n_grid_locations", "n_cropland_locations", "reliable_frac_all_land",
            "reliable_frac_cropland", "cropland_OUTSIDE_reliable_pct",
            "paws_mm_median_cropland", "paws_pi_mm_median_cropland",
            "paws_pi_as_frac_of_estimate_cropland", "aoa_threshold"]
sig = [n_land, n_crop, m13["reliable_frac_all_land"], m13["reliable_frac_cropland"],
       m13["cropland_OUTSIDE_reliable_pct"], m13["paws_mm_median_cropland"],
       m13["paws_pi_mm_median_cropland"], m14["paws_interval_as_frac_of_estimate"],
       m13["aoa_threshold"]]
(FIG / "significance.csv").write_text(
    ",".join(sig_cols) + "\n" + ",".join(str(v) for v in sig) + "\n")

# --- downstream_summary.csv (fig8 subtitle scalars) ---
reliable_crop_n = round(m13["reliable_frac_cropland"] * m14["n_cropland"])
ds_cols = ["etc_mm_day", "etc_band", "root_mm", "threshold_days", "n_cropland",
           "buffer_days_median", "buffer_days_90pi_median", "buffer_days_iqr",
           "ambiguous_range_pct", "ambiguous_central_14d_et5_pct",
           "central_drought_vulnerable_14d_et5_pct", "reliable_cropland_n",
           "reliable_cropland_ambiguous_14d_et5_pct"]
iqr = f'[{m14["buffer_lo_median"]}, {m14["buffer_hi_median"]}]'
rng = f'[{m14["ambiguous_range_pct"][0]}, {m14["ambiguous_range_pct"][1]}]'
ds = [m14["etc_mm_day"], '"[3.0, 7.0]"', m14["root_mm"], 14.0, m14["n_cropland"],
      m14["buffer_days_median"], m14["buffer_days_90pi_median"], f'"{iqr}"', f'"{rng}"',
      m14["ambiguous_central_14d_et5_pct"], m14["central_drought_vulnerable_14d_et5_pct"],
      reliable_crop_n, m14["reliable_cropland_ambiguous_14d_et5_pct"]]
(FIG / "downstream_summary.csv").write_text(
    ",".join(ds_cols) + "\n" + ",".join(str(v) for v in ds) + "\n")

print("significance.csv       :", m13["reliable_frac_cropland"], "cropland reliable,",
      m13["cropland_OUTSIDE_reliable_pct"], "% outside")
print("downstream_summary.csv :", m14["ambiguous_central_14d_et5_pct"], "% ambiguous @14d,",
      m14["buffer_days_median"], "d buffer")
