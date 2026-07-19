"""Paper 1 / M1 — reproject the CORE predictors + VWC targets to EPSG:6933 1 km.

Focused sweep (not the repo-wide reproject_static.py): only the global single-raster
sources Paper 1 needs, skipping SOC/gsocmap/hwsd and the WorldClim zips (unextracted)
and MERIT tiles (308 regional tiles — handled separately in M1b via VRT mosaic).

Sources: VWC targets (Zenodo 13837179), SoilGrids 1 km, CHELSA, Copernicus LULC.
Idempotent: skips outputs already present. Run on Mac.

Run:  python Paper1/src/reproject_paper1.py
Out:  {DATA_ROOT}/interim/reproj/<source>/<rel>.tif
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import RAW, INTERIM, warp_to_grid, LOGS  # noqa: E402

# (source dir, resampling, glob, optional file-name filter)
JOBS = [
    ("openlandmap",     "bilinear", "zenodo_13837179/Global_VWC_*.tif"),  # targets
    ("soilgrids_isric", "bilinear", "**/*_1000.tif"),                     # predictors
    ("chelsa",          "bilinear", "**/*.tif"),                          # climate
    ("copernicus_lulc", "nearest",  "**/*.tif"),                          # land cover
]


def main():
    log = (LOGS / "paper1_reproject.log").open("a")
    def out(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()

    out(f"=== Paper1 reproject start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    n_ok = n_skip = n_err = 0
    for source, resamp, pattern in JOBS:
        src_dir = RAW / source
        if not src_dir.exists():
            out(f"MISS source dir {source}"); continue
        files = [f for f in sorted(src_dir.glob(pattern)) if not f.name.startswith("._")]
        out(f"-- {source}: {len(files)} files ({resamp})")
        for f in files:
            rel = f.relative_to(src_dir)
            dst = INTERIM / "reproj" / source / rel
            if dst.exists():
                n_skip += 1; continue
            try:
                t0 = time.time()
                warp_to_grid(f, dst, resampling=resamp)
                n_ok += 1
                out(f"OK  {source}/{f.name}  ({time.time()-t0:.0f}s)")
            except Exception as e:
                n_err += 1
                out(f"ERR {source}/{f.name}: {e}")
    out(f"=== done: {n_ok} warped, {n_skip} skipped, {n_err} errors ===")
    log.close()


if __name__ == "__main__":
    main()
