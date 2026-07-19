"""Paper1 / M1 QA — decisive valid-pixel-fraction scan of reproj outputs.

Stale ISRIC checksums and the rasterio error log both mislead; the physical truth
is coverage. Healthy global-land soilgrids/VWC reproj ~= 0.27 valid fraction.
Anything materially below that has data holes from a truncated source -> re-fetch.

Writes the bad list to /tmp/bad_reproj.txt as "<source>/<relpath>" for re-warp,
and prints a per-file table.
"""
import glob, os, sys
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import INTERIM  # noqa: E402

BASE = str(INTERIM / "reproj")
VARS = ["clay", "sand", "silt", "bdod", "soc", "phh2o", "cec", "nitrogen", "cfvo"]
DEPTHS = ["0-5cm", "15-30cm", "30-60cm", "60-100cm"]   # build_table DEPTH_MAP
THRESH = 0.20   # below this => holey/corrupt (healthy ~0.27)

bad = []
rows = []

def frac(p):
    with rasterio.open(p) as d:
        a = d.read(1, out_shape=(732, 1737), resampling=Resampling.average)  # ~1/20
    return float(((a > -9000) & np.isfinite(a) & (a < 1e6)).mean())

print(f"{'file':52s} valid_frac  status")
# soilgrids
for v in VARS:
    for dp in DEPTHS:
        rel = f"soilgrids_isric/{v}/{v}_{dp}_mean_1000.tif"
        p = f"{BASE}/{rel}"
        if not os.path.exists(p):
            print(f"{rel:52s}   MISSING"); bad.append(rel); continue
        fr = frac(p); ok = fr >= THRESH
        print(f"{rel:52s}   {fr:.3f}     {'ok' if ok else 'BAD'}")
        if not ok: bad.append(rel)
# VWC targets
for f in sorted(glob.glob(f"{BASE}/openlandmap/zenodo_13837179/Global_VWC_*.tif")):
    if os.path.basename(f).startswith("._"): continue
    rel = f.replace(BASE + "/", "")
    fr = frac(f); ok = fr >= THRESH
    print(f"{os.path.basename(f):52s}   {fr:.3f}     {'ok' if ok else 'BAD'}")
    if not ok: bad.append(rel)

with open("/tmp/bad_reproj.txt", "w") as fh:
    fh.write("\n".join(bad) + ("\n" if bad else ""))
print(f"\n{len(bad)} BAD files -> /tmp/bad_reproj.txt")
