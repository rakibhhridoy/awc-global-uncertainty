"""Paper 1 / M4 — build observed soil-moisture climatology from ESA-CCI.

Streams all daily ESA-CCI SM files (2010-2014, 0.25 deg global), masks by the
quality flag, and accumulates per-pixel mean / std / valid-count without holding
the stack in memory. Writes a small native-grid GeoTIFF (3 bands), then reprojects
to the Paper1 common grid (EPSG:6933 1 km) for confrontation with predicted
hydraulic properties.

ESA-CCI `sm` is SURFACE volumetric soil moisture (m3/m3) — the same physical
quantity as our VWC targets, so it can be confronted directly with the 0 cm layer.

Out (on $PEDOFLUX_DATA):
  paper1_hydraulic/interim/esacci_sm_climatology_025.tif   (mean, std, n; EPSG:4326)
  interim/reproj/esacci/esacci_sm_{mean,std,n}.tif          (EPSG:6933 1 km)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import RAW, INTERIM, DATA_ROOT, warp_to_grid  # noqa: E402

ES = RAW / "esacci_sm"
OUT_025 = DATA_ROOT / "paper1_hydraulic/interim/esacci_sm_climatology_025.tif"
REPROJ = INTERIM / "reproj" / "esacci"
NODATA = -9999.0


def daily_files():
    fs = []
    for y in ["2010", "2011", "2012", "2013", "2014"]:
        fs += [p for p in sorted((ES / y).glob("*.nc")) if not p.name.startswith("._")]
    return fs


def main():
    files = daily_files()
    print(f"streaming {len(files)} daily ESA-CCI files ...", flush=True)

    s = s2 = n = None        # running sum, sum-of-squares, count (720x1440 float64)
    lat_desc = None
    for i, f in enumerate(files):
        try:
            ds = xr.open_dataset(f)
        except Exception as e:
            print(f"  skip {f.name}: {e}"); continue
        sm = ds["sm"].isel(time=0).values.astype("float64")     # (lat, lon)
        flag = ds["flag"].isel(time=0).values
        if lat_desc is None:
            lat_desc = float(ds.lat.values[0]) > float(ds.lat.values[-1])
        ds.close()
        valid = np.isfinite(sm) & (flag == 0)                   # flag==0 -> clean retrieval
        sm = np.where(valid, sm, 0.0)
        v = valid.astype("float64")
        if s is None:
            s = np.zeros_like(sm); s2 = np.zeros_like(sm); n = np.zeros_like(sm)
        s += sm; s2 += sm * sm; n += v
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / n, NODATA)
        var = np.where(n > 0, s2 / n - (s / n) ** 2, np.nan)
        std = np.where(n > 0, np.sqrt(np.clip(var, 0, None)), NODATA)
    n_out = np.where(n > 0, n, NODATA)

    # ensure north-up for GeoTIFF (lat descending = row 0 is north)
    if not lat_desc:
        mean, std, n_out = mean[::-1], std[::-1], n_out[::-1]
    transform = from_origin(-180.0, 90.0, 0.25, 0.25)           # 1440x720 global

    OUT_025.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=720, width=1440, count=3, dtype="float32",
                crs="EPSG:4326", transform=transform, nodata=NODATA,
                compress="zstd", tiled=True)
    with rasterio.open(OUT_025, "w", **prof) as dst:
        dst.write(mean.astype("float32"), 1)
        dst.write(std.astype("float32"), 2)
        dst.write(n_out.astype("float32"), 3)
        dst.set_band_description(1, "sm_mean"); dst.set_band_description(2, "sm_std")
        dst.set_band_description(3, "n_obs")
    valid_frac = float((n > 0).mean())
    print(f"wrote {OUT_025}  (valid pixels {valid_frac:.2f}, "
          f"mean SM range {np.nanmin(mean[mean>NODATA]):.3f}..{np.nanmax(mean):.3f})")

    # reproject each band to the common grid (bilinear; coarse->fine upsample)
    REPROJ.mkdir(parents=True, exist_ok=True)
    for band, name in [(1, "mean"), (2, "std"), (3, "n")]:
        # warp_to_grid works per-file; write a temp single-band then warp
        tmp = OUT_025.parent / f"_esacci_{name}.tif"
        with rasterio.open(OUT_025) as src:
            p = src.profile; p.update(count=1)
            with rasterio.open(tmp, "w", **p) as t:
                t.write(src.read(band), 1)
        warp_to_grid(tmp, REPROJ / f"esacci_sm_{name}.tif", resampling="bilinear")
        tmp.unlink()
    print(f"reprojected to {REPROJ}")


if __name__ == "__main__":
    main()
