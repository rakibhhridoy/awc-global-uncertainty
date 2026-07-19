"""Paper 1 — self-contained grid + reprojection helpers.

Vendored so Paper1 is a standalone repo (no reach into the parent ~/Soil tree).
Target grid spec lives in Paper1/configs/grid.yaml (EPSG:6933, 1 km global).

Data paths resolve via PEDOFLUX_DATA (default the external SSD); raw/interim live
there and are gitignored. Only code/config/docs are tracked.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                       # Paper1/
DATA_ROOT = Path(os.environ.get("PEDOFLUX_DATA", "/Volumes/SSD Ex/PEDOFLUX_data"))
RAW = DATA_ROOT / "raw"
INTERIM = DATA_ROOT / "interim"
LOGS = DATA_ROOT / "logs"
for p in (INTERIM, LOGS):
    p.mkdir(parents=True, exist_ok=True)


def load_grid():
    with (REPO / "configs" / "grid.yaml").open() as f:
        return yaml.safe_load(f)


def target_profile():
    """rasterio-style profile for the target EPSG:6933 1 km grid."""
    g = load_grid()["target"]
    res = g["resolution_m"]
    b = g["bounds"]
    from rasterio.transform import from_origin
    return {
        "crs": g["crs"],
        "transform": from_origin(b["xmin"], b["ymax"], res, res),
        "width": int(round((b["xmax"] - b["xmin"]) / res)),
        "height": int(round((b["ymax"] - b["ymin"]) / res)),
        "nodata": g["nodata"]["float32"],
    }


def warp_to_grid(src_path: Path, dst_path: Path, resampling: str = "bilinear",
                 dtype: str = "float32") -> Path:
    """Reproject any raster onto the Paper1 common grid (zstd COG, 512 tiles)."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    tp = target_profile()
    r_map = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest,
             "average": Resampling.average, "cubic": Resampling.cubic}
    resamp = r_map[resampling]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        profile = src.profile
        profile.update(
            crs=tp["crs"], transform=tp["transform"],
            width=tp["width"], height=tp["height"],
            dtype=dtype, nodata=tp["nodata"], compress="zstd", tiled=True,
            blockxsize=512, blockysize=512,
        )
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=tp["transform"], dst_crs=tp["crs"],
                    resampling=resamp,
                )
    return dst_path
