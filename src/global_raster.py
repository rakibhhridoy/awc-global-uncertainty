"""Paper 1 / M13 -- seamless global AWC raster with honest, conformal uncertainty.

Extends the sampled global product (M7) to a full 1 km raster (EASE-Grid 2.0,
EPSG:6933, 34735x14629). For every land pixel it predicts root-zone available water
capacity (AWC, mean over 0/30/60/100 cm), a dissimilarity-conditional (Mondrian) 90%
prediction interval, the dissimilarity index (DI), and an area-of-applicability
reliability flag -- the released product that turns the paper's critique into a resource
that says WHERE it can be trusted.

Method (mirrors M5/M7/M12; all trained on ALL measured data to match the manuscript):
  * Point + 5th/95th quantile histogram-GBT, fitted on all measured layers.
  * DI = nearest-training-neighbour distance / mean pairwise training distance, in
    standardised predictor space. AoA threshold = Q3 + 1.5 IQR of the de-duplicated
    ALL-data DI (~0.287, matching M7/the manuscript; a train-subset threshold would
    loosen it and overstate reliability). A pixel is reliable only if inside the AoA at
    ALL four depths.
  * Intervals are conformalized (CQR) with a Mondrian (per-DI-stratum) correction. Because
    the final quantile models use all data, the correction is calibrated on CROSS-CONFORMAL
    out-of-fold residuals (5-fold), so coverage stays valid; intervals widen out of domain
    and keep ~90% coverage (Table S13). Output bounds are the depth-mean of the corrected
    per-depth bounds.

Outputs (GeoTIFF, LZW, tiled) under {DATA_ROOT}/paper1_hydraulic/product/:
  awc_mean, awc_lo, awc_hi, awc_width, di, reliable, paws_mm
plus product_summary.json (global + cropland reliable fractions; cross-checks M7).

Run:  python src/global_raster.py [--block-rows 128] [--max-blocks N (smoke test)]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harmonize import DATA_ROOT, INTERIM  # noqa

MEAS = DATA_ROOT / "paper1_hydraulic/interim/measured_table.parquet"
GRID = DATA_ROOT / "paper1_hydraulic/interim/train_table.parquet"
REPROJ = INTERIM / "reproj"
OUTDIR = DATA_ROOT / "paper1_hydraulic/product"
SUMMARY = DATA_ROOT / "paper1_hydraulic/eval/m13_global_raster.json"
SEED = 0
ALPHA = 0.10
DI_BINS = 5
ROOTZONE_MM = 1000.0
CROPLAND_CLASS = 40
DEPTH_MAP = {0: "0-5cm", 30: "15-30cm", 60: "30-60cm", 100: "60-100cm"}
SG_VARS = ["clay", "sand", "silt", "bdod", "soc", "phh2o", "cec", "nitrogen", "cfvo"]
LULC2018 = REPROJ / "copernicus_lulc" / "2018" / "PROBAV_LC100_global_v3.0.1_2018-conso_Discrete-Classification-map_EPSG-4326.tif"
LAYERS = ["awc_mean", "awc_lo", "awc_hi", "awc_width", "di", "paws_mm"]  # float32
NODATA = -9999.0


def gbt(**kw):
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_iter=400, max_depth=8,
                         learning_rate=0.06, random_state=SEED, **kw))


def feat_to_raster(feat: str, depth_cm: int) -> Path:
    if feat.startswith("sg_"):
        var = feat[3:]
        return REPROJ / "soilgrids_isric" / var / f"{var}_{DEPTH_MAP[depth_cm]}_mean_1000.tif"
    if feat.startswith("chelsa__"):
        return REPROJ / "chelsa" / f"{feat.split('chelsa__', 1)[1]}.tif"
    raise ValueError(feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-rows", type=int, default=128)
    ap.add_argument("--max-blocks", type=int, default=0, help="0 = whole grid; >0 = smoke test")
    args = ap.parse_args()
    t0 = time.time()

    m = pd.read_parquet(MEAS)
    gcols = pd.read_parquet(GRID, columns=None).columns
    feats = [c for c in m.columns if c.startswith("sg_") and c != "sg_label"] + \
            sorted(c for c in m.columns if c.startswith("chelsa__CHELSA_bio")) + ["depth_cm"]
    feats = [c for c in feats if c in gcols]
    sg_feats = [f for f in feats if f.startswith("sg_")]
    clim_feats = [f for f in feats if f.startswith("chelsa__")]
    print(f"features: {len(feats)} ({len(sg_feats)} soil x depth + {len(clim_feats)} clim + depth)")

    d = m[np.isfinite(m.meas_awc)].copy().reset_index(drop=True)
    y = d.meas_awc.values
    rng = np.random.default_rng(SEED)
    Xall = d[feats]

    # --- final models on ALL measured data (best product; matches M7/manuscript) ---
    point = gbt().fit(Xall, y)
    q_lo = gbt(loss="quantile", quantile=ALPHA / 2).fit(Xall, y)
    q_hi = gbt(loss="quantile", quantile=1 - ALPHA / 2).fit(Xall, y)

    # --- DI reference + AoA threshold from ALL de-duplicated data ---
    # (must match the manuscript: threshold ~0.287. Using a train subset would
    #  loosen it and overstate reliability, the very error the paper critiques.)
    sc = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(Xall)
    Xall_s = sc.transform(Xall)
    smp = Xall_s[rng.choice(len(Xall_s), size=2000, replace=False)]
    dbar = float(np.mean(np.linalg.norm(smp[:, None] - smp[None, :], axis=2)))
    dd = d.assign(_k=d.profile_id.astype(str) + "_" + d.depth_cm.astype(str)).drop_duplicates("_k")
    Xdd = sc.transform(dd[feats])
    nn = cKDTree(Xdd).query(Xdd, k=2)[0][:, 1] / dbar
    q1, q3 = np.quantile(nn, [0.25, 0.75]); thr = float(q3 + 1.5 * (q3 - q1))
    tree = cKDTree(Xall_s)

    # --- Mondrian conformal Q via CROSS-CONFORMAL (out-of-fold residuals) ---
    # Final models use all data, so a valid coverage guarantee needs residuals from
    # models that did NOT see each calibration point: K-fold OOF conformity scores,
    # binned by the all-data DI (consistent with the grid DI reference).
    from sklearn.model_selection import KFold
    E = np.empty(len(d))
    for tr_i, te_i in KFold(5, shuffle=True, random_state=SEED).split(Xall):
        lo_f = gbt(loss="quantile", quantile=ALPHA / 2).fit(Xall.iloc[tr_i], y[tr_i]).predict(Xall.iloc[te_i])
        hi_f = gbt(loss="quantile", quantile=1 - ALPHA / 2).fit(Xall.iloc[tr_i], y[tr_i]).predict(Xall.iloc[te_i])
        E[te_i] = np.maximum(lo_f - y[te_i], y[te_i] - hi_f)
    di_cal = tree.query(Xall_s, k=2)[0][:, 1] / dbar        # k=2: exclude self (cal pts are training pts)
    n = len(E); Qglobal = np.sort(E)[min(max(int(np.ceil((1 - ALPHA) * (n + 1))), 1), n) - 1]
    edges = np.quantile(di_cal, np.linspace(0, 1, DI_BINS + 1)); edges[0], edges[-1] = -np.inf, np.inf
    Qbin = np.empty(DI_BINS)
    for b in range(DI_BINS):
        mb = (di_cal >= edges[b]) & (di_cal < edges[b + 1])
        if mb.sum() >= 20:
            Eb = np.sort(E[mb]); nb = len(Eb)
            Qbin[b] = Eb[min(max(int(np.ceil((1 - ALPHA) * (nb + 1))), 1), nb) - 1]
        else:
            Qbin[b] = Qglobal
    Qbin = np.maximum.accumulate(Qbin)
    print(f"AoA threshold={thr:.4f} (target ~0.287) | Mondrian Q per DI-bin={np.round(Qbin,3).tolist()}")

    # --- open inputs (one handle each) ---
    with rasterio.open(feat_to_raster("sg_clay", 0)) as ref:
        profile = ref.profile; H, W = ref.height, ref.width
    sg_ds = {(f, dep): rasterio.open(feat_to_raster(f, dep)) for f in sg_feats for dep in DEPTH_MAP}
    clim_ds = {f: rasterio.open(feat_to_raster(f, 0)) for f in clim_feats}
    lulc_ds = rasterio.open(LULC2018)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    oprof = profile.copy(); oprof.update(dtype="float32", count=1, nodata=NODATA,
                                         compress="lzw", tiled=True, blockxsize=256, blockysize=256)
    rprof = oprof.copy(); rprof.update(dtype="uint8", nodata=255)
    out = {name: rasterio.open(OUTDIR / f"{name}.tif", "w", **oprof) for name in LAYERS}
    out["reliable"] = rasterio.open(OUTDIR / "reliable.tif", "w", **rprof)

    depths = list(DEPTH_MAP)
    n_land = n_rel = n_crop = n_crop_rel = 0
    paws_crop, pawspi_crop = [], []
    nblocks = (H + args.block_rows - 1) // args.block_rows
    if args.max_blocks: nblocks = min(nblocks, args.max_blocks)

    for bi in range(nblocks):
        r0 = bi * args.block_rows; nr = min(args.block_rows, H - r0)
        win = Window(0, r0, W, nr); npx = nr * W
        clim = {f: clim_ds[f].read(1, window=win).reshape(-1).astype("float32") for f in clim_feats}
        for f in clim_feats:
            v = clim[f]; nd = clim_ds[f].nodata
            if nd is not None: v[v == nd] = np.nan
        lulc = lulc_ds.read(1, window=win).reshape(-1)

        awc_sum = np.zeros(npx); lo_sum = np.zeros(npx); hi_sum = np.zeros(npx)
        di_sum = np.zeros(npx); all_reliable = np.ones(npx, bool); any_valid = np.zeros(npx, bool)
        for dep in depths:
            cols = {}
            ok = np.ones(npx, bool)
            for f in sg_feats:
                ds = sg_ds[(f, dep)]; v = ds.read(1, window=win).reshape(-1).astype("float32")
                nd = ds.nodata
                if nd is not None: v[v == nd] = np.nan
                cols[f] = v; ok &= np.isfinite(v)
            for f in clim_feats:
                ok &= np.isfinite(clim[f])
            if ok.sum() == 0:
                all_reliable &= False; continue
            X = np.empty((ok.sum(), len(feats)), "float32")
            for j, f in enumerate(feats):
                if f == "depth_cm": X[:, j] = dep
                elif f.startswith("sg_"): X[:, j] = cols[f][ok]
                else: X[:, j] = clim[f][ok]
            Xdf = pd.DataFrame(X, columns=feats)
            p = point.predict(Xdf); lo = q_lo.predict(Xdf); hi = q_hi.predict(Xdf)
            di = tree.query(sc.transform(Xdf), k=1, workers=-1)[0] / dbar
            bidx = np.clip(np.digitize(di, edges[1:-1]), 0, DI_BINS - 1)
            Qm = Qbin[bidx]
            lo_c = lo - Qm; hi_c = hi + Qm
            awc_sum[ok] += p; lo_sum[ok] += lo_c; hi_sum[ok] += hi_c; di_sum[ok] += di
            all_reliable[ok] &= (di <= thr)
            all_reliable[~ok] = False
            any_valid |= ok

        nd_out = np.full(npx, NODATA, "float32")
        awc = np.where(any_valid, awc_sum / len(depths), NODATA).astype("float32")
        lo_m = np.where(any_valid, lo_sum / len(depths), NODATA).astype("float32")
        hi_m = np.where(any_valid, hi_sum / len(depths), NODATA).astype("float32")
        width = np.where(any_valid, (hi_sum - lo_sum) / len(depths), NODATA).astype("float32")
        di_m = np.where(any_valid, di_sum / len(depths), NODATA).astype("float32")
        paws = np.where(any_valid, awc * ROOTZONE_MM, NODATA).astype("float32")
        rel = np.where(any_valid, all_reliable.astype("uint8"), 255).astype("uint8")

        for name, arr in [("awc_mean", awc), ("awc_lo", lo_m), ("awc_hi", hi_m),
                          ("awc_width", width), ("di", di_m), ("paws_mm", paws)]:
            out[name].write(arr.reshape(nr, W), 1, window=win)
        out["reliable"].write(rel.reshape(nr, W), 1, window=win)

        vmask = any_valid; cmask = vmask & (lulc == CROPLAND_CLASS)
        n_land += int(vmask.sum()); n_rel += int((vmask & all_reliable).sum())
        n_crop += int(cmask.sum()); n_crop_rel += int((cmask & all_reliable).sum())
        if cmask.any():
            paws_crop.append(paws[cmask]); pawspi_crop.append(width[cmask] * ROOTZONE_MM)
        if bi % 10 == 0 or bi == nblocks - 1:
            print(f"block {bi+1}/{nblocks} rows {r0}-{r0+nr}  land={n_land:,}  "
                  f"reliable={n_rel:,}  ({time.time()-t0:.0f}s)")

    for ds in list(sg_ds.values()) + list(clim_ds.values()) + [lulc_ds] + list(out.values()):
        ds.close()

    pc = np.concatenate(paws_crop) if paws_crop else np.array([0.0])
    pp = np.concatenate(pawspi_crop) if pawspi_crop else np.array([0.0])
    summary = {
        "grid": [W, H], "block_rows": args.block_rows, "blocks_written": nblocks,
        "n_land_px": n_land, "reliable_frac_all_land": round(n_rel / max(n_land, 1), 4),
        "n_cropland_px": n_crop, "reliable_frac_cropland": round(n_crop_rel / max(n_crop, 1), 4),
        "cropland_OUTSIDE_reliable_pct": round(100 * (1 - n_crop_rel / max(n_crop, 1)), 1),
        "paws_mm_median_cropland": round(float(np.median(pc)), 1),
        "paws_pi_mm_median_cropland": round(float(np.median(pp)), 1),
        "aoa_threshold": round(thr, 4), "mondrian_Q_by_di_bin": np.round(Qbin, 4).tolist(),
        "runtime_s": round(time.time() - t0, 1),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SUMMARY, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> {OUTDIR}/*.tif\n-> {SUMMARY}")


if __name__ == "__main__":
    main()
