# Global Soil → Water Hydraulic Mapping

First of two PEDOFLUX study. A data-driven global map of soil water-holding
properties (field capacity, wilting point, available water capacity) predicted from
soil composition + climate + land cover, validated against observed soil-moisture
dynamics.

- Plan: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)
- Data provenance: [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)
- Grid: EPSG:6933, 1 km global ([`configs/grid.yaml`](configs/grid.yaml))

## Reproducing

Data and derived artifacts (rasters, training table, maps) live on an external SSD
(`$PEDOFLUX_DATA`) and are **not** in git — only code, configs, docs, and small
metrics are tracked. The derived data products (reliability-masked prediction rasters +
tables) are archived on Zenodo: **[doi.org/10.5281/zenodo.21429103](https://doi.org/10.5281/zenodo.21429103)**.
Explore them interactively (map + API) at
**[fermium.systems/research/tools/available-water](https://fermium.systems/research/tools/available-water/)**.
Rebuild everything from public sources:

```bash
make deps                 # install pinned dependencies (Python 3.11)
make fetch                # M0  download VWC hydraulic targets (Zenodo 13837179)
# ... plus SoilGrids/CHELSA/LULC per docs/DATA_PROVENANCE.md
make reproject            # M1  warp to EPSG:6933 1 km
make verify               # M1  physical coverage QA
make table                # M2  build training table
make model                # M3  spatial-CV baseline vs GBT
# or simply:
make all
```

Override the data location: `make all PEDOFLUX_DATA=/path/to/data`.

## Headline result — benchmark against MEASURED retention (WoSIS, n=17,696)

The framing is an honest, spatially blocked benchmark (not a "better map" claim).
Skill against independent **measured** lab retention:

Skill is **strongly protocol-dependent** (the headline):

| Target | PTF product | Random CV | **Spatial-block (declustered)** | Continental (LOCO) |
|--------|:---:|:---:|:---:|:---:|
| Field capacity | 0.28 | 0.80 | **0.50** | 0.16 |
| Wilting point | 0.20 | 0.79 | **0.47** | 0.29 |
| Available water | 0.35 | 0.72 | **0.47** | −0.00 |

Findings: (1) the widely used global PTF product matches measurements only weakly;
(2) random CV overstates the *fair* (spatial-block, density-declustered) skill by ~60%,
and declustering confirms the 70%-Africa imbalance is **not** the cause — autocorrelation
is; (3) AWC can be *interpolated* within sampled regions (~0.47) but not *extrapolated*
to unsampled continents (~0). Wilting point transfers most consistently. A separate
ESA-CCI confrontation shows mapped hydraulics add only +0.03 R² to climate.

**The usable part (area of applicability):** within an explicit AoA, predictions are
reliable (R² 0.73–0.80) with calibrated 90% intervals (85% empirical coverage) — but
the AoA spans only ~1% of global land and *none* of the cross-continental predictions,
which mechanistically explains the spatial collapse. So an honest global hydraulic map
is trustworthy over a small, delineable subset of land, not everywhere.

Metrics: [`results/m4b_measured_validation.json`](results/m4b_measured_validation.json),
[`results/m4_confrontation.json`](results/m4_confrontation.json),
[`results/model_cv_results.json`](results/model_cv_results.json) (M3, predicting the PTF product).
Manuscript draft: [`manuscript/Manuscript.tex`](manuscript/Manuscript.tex).

## Layout
| Path | Purpose |
|------|---------|
| `src/` | pipeline (`_harmonize` grid helper, fetch, reproject, scan, build_table, model_cv) |
| `configs/grid.yaml` | common grid spec |
| `docs/` | research plan, data provenance |
| `results/` | small tracked metrics (JSON) |
| `data/` | → SSD (gitignored): raw, interim, outputs |

**Venue:** Geoderma / HESS / SOIL / RSE. **Compute:** Mac-first, small pod burst for global inference only.
