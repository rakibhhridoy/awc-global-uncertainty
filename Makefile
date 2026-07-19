# Paper 1 — Global Soil -> Water Hydraulic Mapping : reproducible pipeline
#
# Data + derived artifacts live on the external SSD (PEDOFLUX_DATA), NOT in git.
# Override the data root if needed:  make all PEDOFLUX_DATA=/path/to/data
#
# Stages (each depends on the previous):
#   fetch     M0  download VWC hydraulic targets (OpenLandMap / Zenodo 13837179)
#   reproject M1  warp predictors + targets to EPSG:6933 1 km
#   verify    M1  physical valid-pixel coverage QA (>=0.27 land fraction)
#   table     M2  build stratified soil->water training table
#   model     M3  Ridge vs GBT under random + leave-one-region-out CV
#
# SoilGrids re-download helpers (refetch_soilgrids.sh / rewarp_fixed.sh) are
# recovery tools for truncated downloads, run manually when verify flags gaps.

PY ?= python3
export PEDOFLUX_DATA ?= /Volumes/SSD Ex/PEDOFLUX_data

.PHONY: all fetch reproject verify table model aoa mechanism conformal globalraster clean-pyc help

all: model ## run the full pipeline end-to-end

fetch: ## M0 download VWC hydraulic targets
	bash src/fetch_vwc_targets.sh

reproject: ## M1 warp predictors + targets to the common grid
	$(PY) src/reproject_paper1.py

verify: ## M1 physical coverage QA of reproj outputs
	$(PY) src/scan_coverage.py

table: ## M2 build the training table
	$(PY) src/build_table.py --n 200000

model: ## M3 spatial-CV baseline vs GBT
	$(PY) src/model_cv.py

climatology: ## M4 build observed ESA-CCI soil-moisture climatology
	$(PY) src/esacci_climatology.py

confront: ## M4 confront predicted hydraulics vs observed ESA-CCI SM (coupling test)
	$(PY) src/confront_esacci.py

aoa: ## M5 area of applicability + calibrated uncertainty
	$(PY) src/area_of_applicability.py

mechanism: ## M11 demonstrated transfer mechanism (importance-weighted covariate shift vs LOCO skill)
	$(PY) src/mechanism.py

conformal: ## M12 honest out-of-domain intervals (split-CQR + Mondrian/DI-CQR conditional coverage)
	$(PY) src/conformal.py

product: ## M7 global reliability-masked product + agricultural significance
	$(PY) src/global_product.py

globalraster: ## M13 seamless global AWC raster (conformal interval + AoA mask) -> product/*.tif
	$(PY) src/global_raster.py

figures: ## M5 build D3 (SVG) publication figures + PDFs
	$(PY) figures/export_figure_data.py
	cd figures && npm install --silent && for f in fig1_cv_gap fig2_continent fig3_scatter fig4_map fig5_aoa fig6_global_map fig7_significance fig8_downstream fig9_recovery fig10_mechanism fig11_conformal figS1_waterbalance graphical_abstract; do \
	  node d3/$$f.js; rsvg-convert -f pdf -o out/$$f.pdf out/$$f.svg; done

notebook: ## build + execute the end-to-end analysis notebook (M3-M8 from prepared tables)
	$(PY) notebooks/_build_notebook.py
	cd notebooks && jupyter nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=2400 Paper1_analysis.ipynb

guide: ## compile the technical reproducibility guide (PDF)
	cd docs && pdflatex -interaction=nonstopmode Technical_Guide.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode Technical_Guide.tex >/dev/null && echo "built Technical_Guide.pdf"

explainer: ## compile the plain-language explainer (PDF)
	cd docs && pdflatex -interaction=nonstopmode Plain_Language_Explainer.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode Plain_Language_Explainer.tex >/dev/null && echo "built Plain_Language_Explainer.pdf"

manuscript: figures ## build the manuscript PDF (needs figures first)
	cd manuscript && pdflatex -interaction=nonstopmode Manuscript.tex >/dev/null && \
	  bibtex Manuscript >/dev/null && \
	  pdflatex -interaction=nonstopmode Manuscript.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode Manuscript.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode Supplementary.tex >/dev/null && echo "built Manuscript.pdf + Supplementary.pdf"

coverletter: ## build cover letter (letter-class) in pdf + docx
	cd manuscript && pdflatex -interaction=nonstopmode cover_letter.tex >/dev/null && \
	  pandoc cover_letter.tex -o cover_letter.docx && \
	  echo "built cover_letter.pdf + cover_letter.docx"

reviewers: ## build suggested-reviewers sheet in pdf + docx
	cd manuscript && pdflatex -interaction=nonstopmode suggested_reviewers.tex >/dev/null && \
	  pandoc suggested_reviewers.tex -o suggested_reviewers.docx && \
	  echo "built suggested_reviewers.pdf + suggested_reviewers.docx"

commsenv: figures ## build the Communications Earth & Environment (Nature Portfolio) PDF
	cd manuscript && python3 _build_commsenv.py && \
	  pdflatex -interaction=nonstopmode Manuscript_CommsEnv.tex >/dev/null && \
	  bibtex Manuscript_CommsEnv >/dev/null && \
	  pdflatex -interaction=nonstopmode Manuscript_CommsEnv.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode Manuscript_CommsEnv.tex >/dev/null && echo "built Manuscript_CommsEnv.pdf"

stoten: figures ## build the Elsevier/STOTEN submission PDF (elsarticle)
	cd manuscript && pdflatex -interaction=nonstopmode Manuscript_STOTEN.tex >/dev/null && \
	  bibtex Manuscript_STOTEN >/dev/null && \
	  pdflatex -interaction=nonstopmode Manuscript_STOTEN.tex >/dev/null && \
	  pdflatex -interaction=nonstopmode Manuscript_STOTEN.tex >/dev/null && echo "built Manuscript_STOTEN.pdf"

deps: ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
