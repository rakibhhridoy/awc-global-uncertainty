#!/usr/bin/env bash
# Paper 1 / M1 — after SoilGrids re-download, delete the holey reproj outputs for
# the originally-corrupt files so reproject_paper1.py regenerates them from the
# now-complete sources. soc/phh2o reproj outputs never existed (empty sources) and
# will be created fresh by the sweep.
set -uo pipefail
REPROJ="/Volumes/SSD Ex/PEDOFLUX_data/interim/reproj/soilgrids_isric"

# the original 28 corrupt files (var/file). Regenerate from /tmp if present, else hardcode vars.
VARS_DEPTHS=(
  clay/clay_0-5cm clay/clay_15-30cm clay/clay_30-60cm clay/clay_60-100cm
  sand/sand_0-5cm sand/sand_15-30cm sand/sand_30-60cm sand/sand_60-100cm
  silt/silt_0-5cm silt/silt_15-30cm silt/silt_30-60cm silt/silt_60-100cm
  nitrogen/nitrogen_0-5cm nitrogen/nitrogen_15-30cm nitrogen/nitrogen_30-60cm nitrogen/nitrogen_60-100cm
  cfvo/cfvo_15-30cm cfvo/cfvo_30-60cm cfvo/cfvo_60-100cm
  bdod/bdod_30-60cm
  soc/soc_0-5cm soc/soc_15-30cm soc/soc_30-60cm soc/soc_60-100cm
  phh2o/phh2o_0-5cm phh2o/phh2o_15-30cm phh2o/phh2o_30-60cm phh2o/phh2o_60-100cm
)
n=0
for vd in "${VARS_DEPTHS[@]}"; do
  f="$REPROJ/${vd}_mean_1000.tif"
  if [[ -f "$f" ]]; then rm -f "$f" && n=$((n+1)); fi
done
echo "[rewarp] removed $n holey reproj outputs; now run reproject_paper1.py to regenerate"
