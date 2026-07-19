#!/usr/bin/env bash
# Paper 1 / M1-fix — re-download the truncated SoilGrids 1km tifs robustly.
#
# The original repo download() left 28+ truncated files (network stalls); rasterio
# warp fails with "Chunk and warp failed" / TIFFReadEncodedTile on them. We delete
# the bad ones (partial bytes can't be safely resumed) and parallel-fetch fresh
# with stall-abort, then verify against ISRIC's per-var checksum.sha256.txt.
#
# Bad list = the files rasterio errored on (passed in /tmp/corrupt_sg.txt, one
# "<var>/<file>" per line). Falls back to size<5MB scan if that file is absent.
set -uo pipefail

SG="/Volumes/SSD Ex/PEDOFLUX_data/raw/soilgrids_isric"
BASE="https://files.isric.org/soilgrids/latest/data_aggregated/1000m"
PAR=6
LIST=/tmp/corrupt_sg.txt

# build the bad list as "<var>/<filename>"
if [[ -f "$LIST" ]]; then
  mapfile -t BAD < "$LIST"
else
  echo "[refetch] /tmp/corrupt_sg.txt absent; scanning for <5MB tifs"
  BAD=()
  while IFS= read -r f; do
    rel="${f#$SG/}"; BAD+=("$rel")
  done < <(find "$SG" -name '*_1000.tif' ! -name '._*' -size -5000k)
fi
echo "[refetch] ${#BAD[@]} files to re-download (parallel $PAR)"

get_one() {
  rel="$1"                                  # e.g. cfvo/cfvo_30-60cm_mean_1000.tif
  # tolerate list entries that are just the filename (no var dir)
  if [[ "$rel" != */* ]]; then
    var="${rel%%_*}"; rel="$var/$rel"
  fi
  dest="$SG/$rel"; url="$BASE/$rel"
  mkdir -p "$(dirname "$dest")"
  rm -f "$dest"                             # drop partial/corrupt bytes; fetch fresh
  echo "[get ] $rel"
  curl -fL -C - --retry 20 --retry-delay 5 --retry-all-errors \
       --speed-limit 30000 --speed-time 60 --connect-timeout 30 \
       -o "$dest" "$url"
  rc=$?
  [[ $rc -ne 0 ]] && { echo "[ERR ] $rel curl rc=$rc"; return $rc; }
  echo "[done] $rel ($(stat -f%z "$dest") bytes)"
}
export -f get_one; export SG BASE

i=0
for rel in "${BAD[@]}"; do
  [[ -z "$rel" ]] && continue
  get_one "$rel" &
  i=$((i+1)); (( i % PAR == 0 )) && wait
done
wait

# NOTE: ISRIC's published checksum.sha256.txt are stale/version-mismatched here
# (valid files report FAILED), so we verify integrity downstream via the physical
# coverage scan (Paper1/src/scan_coverage.py) instead of checksums.
echo "[refetch] downloads done. Final sizes:"
for rel in "${BAD[@]}"; do
  [[ -z "$rel" ]] && continue
  [[ "$rel" != */* ]] && rel="${rel%%_*}/$rel"
  p="$SG/$rel"; [[ -f "$p" ]] && printf "  %9d MB  %s\n" "$(( $(stat -f%z "$p")/1000000 ))" "$rel"
done
