#!/usr/bin/env bash
# Paper 1 / M0 — robust parallel fetch of the OpenLandMap VWC hydraulic targets
# (Zenodo record 13837179) ONLY. Resumable, stall-aborting, size-verified.
#
# Why not the python downloader: it re-HEADs/re-fetches existing SOC files and
# hangs ~45 min on Zenodo read-timeouts. Zenodo throttles per-connection, so we
# run files in parallel (each ~0.45 MB/s) -> ~1.8 MB/s aggregate at -P 4.
set -uo pipefail

REC=13837179
DEST="/Volumes/SSD Ex/PEDOFLUX_data/raw/openlandmap/zenodo_${REC}"
PAR=4
mkdir -p "$DEST"

echo "[fetch] querying Zenodo record $REC file list ..."
# emit "size<TAB>key<TAB>url" lines for .tif assets
python3 - "$REC" > /tmp/vwc_manifest.tsv <<'PY'
import sys, json, urllib.request
rec = sys.argv[1]
d = json.load(urllib.request.urlopen(f"https://zenodo.org/api/records/{rec}", timeout=60))
for f in d.get("files", []):
    k = f["key"]
    if not k.lower().endswith(".tif"):
        continue
    url = f.get("links", {}).get("self") or f"https://zenodo.org/api/records/{rec}/files/{k}/content"
    print(f'{f.get("size",0)}\t{k}\t{url}')
PY

n=$(wc -l < /tmp/vwc_manifest.tsv | tr -d ' ')
echo "[fetch] $n target tifs; downloading up to $PAR in parallel into $DEST"

fetch_one() {
  size="$1"; key="$2"; url="$3"; dest="$4/$key"
  if [[ -f "$dest" ]]; then
    have=$(stat -f%z "$dest" 2>/dev/null || echo 0)
    if [[ "$have" == "$size" ]]; then echo "[skip] $key (size match)"; return 0; fi
  fi
  echo "[get ] $key ($((size/1000000)) MB)"
  curl -fL -C - \
       --retry 20 --retry-delay 5 --retry-all-errors \
       --speed-limit 30000 --speed-time 60 \
       --connect-timeout 30 \
       -o "$dest" "$url"
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "[ERR ] $key curl rc=$rc"; return $rc; fi
  have=$(stat -f%z "$dest" 2>/dev/null || echo 0)
  if [[ "$have" == "$size" ]]; then echo "[done] $key"; else echo "[WARN] $key size $have != $size"; fi
}
# launch with a simple concurrency gate (portable to bash 3.2)
i=0
while IFS=$'\t' read -r size key url; do
  [[ -z "$key" ]] && continue
  fetch_one "$size" "$key" "$url" "$DEST" &
  i=$((i+1))
  if (( i % PAR == 0 )); then wait; fi   # drain a batch before starting more
done < /tmp/vwc_manifest.tsv
wait

echo "[fetch] complete. Final listing:"
ls -la "$DEST"/*.tif 2>/dev/null | awk '{print $5, $9}'
