#!/usr/bin/env bash
# Resumable download of the LARGE optional datasets (real-world P&IDs).
# Zenodo throttles hard (~30 KB/s observed). Run this overnight; it resumes on restart.
#   bash scripts/fetch_optional_datasets.sh
set -u
cd "$(dirname "$0")/../data/raw" || exit 1

fetch () {  # name url
  echo "[$(date +%H:%M)] fetching $1 ..."
  curl -L -C - --retry 999 --retry-delay 10 --retry-all-errors \
       --speed-limit 1024 --speed-time 120 \
       -o "$1" "$2"
}

# PID2Graph  (9.3 GB, CC-BY-SA-4.0) - real + synthetic P&IDs WITH graph ground truth
fetch PID2Graph.zip  "https://zenodo.org/api/records/14803338/files/PID2Graph.zip/content"

# PID_dataset (6.7 GB, CC-BY-4.0) - real industry / webscraped P&IDs, images only
fetch PID_Dataset.zip "https://zenodo.org/api/records/8028570/files/PID_Dataset.zip/content"

echo "[$(date +%H:%M)] done. unzip with: unzip -q PID2Graph.zip -d PID2Graph/"
