#!/usr/bin/env bash
# Promote the newest training checkpoint to the live detector and verify it.
set -e
cd "$(dirname "$0")/.."
SRC=${1:-runs/pid_v2/weights/best.pt}
[ -f "$SRC" ] || { echo "no checkpoint at $SRC"; exit 1; }
mkdir -p models && cp "$SRC" models/yolo_pid.pt
echo "promoted $SRC -> models/yolo_pid.pt ($(du -h models/yolo_pid.pt | cut -f1))"
