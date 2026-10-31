#!/usr/bin/env bash
R=runs/pid_s640
if [ -f $R/results.csv ]; then
  echo "epoch  box_loss  cls_loss   mAP50   mAP50-95"
  tail -n +2 $R/results.csv | awk -F, '{printf "%5s  %8.3f  %8.3f  %7.4f  %8.4f\n",$1,$3,$4,$8,$9}' | tail -8
  echo "best mAP50: $(tail -n +2 $R/results.csv | awk -F, 'BEGIN{m=0}{if($8>m)m=$8}END{printf "%.4f",m}')"
else
  echo -n "epoch 1 in progress: step "
  tr '\r' '\n' < scratch/train.log | grep -oE "[0-9]+/1015" | tail -1
fi
