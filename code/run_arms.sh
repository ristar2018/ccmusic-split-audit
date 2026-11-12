#!/usr/bin/env bash
# Generic arm driver: bash code/run_arms.sh <tag> <arm1,arm2,...> [seeds]
# One log per run: logs/<tag>_<lib>_<arm>_<bb>.log; progress in logs/<tag>_driver.log
set -u
cd "$(dirname "$0")/.." || exit 1
PY="D:/XiaBao/AIArtDetect/envs/fapn/python.exe"
export TORCH_HOME="$PWD/data/torch_home"
TAG="${1:?tag required}"; IFS=',' read -r -a ARMS <<< "${2:?arms required, e.g. A2 or A0,A1}"; SEEDS="${3:-0,1,2}"
LIBS="${4:-CTIS,music_genre,pianos,CNPM,bel_canto}"
DRV="logs/${TAG}_driver.log"; : > "$DRV"
IFS="," read -r -a LIBARR <<< "$LIBS"
n_lib=${#LIBARR[@]}
total=$(( n_lib * 2 * ${#ARMS[@]} )); n=0; t_all0=$(date +%s)
for lib in "${LIBARR[@]}"; do
  for bb in alexnet swin_t; do
    for arm in "${ARMS[@]}"; do
      n=$((n+1)); log="logs/${TAG}_${lib}_${arm}_${bb}.log"; skip=1
      for s in ${SEEDS//,/ }; do [ -f "runs/$lib/$arm/$bb/seed$s/result.json" ] || { skip=0; break; }; done
      [ "$skip" = 1 ] && { echo "[$n/$total] $lib $arm $bb SKIP (already done)" | tee -a "$DRV"; continue; }
      t0=$(date +%s)
      "$PY" code/train_e1.py --lib "$lib" --arm "$arm" --bb "$bb" --seeds "$SEEDS" --tag "$TAG" > "$log" 2>&1; rc=$?
      echo "[$n/$total] $lib $arm $bb rc=$rc $(( ($(date +%s)-t0)/60 ))min  $(date '+%H:%M:%S')" | tee -a "$DRV"
      [ $rc -ne 0 ] && { echo "!! failed, log tail:" | tee -a "$DRV"; tail -5 "$log" | tee -a "$DRV"; }
    done; done; done
echo "ALL DONE $(date '+%H:%M:%S') total $(( ($(date +%s)-t_all0)/60 ))min" | tee -a "$DRV"
