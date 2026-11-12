#!/usr/bin/env bash
# E1f: 3 negative-control libraries x 2 backbones x {A0,A2} x 1 seed = 12 runs (design note §3.7 :298-310)
#   chest_falsetto 4 classes / erhu_playing_tech 11 classes / GZ_IsoTech 8 classes
#   Guzheng_Tech99 is not in the manifest: no per-recording class layer (see the excluded section of data/manifest/summary_neg.json),
#   so the 16 runs of the design note drop to 12 runs.
# Usage: bash code/run_e1f.sh [tag]
# One log per run: logs/e1f_<lib>_<arm>_<bb>.log; overall progress in logs/e1f_driver.log; runs with an existing result.json are skipped.
set -u
cd "$(dirname "$0")/.." || exit 1
PY="D:/XiaBao/AIArtDetect/envs/fapn/python.exe"
export TORCH_HOME="$PWD/data/torch_home"

TAG="${1:-e1f}"
LIBS="chest_falsetto erhu_playing_tech GZ_IsoTech"
ARMS="A0 A2"
BBS="alexnet swin_t"
SEEDS="0"
DRV="logs/${TAG}_driver.log"
: > "$DRV"

total=$(( 3 * 2 * 2 ))
n=0
t_all0=$(date +%s)
for lib in $LIBS; do
  for bb in $BBS; do
    for arm in $ARMS; do
      n=$((n+1))
      log="logs/${TAG}_${lib}_${arm}_${bb}.log"
      if [ -f "runs/$lib/$arm/$bb/seed0/result.json" ]; then
        echo "[$n/$total] $lib $arm $bb SKIP (already done)" | tee -a "$DRV"
        continue
      fi
      t0=$(date +%s)
      "$PY" code/train_e1.py --lib "$lib" --arm "$arm" --bb "$bb" --seeds "$SEEDS" --tag "$TAG" \
        > "$log" 2>&1
      rc=$?
      t1=$(date +%s)
      echo "[$n/$total] $lib $arm $bb rc=$rc $(( (t1-t0)/60 ))min  $(date '+%H:%M:%S')" | tee -a "$DRV"
      if [ $rc -ne 0 ]; then
        echo "!! failed, log tail:" | tee -a "$DRV"
        tail -5 "$log" | tee -a "$DRV"
      fi
    done
  done
done
t_all1=$(date +%s)
echo "ALL DONE $(date '+%H:%M:%S') total $(( (t_all1-t_all0)/60 ))min" | tee -a "$DRV"
