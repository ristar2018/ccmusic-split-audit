#!/usr/bin/env bash
# B2: negative controls with a uniform 3 seeds (the noise objection raised in external review M6)
#   3 libraries (chest_falsetto / erhu_playing_tech / GZ_IsoTech) x 2 backbones x {A0,A2} x seed0/1/2 = 36 runs
#   Results land in a separate runs_b2/, leaving the sealed single-seed evidence of the E1f round in runs/ untouched.
#   Leakage caliber: each recording of a negative-control library has 1 slice (K̄≈1); A0/A2 both split at recording level and the training side holds no Q_test recording;
#   this script creates no new split, it only changes the seed.
# Usage: bash code/run_b2.sh [tag]
set -u
cd "$(dirname "$0")/.." || exit 1
PY="D:/XiaBao/AIArtDetect/envs/fapn/python.exe"
export TORCH_HOME="$PWD/data/torch_home"

TAG="${1:-b2}"
LIBS="chest_falsetto erhu_playing_tech GZ_IsoTech"
ARMS="A0 A2"
BBS="alexnet swin_t"
SEEDS="0,1,2"
OUT="runs_b2"
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
      done3=1
      for s in 0 1 2; do
        [ -f "$OUT/$lib/$arm/$bb/seed$s/result.json" ] || done3=0
      done
      if [ "$done3" = "1" ]; then
        echo "[$n/$total] $lib $arm $bb SKIP (3 seeds done)" | tee -a "$DRV"
        continue
      fi
      t0=$(date +%s)
      "$PY" code/train_e1.py --lib "$lib" --arm "$arm" --bb "$bb" --seeds "$SEEDS" \
        --out "$OUT" --resume --tag "$TAG" > "$log" 2>&1
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
