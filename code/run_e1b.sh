#!/usr/bin/env bash
# E1b：A0 vs A1，5 库 × 2 骨干 × 3 seed = 60 runs。
# 用法：bash code/run_e1b.sh [tag]
# 每 run 一个日志 logs/e1b_<tag>_<lib>_<arm>_<bb>.log；总进度写 logs/e1b_<tag>_driver.log
set -u
cd "$(dirname "$0")/.." || exit 1
PY="D:/XiaBao/AIArtDetect/envs/fapn/python.exe"
export TORCH_HOME="$PWD/data/torch_home"
TAG="${1:-e1b}"
DRV="logs/${TAG}_driver.log"
: > "$DRV"
n=0
for lib in CTIS music_genre pianos CNPM bel_canto; do
  for bb in alexnet swin_t; do
    for arm in A0 A1; do
      n=$((n+1))
      log="logs/${TAG}_${lib}_${arm}_${bb}.log"
      t0=$(date +%s)
      "$PY" code/train_e1.py --lib "$lib" --arm "$arm" --bb "$bb" --seeds 0,1,2 --tag "$TAG" \
        > "$log" 2>&1
      rc=$?
      t1=$(date +%s)
      echo "[$n/20] $lib $arm $bb rc=$rc $(( (t1-t0)/60 ))min  $(date '+%H:%M:%S')" | tee -a "$DRV"
      if [ $rc -ne 0 ]; then
        echo "!! 失败，日志尾部：" | tee -a "$DRV"
        tail -5 "$log" | tee -a "$DRV"
      fi
    done
  done
done
echo "ALL DONE $(date '+%H:%M:%S')" | tee -a "$DRV"
