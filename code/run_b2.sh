#!/usr/bin/env bash
# B2：负对照统一 3 seed（外审 M6 的噪声质疑）
#   3 库（chest_falsetto / erhu_playing_tech / GZ_IsoTech）× 2 骨干 × {A0,A2} × seed0/1/2 = 36 runs
#   落到独立 runs_b2/，不动 runs/ 里已封存的单 seed 证据（E1f 那一轮）。
#   泄漏口径：负对照库每条录音 1 片（K̄≈1），A0/A2 都是录音级切分，训练侧不含 Q_test 录音；
#   本脚本不新建任何切分，只换 seed。
# 用法：bash code/run_b2.sh [tag]
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
        echo "[$n/$total] $lib $arm $bb SKIP（3 seed 已完成）" | tee -a "$DRV"
        continue
      fi
      t0=$(date +%s)
      "$PY" code/train_e1.py --lib "$lib" --arm "$arm" --bb "$bb" --seeds "$SEEDS" \
        --out "$OUT" --resume --tag "$TAG" > "$log" 2>&1
      rc=$?
      t1=$(date +%s)
      echo "[$n/$total] $lib $arm $bb rc=$rc $(( (t1-t0)/60 ))min  $(date '+%H:%M:%S')" | tee -a "$DRV"
      if [ $rc -ne 0 ]; then
        echo "!! 失败，日志尾部：" | tee -a "$DRV"
        tail -5 "$log" | tee -a "$DRV"
      fi
    done
  done
done
t_all1=$(date +%s)
echo "ALL DONE $(date '+%H:%M:%S') 总耗时 $(( (t_all1-t_all0)/60 ))min" | tee -a "$DRV"
