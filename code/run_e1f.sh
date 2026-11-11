#!/usr/bin/env bash
# E1f：负对照 3 库 × 2 骨干 × {A0,A2} × 1 seed = 12 runs（设计稿 §3.7 :298-310）
#   chest_falsetto 4 类 / erhu_playing_tech 11 类 / GZ_IsoTech 8 类
#   Guzheng_Tech99 未进清单：无逐录音类别层（见 data/manifest/summary_neg.json → excluded），
#   故设计稿的 16 runs 降为 12 runs。
# 用法：bash code/run_e1f.sh [tag]
# 每 run 一日志 logs/e1f_<lib>_<arm>_<bb>.log；总进度 logs/e1f_driver.log；已有 result.json 自动跳过。
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
        echo "[$n/$total] $lib $arm $bb SKIP（已完成）" | tee -a "$DRV"
        continue
      fi
      t0=$(date +%s)
      "$PY" code/train_e1.py --lib "$lib" --arm "$arm" --bb "$bb" --seeds "$SEEDS" --tag "$TAG" \
        > "$log" 2>&1
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
