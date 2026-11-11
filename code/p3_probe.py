# -*- coding: utf-8 -*-
"""从已保存的 P3 检查点重算乐器身份探针（不需要重新训练）。

背景：探针的评估集最初取 held-out 折，但 A1/A2 是**留乐器**折——测试折的乐器在训练侧
完全没出现，探针 top1 恒为 0、AUC 无定义。改为「训练侧按录音 8:2 切」后需要一个
回填工具：读 `runs_p3/<arm>/<bb>/seed<N>/last_fold<f>.pt`，用该折的**训练行**重算，
写 `fold<f>_probe.json`。analyze_p3.py 优先读这个文件。

用法：
  python code/p3_probe.py                 # 全部已有 run
  python code/p3_probe.py --arms A0,A1 --bbs alexnet
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_p3 as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A0,A1,A2")
    ap.add_argument("--bbs", default="alexnet,swin_t")
    ap.add_argument("--out", default=T.RUNS)
    a = ap.parse_args()
    arms = {s.strip() for s in a.arms.split(",") if s.strip()}
    bbs = {s.strip() for s in a.bbs.split(",") if s.strip()}

    folds, rows, classes, insts = T.load_p3()
    T.split_a0(rows)
    n_cls, n_inst = len(classes), len(insts)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pat = os.path.join(a.out, "*", "*", "seed*", "last_fold1.pt")
    done = 0
    for ck in sorted(glob.glob(pat)):
        run_dir = os.path.dirname(ck)
        arm, bb, sd = run_dir.split(os.sep)[-3:]
        if arm not in arms or bb not in bbs:
            continue
        key = "fold_a0" if arm == "A0" else "fold"
        for f in (1, 2, 3):
            pt = os.path.join(run_dir, "last_fold%d.pt" % f)
            if not os.path.exists(pt):
                continue
            model, _ = T.build_model(bb, n_cls, None)
            model.load_state_dict(torch.load(pt, map_location=device))
            model = model.to(device).eval()
            tr_rows = [r for r in rows if r[key] != f]
            pr = T.instrument_probe(model, bb, device, tr_rows, None, n_inst)
            json.dump(pr, open(os.path.join(run_dir, "fold%d_probe.json" % f), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=2)
            print("[probe] %s/%s/%s fold%d  top1 %.4f (多数类 %.4f)  AUC %s"
                  % (arm, bb, sd, f, pr["probe_top1"], pr["probe_majority_baseline"],
                     pr["probe_auc_ovr_macro"]), flush=True)
            done += 1
    print("[done] %d 个折的探针已落盘" % done)


if __name__ == "__main__":
    main()
