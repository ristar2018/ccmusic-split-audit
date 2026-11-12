# -*- coding: utf-8 -*-
"""Recompute the instrument-identity probe from saved P3 checkpoints (no retraining needed).

Background: the probe evaluation set was originally the held-out fold, but the A1/A2 folds are **leave-instrument-out** folds --
the test-fold instruments never appear on the training side, so probe top1 is always 0 and AUC is undefined. After switching
to an 8:2 recording-level split of the training side, a backfill tool is needed: read
`runs_p3/<arm>/<bb>/seed<N>/last_fold<f>.pt`, recompute on that fold's **training rows**, and write `fold<f>_probe.json`.
analyze_p3.py reads that file first.

Usage:
  python code/p3_probe.py                 # every existing run
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
            print("[probe] %s/%s/%s fold%d  top1 %.4f (majority class %.4f)  AUC %s"
                  % (arm, bb, sd, f, pr["probe_top1"], pr["probe_majority_baseline"],
                     pr["probe_auc_ovr_macro"]), flush=True)
            done += 1
    print("[done] probes written for %d folds" % done)


if __name__ == "__main__":
    main()
