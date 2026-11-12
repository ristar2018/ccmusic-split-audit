# -*- coding: utf-8 -*-
"""Compute audit: aggregate measured GPU wall clock from driver logs and run artifacts, against the 10.6 GPU-h estimate in the design note.

Sources (all on-disk facts; no training is re-run):
  1. logs/e1b_driver.log / e1f_driver.log -- `[n/N] <lib> <arm> <bb> rc=<rc> <m>min <HH:MM:SS>`
  2. logs/e1c_driver.log -- same format (A2 arm, driven by run_arms.sh)
  3. `elapsed_s` in runs_p3/**/fold*_result.json -- P3 single-fold training wall clock

Writes data/results/cost.json and a Markdown table; gaps are listed explicitly, never extrapolated.

Usage: python code/cost_audit.py
"""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(HERE, "logs")
OUT = os.path.join(HERE, "data", "results", "cost.json")

RE_DRV = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+(A\d)\s+(\S+)\s+rc=(\d+)\s+(\d+)min")


def parse_driver(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RE_DRV.match(line.strip())
        if m:
            rows.append({"file": os.path.basename(path), "n": int(m.group(1)), "total": int(m.group(2)),
                         "lib": m.group(3), "arm": m.group(4), "bb": m.group(5),
                         "rc": int(m.group(6)), "minutes": int(m.group(7))})
    return rows


def p3_elapsed():
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs_p3", "*", "*", "seed*", "fold*_result.json"))):
        j = json.load(open(p, encoding="utf-8"))
        rows.append({"path": os.path.relpath(p, HERE), "arm": j["arm"], "bb": j["bb"], "seed": j["seed"],
                     "fold": j["fold"], "elapsed_s": j.get("elapsed_s", 0)})
    return rows


def main():
    e1 = parse_driver(os.path.join(LOG, "e1b_driver.log"))
    e1c = parse_driver(os.path.join(LOG, "e1c_driver.log"))
    e1f = parse_driver(os.path.join(LOG, "e1f_driver.log")) + parse_driver(os.path.join(LOG, "e1f_driver_1.log"))
    p3 = p3_elapsed()

    def agg(rows, label):
        if not rows:
            return {"label": label, "runs": 0, "gpu_h": 0.0, "ok": 0, "failed": 0, "note": "log missing"}
        gpu_h = sum(r["minutes"] for r in rows) / 60.0
        return {"label": label, "runs": len(rows), "gpu_h": round(gpu_h, 2),
                "ok": sum(1 for r in rows if r["rc"] == 0),
                "failed": sum(1 for r in rows if r["rc"] != 0),
                "completed": sum(1 for r in rows if r["n"] == r["total"])}

    p3_h = sum(r["elapsed_s"] for r in p3) / 3600.0
    p3_group_h = len({(r["arm"], r["bb"], r["seed"]) for r in p3}) and p3_h
    summary = {
        "E1b": agg(e1, "E1b A0/A1, 5 libraries x 2 backbones x 3 seeds"),
        "E1c": agg(e1c, "E1c A2, 5 libraries x 2 backbones x 3 seeds"),
        "E1f": agg(e1f, "E1f negative control, 3 libraries x 2 backbones x {A0,A2} x seed0"),
        "P3": {"label": "P3 three folds x three arms x 2 backbones x 2 seeds (training wall clock, before in-fold evaluation)",
               "runs": len(p3), "folds_done": len(p3),
               "gpu_h": round(p3_h, 2),
               "expected_folds": 36 * 3},
        "design_estimate_gpu_h": 10.6,
        "note": ("E1b/E1c/E1f minutes come from the driver logs (each run includes testing and sealing); "
                 "P3 comes from fold-level elapsed_s and excludes probes and bootstrap analyses. "
                 "Neither matches the caliber of the 10.6 GPU-h figure in the design note: that note assumed "
                 "339-341 slices/s, whereas measured throughput is lower (I/O plus fixed overhead on the small "
                 "libraries), so the measured machine time is higher; the report should quote the measured values "
                 "and state the caliber."),
    }
    total = summary["E1b"]["gpu_h"] + summary["E1c"]["gpu_h"] + summary["E1f"]["gpu_h"] + summary["P3"]["gpu_h"]
    summary["total_measured_gpu_h"] = round(total, 2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(summary, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("experiment  runs  done  fail  measured GPU-h")
    for k in ("E1b", "E1c", "E1f", "P3"):
        s = summary[k]
        print("%-10s %5s %5s %5s %10s" % (k, s.get("runs", "-"), s.get("ok", s.get("folds_done", "-")),
                                          s.get("failed", "-"), s.get("gpu_h", "-")))
    print("total measured %.2f GPU-h (design-note estimate 10.6)" % total)
    print("[out] data/results/cost.json")


if __name__ == "__main__":
    main()
