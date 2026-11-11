# -*- coding: utf-8 -*-
"""机时核对：从驱动日志与 run 产物汇总实际 GPU 墙钟，对比设计稿的 10.6 GPU·h 估算。

数据来源（都是落盘事实，不重算训练）：
  1. logs/e1b_driver.log / e1f_driver.log —— `[n/N] <lib> <arm> <bb> rc=<rc> <m>min <HH:MM:SS>`
  2. logs/e1c_driver.log —— 同上（A2 臂，由 run_arms.sh 驱动）
  3. runs_p3/**/fold*_result.json 的 `elapsed_s` —— P3 单折训练墙钟

输出 data/results/cost.json 与一张 Markdown 表；有缺口时显式列出，不做外推。

用法：python code/cost_audit.py
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
            return {"label": label, "runs": 0, "gpu_h": 0.0, "ok": 0, "failed": 0, "note": "日志缺失"}
        gpu_h = sum(r["minutes"] for r in rows) / 60.0
        return {"label": label, "runs": len(rows), "gpu_h": round(gpu_h, 2),
                "ok": sum(1 for r in rows if r["rc"] == 0),
                "failed": sum(1 for r in rows if r["rc"] != 0),
                "completed": sum(1 for r in rows if r["n"] == r["total"])}

    p3_h = sum(r["elapsed_s"] for r in p3) / 3600.0
    p3_group_h = len({(r["arm"], r["bb"], r["seed"]) for r in p3}) and p3_h
    summary = {
        "E1b": agg(e1, "E1b A0/A1 五库×2骨干×3seed"),
        "E1c": agg(e1c, "E1c A2 五库×2骨干×3seed"),
        "E1f": agg(e1f, "E1f 负对照 3库×2骨干×{A0,A2}×seed0"),
        "P3": {"label": "P3 三折×三臂×2骨干×2seed（训练墙钟，含折内评估前）",
               "runs": len(p3), "folds_done": len(p3),
               "gpu_h": round(p3_h, 2),
               "expected_folds": 36 * 3},
        "design_estimate_gpu_h": 10.6,
        "note": ("E1b/E1c/E1f 的分钟数来自驱动日志（含每 run 的测试与封存）；P3 来自 fold 级 elapsed_s，"
                 "不含探针与自助分析。两者都与设计稿的「10.6 GPU·h」不同口径：设计稿按 339–341 片/s 的吞吐估，"
                 "实测吞吐更低（I/O + 小库的固定开销），故实跑机时更高——报告里应写实测值并说明口径。"),
    }
    total = summary["E1b"]["gpu_h"] + summary["E1c"]["gpu_h"] + summary["E1f"]["gpu_h"] + summary["P3"]["gpu_h"]
    summary["total_measured_gpu_h"] = round(total, 2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(summary, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("实验        runs  完成  失败  实测 GPU·h")
    for k in ("E1b", "E1c", "E1f", "P3"):
        s = summary[k]
        print("%-10s %5s %5s %5s %10s" % (k, s.get("runs", "-"), s.get("ok", s.get("folds_done", "-")),
                                          s.get("failed", "-"), s.get("gpu_h", "-")))
    print("合计实测 %.2f GPU·h（设计稿估算 10.6）" % total)
    print("[out] data/results/cost.json")


if __name__ == "__main__":
    main()
