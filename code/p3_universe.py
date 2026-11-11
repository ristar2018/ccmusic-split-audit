#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 前置（零 GPU）：43 类属性宇宙冻结 + 乐器名归一候选。

设计依据
  report/聚焦版实验设计方案v2.8.md:635-636 —— 写 code/p3_split.py 之前先做两件数据活：
    ① 乐器名归一（212 个乐器里简繁/别称合并，记录合并表）；
    ② 剔除 6 个内容类标签后的 43 类宇宙冻结（重跑 ctis_capacity 基线）。
  同文件 §7.2 —— 主口径＝「格 ≥2 且覆盖 ≥3 个乐器」+ 剔除「音阶/空弦音/筒音/单音/乐曲片段/
    特有演奏技法」6 个内容类标签 → 43 类 / 1,388 录音（中位 10 条）。

口径（与 code/ctis_capacity.py 的 sweep 逐格对齐，已对拍）
  - 单元格 = (技法, 乐器) 的**录音**数（recording_uid 去重；不是切片数——
    切片数会把 K̄ 大的小类抬过门槛，实测给 63 类 / 36,369 片，与设计稿的 49 类不符）。
  - 收录口径：某技法收录 = 「该技法下录音数 ≥ MIN_CELL 的乐器」个数 ≥ MIN_INSTR。
  - 计数口径：只数达到 MIN_CELL 的格子里的录音（ctis_capacity.py:40 的 `used`
    就是 `sum(cells[(t,i)] for t in uni for i in ok[t])`），不把未达格的录音算进来。
    ← 早先按「技法收录后全部录音」计数给 4,129 / 1,455，比设计稿多 92 / 67 条，就是这个差别。

产物
  data/p3/universe_43.csv        43 类冻结名单（每类录音数/覆盖乐器数/格子中位）
  data/p3/universe_before_drop.csv  剔除前 49 类（留痕，供 P3 预注册写「剔除规则」）
  data/p3/instrument_merge.csv   乐器名归一的机械合并候选（待人工确认，不自动改数据）
  data/p3/P3_universe.json       汇总与断言

用法
  python code/p3_universe.py            # 只读 manifest，不碰数据根
"""
from __future__ import annotations

import collections
import csv
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(HERE, "data", "manifest", "CTIS.csv")
OUT = os.path.join(HERE, "data", "p3")

MIN_CELL = 2      # 每个 (技法, 乐器) 格的录音数下限
MIN_INSTR = 3     # 技法覆盖的乐器数下限
DROP = ["音阶", "空弦音", "筒音", "单音", "乐曲片段", "特有演奏技法"]   # 内容类标签（不是演奏技法）

# 设计稿 §7.2 的账面值，用来对拍
EXPECT = {"before": (49, 4037), "after": (43, 1388)}


def load_rows():
    with open(MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def recording_cells(rows):
    """recording_uid -> (instrument, technique)，取首见（同一录音的这两列恒定）。"""
    rec = {}
    for r in rows:
        if r["instrument"] and r["technique"]:
            rec.setdefault(r["recording_uid"], (r["instrument"], r["technique"]))
    cells = collections.Counter((t, i) for i, t in rec.values())
    return rec, cells


def universe(cells, drop):
    ok = collections.defaultdict(set)
    for (t, i), c in cells.items():
        if c >= MIN_CELL:
            ok[t].add(i)
    uni = [t for t in ok if len(ok[t]) >= MIN_INSTR and t not in drop]
    uni.sort(key=lambda t: (-sum(cells[(t, i)] for i in ok[t]), t))
    used = sum(cells[(t, i)] for t in uni for i in ok[t])
    rows = []
    for t in uni:
        insts = ok[t]
        n = sum(cells[(t, i)] for i in insts)
        per = [cells[(t, i)] for i in insts]
        rows.append({"technique": t, "recordings": n, "instruments": len(insts),
                     "cell_median": statistics.median(per), "cell_min": min(per),
                     "cell_max": max(per)})
    return rows, used


SUFFIX_KEY = re.compile(r"[\s\-_]*(?:[A-Ga-g]调|D调|F调|G调)\s*$")
PAREN = re.compile(r"[（(][^）)]*[）)]")


def norm_name(name):
    """机械归一键：去括注、去调名后缀、全角空格归一。只用于**找候选**，不直接改数据。"""
    s = name.replace("\u3000", " ").strip()
    s = PAREN.sub("", s)
    s = SUFFIX_KEY.sub("", s)
    return s.strip()


# 乐器名归一的裁决（2026-11-05 定稿）：6 组 14 名全部合并，212 → 204。
# 理由：折分组的目的是「留出乐器」，若把同一乐器的不同制式/调名/写法拆成两组，
# 折间就会出现「训练见过 扬剧主胡 D调、测试 扬剧主胡 F调」这类同音色泄漏，
# 使跨乐器泛化的结论偏乐观。合并是保守口径；未合并版（212）只作敏感性说明。
RATIONALE = {
    "扬剧主胡": "同一乐器的 D 调 / F 调 / 小西皮唱腔写法，音色同源",
    "伽倻琴": "传统与改良制式，同族同音色",
    "奚琴": "传统与改良制式，同族同音色",
    "盖板": "传统与新制式（D 调），同族同音色",
    "短箫": "带/不带「（传统）」后缀的同名乐器",
    "陇剧陇胡": "传统与改良制式（D 调），同族同音色",
}


def merge_candidates(rows):
    inst = collections.Counter(r["instrument"] for r in rows if r["instrument"])
    groups = collections.defaultdict(list)
    for name in inst:
        groups[norm_name(name)].append(name)
    out = []
    for key, names in sorted(groups.items()):
        if len(names) < 2:
            continue
        names.sort()
        canon = min(names, key=lambda s: (len(s.strip()), s.strip())).strip()
        for nm in names:
            out.append({"norm_key": key, "instrument": nm.strip(), "recordings_slices": inst[nm],
                        "merge_to": canon, "auto_rule": "去括注/去调名后缀后同名",
                        "used": 1,
                        "decision": "合并：同一乐器的不同制式/调名/写法",
                        "rationale": RATIONALE.get(key, "同族乐器，合并为保守口径")})
    return out, inst


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows()
    rec, cells = recording_cells(rows)
    print("manifest %d 片 / 可解析录音 %d / 单元格 %d" % (len(rows), len(rec), len(cells)))

    before, used_before = universe(cells, set())
    after, used_after = universe(cells, set(DROP))
    miss = [t for t in DROP if t not in [r["technique"] for r in before]]
    print("剔除前：%d 类 / %d 录音（账面 %d / %d）" % (len(before), used_before, *EXPECT["before"]))
    print("剔除后：%d 类 / %d 录音（账面 %d / %d）中位 %s 最小 %d"
          % (len(after), used_after, *EXPECT["after"],
             statistics.median([r["recordings"] for r in after]),
             min(r["recordings"] for r in after)))
    print("6 个内容类标签里命中门槛的：%s；一个都没命中的：%s"
          % ([t for t in DROP if t not in miss], miss))
    print("Top10：" + "、".join("%s %d" % (r["technique"], r["recordings"]) for r in after[:10]))

    for fn, data in (("universe_43.csv", after), ("universe_before_drop.csv", before)):
        with open(os.path.join(OUT, fn), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    cand, inst = merge_candidates(rows)
    with open(os.path.join(OUT, "instrument_merge.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["norm_key", "instrument", "recordings_slices", "merge_to", "auto_rule", "used", "decision", "rationale"])
        w.writeheader()
        w.writerows(cand)
    print("乐器名 %d 个 → 机械合并候选 %d 个名字 / %d 组（可减 %d 个名字）"
          % (len(inst), len(cand), len({c["norm_key"] for c in cand}), len(cand) - len({c["norm_key"] for c in cand})))

    ok = {"before_classes": len(before) == EXPECT["before"][0],
          "before_recordings": used_before == EXPECT["before"][1],
          "after_classes": len(after) == EXPECT["after"][0],
          "after_recordings": used_after == EXPECT["after"][1]}
    summary = {"generated_by": "code/p3_universe.py",
               "design_ref": ["report/聚焦版实验设计方案v2.8.md:635-636", "同文件 §7.2"],
               "rule": {"min_cell": MIN_CELL, "min_instr": MIN_INSTR, "unit": "recording_uid",
                        "drop_labels": DROP},
               "before": {"classes": len(before), "recordings": used_before},
               "after": {"classes": len(after), "recordings": used_after,
                         "median": statistics.median([r["recordings"] for r in after]),
                         "min": min(r["recordings"] for r in after)},
               "instrument_names": len(inst),
               "merge_candidates": len(cand),
               "checks": ok,
               "all_pass": all(ok.values())}
    with open(os.path.join(OUT, "P3_universe.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("断言：%s → %s" % (ok, "全过" if summary["all_pass"] else "!! 有偏差"))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
