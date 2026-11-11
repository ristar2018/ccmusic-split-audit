"""P3 第三条基线（B1）：录音分组折 + 乐器两侧都出现。

外审 M7 指出 P3 的 A0→A1 把两件事一起改了：
  ① 同源泄漏（测试录音的兄弟切片进训练）—— A0 有、A1 没有；
  ② 乐器新颖性（A1 的训练侧完全不出现测试所用乐器）—— 只有 A1 有。
所以 A0−A1 只能读作「两者之和的上界」。本脚本补第三条基线 A1r：

  * 折按**录音**分组（同一条录音的全部切片同折，同源泄漏与 A0 一样被切断）；
  * 每条录音只进一折 → 折内类分布按类分层保证每类每折 ≥1 条（可行时）；
  * 乐器**在两侧都出现**（只要该乐器的录音数 ≥2，就尽量分到不同折）→ 乐器新颖性不再是变量。

于是：
  A0（切片随机折）      vs A1r（录音分组折） → 只改同源泄漏
  A1r（录音分组折）     vs A1（乐器分组折）  → 只改乐器新颖性

泄漏断言（任一失败即 exit 2，不写产物）：
  L1 每条录音恰好出现在一折，切片不跨折
  L2 同一 audio_hash（recording_canonical）的全部录音同折
  L3 每个 (类, 折) 格的录音数：对录音数 ≥3 的类，必须三折都 ≥1
  L4 乐器覆盖：录音数 ≥3 的乐器，至少出现在 2 折（报告实际比例与阈值）
  L5 三折切片数极差 ≤ 15%（报告实际值）

产物（data/p3/）：
  p3b_splits.csv           sample_id, recording_uid, recording_canonical, technique,
                           instrument, instrument_canon, fold_b1
  p3b_fold_summary.csv     每折：切片数 / 录音数 / 类数 / 乐器数
  p3b_instrument_folds.csv 每个乐器：录音数 / 切片数 / 出现折 / 是否两侧都出现
  p3b_folds.json           assertions + split_hash + 参数

用法：
  python code/p3_split_b1.py            # 写产物，exit 0 表示全部断言通过
  python code/p3_split_b1.py --check    # 只跑断言与统计，不写产物
"""

import argparse
import collections
import csv
import hashlib
import io
import json
import os
import random
import sys

import p3_split
import p3_universe

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "data", "p3")
N_FOLDS = 3
SEED = 20261105
MIN_INSTR_RECS = 3      # 乐器「两侧都出现」的判定门槛（录音数 ≥3 才要求）
MIN_CLASS_RECS = 3      # 类「每折 ≥1 条」的判定门槛
MAX_LOAD_SPREAD = 0.15  # 三折切片数极差上界（占最大折的比例）


def build_slices():
    slices, classes = p3_split.build()
    return slices, classes


def group_recordings(slices):
    """recording_canonical → 该组的切片与录音。同音频多发布名一起进折。"""
    by_grp = collections.defaultdict(list)
    for s in slices:
        by_grp[s["recording_canonical"]].append(s)
    return by_grp


def assign_folds(slices, classes, seed=SEED):
    """按类分层的录音级三折 + 折间负载均衡（只在同类内挪录音）。"""
    rng = random.Random(seed)
    grp = group_recordings(slices)
    # 每个 canon 组：主类（切片最多的类，平手取字典序）、切片数、录音集合
    groups = {}
    for g, rows in grp.items():
        tech = collections.Counter(r["technique"] for r in rows)
        main_cls = sorted(tech.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        groups[g] = {
            "rows": rows,
            "cls": main_cls,
            "n": len(rows),
            "recs": {r["recording_uid"] for r in rows},
            "insts": {r["instrument_canon"] for r in rows},
        }

    by_cls = collections.defaultdict(list)
    for g, meta in groups.items():
        by_cls[meta["cls"]].append(g)

    fold_of = {}
    load = [0] * N_FOLDS
    for cls in sorted(by_cls):
        gs = sorted(by_cls[cls], key=lambda g: (-groups[g]["n"], g))
        start = rng.randrange(N_FOLDS)           # 每类轮换起点，避免系统性偏折
        for i, g in enumerate(gs):
            f = (start + i) % N_FOLDS
            fold_of[g] = f + 1
            load[f] += groups[g]["n"]

    # 局部搜索：只在同类内部换折，目标 = 负载极差（不破坏类分层）
    def spread(l):
        return max(l) - min(l)

    best = spread(load)
    for _ in range(4000):
        cls = rng.choice(sorted(by_cls))
        gs = by_cls[cls]
        a, b = rng.choice(gs), rng.choice(gs)
        fa, fb = fold_of[a], fold_of[b]
        if fa == fb:
            continue
        na, nb = groups[a]["n"], groups[b]["n"]
        trial = list(load)
        trial[fa - 1] += nb - na
        trial[fb - 1] += na - nb
        if spread(trial) < best:
            fold_of[a], fold_of[b] = fb, fa
            load = trial
            best = spread(load)
    return fold_of, groups, load


def assert_and_summarise(slices, groups, fold_of, classes):
    fold = {}
    for g, meta in groups.items():
        for r in meta["rows"]:
            fold[r["sample_id"]] = fold_of[g]

    # L1：一条录音不得跨折
    rec_folds = collections.defaultdict(set)
    for s in slices:
        rec_folds[s["recording_uid"]].add(fold[s["sample_id"]])
    l1_bad = {k: sorted(v) for k, v in rec_folds.items() if len(v) > 1}

    # L2：同 recording_canonical 的所有录音同折
    canon_folds = collections.defaultdict(set)
    for s in slices:
        canon_folds[s["recording_canonical"]].add(fold[s["sample_id"]])
    l2_bad = {k: sorted(v) for k, v in canon_folds.items() if len(v) > 1}

    # L3：类 × 折 支撑
    cls_recs = collections.defaultdict(lambda: collections.defaultdict(set))
    for s in slices:
        cls_recs[s["technique"]][fold[s["sample_id"]]].add(s["recording_uid"])
    cls_total = {c: len({s["recording_uid"] for s in slices if s["technique"] == c}) for c in classes}
    l3_missing = []
    for c in classes:
        for f in range(1, N_FOLDS + 1):
            if len(cls_recs[c][f]) == 0:
                l3_missing.append({"class": c, "fold": f, "class_recordings": cls_total[c]})
    l3_hard = [d for d in l3_missing if d["class_recordings"] >= MIN_CLASS_RECS]

    # L4：乐器覆盖
    inst_recs = collections.defaultdict(set)
    inst_folds = collections.defaultdict(set)
    inst_slices = collections.Counter()
    for s in slices:
        inst_recs[s["instrument_canon"]].add(s["recording_uid"])
        inst_folds[s["instrument_canon"]].add(fold[s["sample_id"]])
        inst_slices[s["instrument_canon"]] += 1
    multi = [i for i in inst_recs if len(inst_recs[i]) >= MIN_INSTR_RECS]
    both = [i for i in multi if len(inst_folds[i]) >= 2]
    inst_rows = [
        {
            "instrument_canon": i,
            "recordings": len(inst_recs[i]),
            "slices": inst_slices[i],
            "folds": "".join(str(f) for f in sorted(inst_folds[i])),
            "in_two_folds": int(len(inst_folds[i]) >= 2),
        }
        for i in sorted(inst_recs, key=lambda x: (-inst_slices[x], x))
    ]
    l4_ratio = len(both) / len(multi) if multi else 1.0
    one_fold_multi = [i for i in multi if len(inst_folds[i]) == 1]

    # L5：折负载
    load = collections.Counter(fold[s["sample_id"]] for s in slices)
    loads = [load[f] for f in range(1, N_FOLDS + 1)]
    spread_ratio = (max(loads) - min(loads)) / max(loads)

    ok = {
        "L1_recording_not_split": not l1_bad,
        "L2_audio_duplicate_same_fold": not l2_bad,
        "L3_class_support_all_folds": not l3_hard,
        "L4_instrument_in_two_folds": l4_ratio >= 0.80,
        "L5_load_spread_le_15pct": spread_ratio <= MAX_LOAD_SPREAD,
    }
    detail = {
        "n_slices": len(slices),
        "n_recordings": len({s["recording_uid"] for s in slices}),
        "n_canon_groups": len(groups),
        "n_classes": len(classes),
        "n_instruments": len(inst_recs),
        "fold_load": {str(f): loads[f - 1] for f in range(1, N_FOLDS + 1)},
        "load_spread_ratio": round(spread_ratio, 4),
        "L1_violations": len(l1_bad),
        "L2_violations": len(l2_bad),
        "L3_missing_cells": len(l3_missing),
        "L3_hard_violations": len(l3_hard),
        "L3_missing_detail": l3_missing,
        "instruments_multi": len(multi),
        "instruments_in_two_folds": len(both),
        "L4_ratio": round(l4_ratio, 4),
        "instruments_multi_single_fold": one_fold_multi,
    }
    return fold, ok, detail, inst_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只跑断言与统计，不写产物")
    a = ap.parse_args()

    slices, classes = build_slices()
    fold_of, groups, _ = assign_folds(slices, classes)
    fold, ok, detail, inst_rows = assert_and_summarise(slices, groups, fold_of, classes)

    print("== B1（录音分组折 + 乐器两侧都出现）")
    print("   切片 %d / 录音 %d / 音频组 %d / 类 %d / 乐器 %d"
          % (detail["n_slices"], detail["n_recordings"], detail["n_canon_groups"],
             detail["n_classes"], detail["n_instruments"]))
    print("   折负载 %s（极差 %.1f%%）"
          % (detail["fold_load"], 100 * detail["load_spread_ratio"]))
    print("   L1 跨折录音 %d；L2 跨折音频组 %d；L3 缺格 %d（其中硬失败 %d）"
          % (detail["L1_violations"], detail["L2_violations"],
             detail["L3_missing_cells"], detail["L3_hard_violations"]))
    print("   L4 录音≥%d 的乐器 %d 个，其中 %d 个落在 ≥2 折（%.1f%%）"
          % (MIN_INSTR_RECS, detail["instruments_multi"],
             detail["instruments_in_two_folds"], 100 * detail["L4_ratio"]))
    if detail["instruments_multi_single_fold"]:
        print("   只在单折出现的多录音乐器：%s" % detail["instruments_multi_single_fold"][:12])
    for k, v in ok.items():
        print("   [%s] %s" % ("PASS" if v else "FAIL", k))

    all_pass = all(ok.values())
    if not all_pass:
        print("!! 断言未全过，不写产物（exit 2）")
        for k, v in ok.items():
            if not v:
                print("   - %s" % k)
        return 2
    if a.check:
        print("--check：不写产物")
        return 0

    os.makedirs(OUT, exist_ok=True)
    rows = []
    for s in slices:
        rows.append({
            "sample_id": s["sample_id"],
            "recording_uid": s["recording_uid"],
            "recording_canonical": s["recording_canonical"],
            "technique": s["technique"],
            "instrument": s["instrument"],
            "instrument_canon": s["instrument_canon"],
            "fold_b1": fold[s["sample_id"]],
        })
    rows.sort(key=lambda r: (int(r["fold_b1"]), r["technique"], r["recording_uid"], r["sample_id"]))
    p1 = os.path.join(OUT, "p3b_splits.csv")
    with io.open(p1, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = []
    for fr in range(1, N_FOLDS + 1):
        sel = [r for r in rows if r["fold_b1"] == fr]
        summary.append({
            "fold": fr,
            "slices": len(sel),
            "recordings": len({r["recording_uid"] for r in sel}),
            "classes": len({r["technique"] for r in sel}),
            "instruments": len({r["instrument_canon"] for r in sel}),
            "train_recordings": len({r["recording_uid"] for r in rows if r["fold_b1"] != fr}),
        })
    p2 = os.path.join(OUT, "p3b_fold_summary.csv")
    with io.open(p2, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    p3 = os.path.join(OUT, "p3b_instrument_folds.csv")
    with io.open(p3, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(inst_rows[0].keys()))
        w.writeheader()
        w.writerows(inst_rows)

    h = hashlib.sha1()
    for r in rows:
        h.update(("%s|%s\n" % (r["sample_id"], r["fold_b1"])).encode("utf-8"))
    meta = {
        "scheme": "B1: recording-grouped folds with instruments on both sides",
        "seed": SEED,
        "n_folds": N_FOLDS,
        "min_instr_recordings": MIN_INSTR_RECS,
        "min_class_recordings": MIN_CLASS_RECS,
        "max_load_spread": MAX_LOAD_SPREAD,
        "split_hash": h.hexdigest(),
        "detail": detail,
        "assertions": ok,
        "all_pass": all_pass,
        "per_fold": summary,
        "rationale": (
            "A0 vs A1r isolates homologous leakage (only the fold unit changes); "
            "A1r vs A1 isolates instrument novelty (only the grouping key changes). "
            "Recording-level folds keep sibling slices on one side, exactly as A0/A1 do."
        ),
    }
    p4 = os.path.join(OUT, "p3b_folds.json")
    with io.open(p4, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("产物：%s / %s / %s / %s" % (p1, p2, p3, p4))
    print("split_hash %s" % meta["split_hash"])
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    raise SystemExit(main())
