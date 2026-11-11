"""C1：无关录音注入对照（外审 M1 的归因缺口）。

问题（M1）：A1 相对 A0 一次改了三件事 —— 暴露（测试录音的兄弟切片）、录音多样性、
切片分布，所以 Δ1 不能单独归给同源泄漏。要拆开，需要一个**体积相同、标签分布相同、
但内容与测试录音无关**的注入臂：

    A0 : m_c 片，全部来自 C（无 Q_pool）
    C1 : 从 A0 的训练集里按类抽掉 |Q_pool_c| 片，再按类补进同样片数 ——
         补进来的片子只来自**非测试录音**（C 的剩余 + R 分区）
    A1 : 从 A0 的训练集里按类抽掉 |Q_pool_c| 片，再补进 Q_pool（测试录音的兄弟切片）

于是 C1 与 A1 唯一的差别是注入内容的来源（非测试 vs 测试录音），体积与类别平衡相同；
C1 与 A0 的差别只是「同类别的非测试内容换了换」，预期 ≈0，恰好是一个内部零对照。

泄漏断言（任一失败即 exit 2，不写产物）：
  K1 注入片与 Q_test 的 sample_id 无交集
  K2 注入片与 Q_test 的 recording_uid 无交集（真·非测试录音）
  K3 注入片的 content_hash 与 Q_test 的 content_hash 无交集
  K4 注入片与 A0 训练集无交集；C1 训练集内部 sample_id 唯一
  K5 每类训练片数与 A0 逐类相同（体积与类别平衡都保住）
  K6 总训练片数与 A1 相差 ≤0.5%

已知代价（写在这里以免后人误读）：注入源里 R 分区占多数时，C1 臂的 R 侧验证会被污染，
因此该臂只报封存的 Q_test；三臂都不做任何选型（固定 10 epoch、取最后一轮权重）。

用法：
  python code/c1_inject.py            # 写 data/splits/v2/CTIS_c1_train.csv + c1_audit.json
  python code/c1_inject.py --check    # 只跑断言与统计
"""

import argparse
import collections
import csv
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SPLITS = os.path.join(PROJ, "data", "splits", "v2")
LIB = "CTIS"
SEED = 20261105
MAX_TOTAL_GAP = 0.005   # 与 A1 的总体积差上界


def read_csv(path):
    with io.open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load():
    d = {}
    for name in ("a0_train", "a1_train", "qpool", "qtest", "c", "r"):
        p = os.path.join(SPLITS, "%s_%s.csv" % (LIB, name))
        d[name] = read_csv(p)
    return d


def build(d, seed=SEED):
    rng = random.Random(seed)
    a0 = d["a0_train"]
    qpool = d["qpool"]
    qtest = d["qtest"]
    test_sids = {r["sample_id"] for r in qtest}
    test_recs = {r["recording_uid"] for r in qtest}
    test_hashes = {r["content_hash"] for r in qtest if r.get("content_hash")}

    # 每类要注入多少片 = Q_pool 的逐类计数
    need = collections.Counter(r["label"] for r in qpool)

    # 候选注入池：C 与 R 中、不在 A0 训练集、且录音不是测试录音的切片
    a0_sids = {r["sample_id"] for r in a0}
    cand = []
    for name in ("c", "r"):
        for r in d[name]:
            if r["sample_id"] in a0_sids:
                continue
            if r["recording_uid"] in test_recs:
                continue
            if r.get("content_hash") and r["content_hash"] in test_hashes:
                continue
            cand.append(dict(r, _src=name))
    by_cls = collections.defaultdict(list)
    for r in cand:
        by_cls[r["label"]].append(r)

    # 逐类按录音轮流取，避免注入片集中在少数几条录音上
    injected, shortfall = [], {}
    for cls, k in sorted(need.items()):
        pool = by_cls.get(cls, [])
        if not pool:
            shortfall[cls] = k
            continue
        by_rec = collections.defaultdict(list)
        for r in pool:
            by_rec[r["recording_uid"]].append(r)
        recs = sorted(by_rec)
        rng.shuffle(recs)
        for rec in recs:
            rng.shuffle(by_rec[rec])
        take, i = [], 0
        while len(take) < k and recs:
            progressed = False
            for rec in list(recs):
                if len(take) >= k:
                    break
                if by_rec[rec]:
                    take.append(by_rec[rec].pop())
                    progressed = True
                if not by_rec[rec]:
                    recs.remove(rec)
            if not progressed:
                break
        injected.extend(take)
        if len(take) < k:
            shortfall[cls] = k - len(take)

    # 从 A0 里按类抽掉同样片数：优先从切片多的录音里抽，尽量不抽空录音
    inj_cnt = collections.Counter(r["label"] for r in injected)
    a0_by_cls = collections.defaultdict(list)
    for r in a0:
        a0_by_cls[r["label"]].append(r)
    drop = []
    for cls, k in sorted(inj_cnt.items()):
        rows = sorted(a0_by_cls.get(cls, []), key=lambda r: r["sample_id"])
        by_rec = collections.defaultdict(list)
        for r in rows:
            by_rec[r["recording_uid"]].append(r)
        recs = sorted(by_rec, key=lambda x: (-len(by_rec[x]), x))
        taken = 0
        # 一轮一条录音，保证每条录音最多被抽到只剩 1 片
        while taken < k:
            progressed = False
            for rec in recs:
                if taken >= k:
                    break
                if len(by_rec[rec]) > 1:
                    drop.append(by_rec[rec].pop())
                    taken += 1
                    progressed = True
            if not progressed:
                break
        if taken < k:      # 兜底：允许把录音抽空
            for rec in recs:
                while by_rec[rec] and taken < k:
                    drop.append(by_rec[rec].pop())
                    taken += 1
                if taken >= k:
                    break

    drop_sids = {r["sample_id"] for r in drop}
    c1 = [r for r in a0 if r["sample_id"] not in drop_sids] + injected
    # 输出用与其它切分一致的列
    cols = ["sample_id", "label", "recording_uid", "recording_canonical",
            "content_hash", "candidate_group", "slice_index"]
    rows = [{c: r.get(c, "") for c in cols} for r in c1]
    rows.sort(key=lambda r: (r["label"], r["recording_uid"], r["sample_id"]))
    return rows, injected, drop, need, shortfall


def audit(rows, injected, drop, need, d):
    test_sids = {r["sample_id"] for r in d["qtest"]}
    test_recs = {r["recording_uid"] for r in d["qtest"]}
    test_hashes = {r["content_hash"] for r in d["qtest"] if r.get("content_hash")}
    inj_sids = {r["sample_id"] for r in injected}
    inj_recs = {r["recording_uid"] for r in injected}
    inj_hashes = {r["content_hash"] for r in injected if r.get("content_hash")}
    a0_sids = {r["sample_id"] for r in d["a0_train"]}

    per_cls_c1 = collections.Counter(r["label"] for r in rows)
    per_cls_a0 = collections.Counter(r["label"] for r in d["a0_train"])
    cls_equal = all(per_cls_c1[c] == per_cls_a0[c] for c in per_cls_a0)
    n_a1 = len(d["a1_train"])
    gap = abs(len(rows) - n_a1) / n_a1

    ok = {
        "K1_no_test_sample_ids": not (inj_sids & test_sids),
        "K2_no_test_recordings": not (inj_recs & test_recs),
        "K3_no_test_content_hash": not (inj_hashes & test_hashes),
        "K4_injection_disjoint_from_a0": not (inj_sids & a0_sids),
        "K4b_train_ids_unique": len({r["sample_id"] for r in rows}) == len(rows),
        "K5_per_class_equal_to_a0": cls_equal,
        "K6_total_within_0.5pct_of_a1": gap <= MAX_TOTAL_GAP,
    }
    detail = {
        "n_train": len(rows),
        "n_a0_train": len(d["a0_train"]),
        "n_a1_train": n_a1,
        "gap_vs_a1": round(gap, 5),
        "n_injected": len(injected),
        "n_dropped": len(drop),
        "injected_from": dict(collections.Counter(r.get("_src", "?") for r in injected)),
        "injected_recordings": len(inj_recs),
        "injected_classes": len({r["label"] for r in injected}),
        "class_need_total": sum(need.values()),
        "class_counts_match": cls_equal,
        "n_classes": len(per_cls_a0),
    }
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = load()
    rows, injected, drop, need, shortfall = build(d)
    ok, detail = audit(rows, injected, drop, need, d)

    print("== C1（无关录音注入对照）")
    print("   A0 %d 片 → C1 %d 片（注入 %d / 抽掉 %d），A1 %d 片"
          % (detail["n_a0_train"], detail["n_train"], detail["n_injected"],
             detail["n_dropped"], detail["n_a1_train"]))
    print("   注入来源 %s，来自 %d 条非测试录音、%d 个类"
          % (detail["injected_from"], detail["injected_recordings"], detail["injected_classes"]))
    if shortfall:
        print("   !! 注入不足的类 %d 个（总缺 %d 片）" % (len(shortfall), sum(shortfall.values())))
    for k, v in ok.items():
        print("   [%s] %s" % ("PASS" if v else "FAIL", k))
    all_pass = all(ok.values())
    if not all_pass:
        print("!! 断言未全过，不写产物（exit 2）")
        return 2
    if a.check:
        print("--check：不写产物")
        return 0

    p = os.path.join(SPLITS, "%s_c1_train.csv" % LIB)
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label", "recording_uid",
                                          "recording_canonical", "content_hash",
                                          "candidate_group", "slice_index"])
        w.writeheader()
        w.writerows(rows)
    meta = {
        "lib": LIB,
        "arm": "C1",
        "scheme": "equal-volume injection from non-test recordings",
        "seed": SEED,
        "assertions": ok,
        "all_pass": all_pass,
        "detail": detail,
        "class_shortfall": shortfall,
        "note": ("Injected slices come from C remainder and R; for this arm only the "
                 "R partition is partially used as a source, so C1's validation "
                 "numbers are not interpretable. No arm performs model selection."),
    }
    with io.open(os.path.join(SPLITS, "c1_audit.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("产物：%s" % p)
    print("        %s" % os.path.join(SPLITS, "c1_audit.json"))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    raise SystemExit(main())
