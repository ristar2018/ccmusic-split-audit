#!/usr/bin/env python3
"""round2_method_facts.py — 只读核算，为 method_disclosure.md 提供可复算数字。

覆盖四类事实（全部来自已落盘产物，不重跑训练/自助法）：
  A) 每库身份核对可用的音频/录音/切片数（data/manifest/*.json + datasets_inventory.json）
  B) 两臂分配事实：qtest/qpool 录音重叠、跨侧同 hash、组规模、跨类同 hash
     （data/splits/v2/<lib>_{qtest,qpool,r,c}.csv）
  C) 类集合与自助缺类：e1_metrics.csv 的 n_classes 与 Q_test 出现类是否一致；
     逐类录音数 → 一次录音簇重抽样里该类完全缺席的概率（解析式，不抽样）
  D) 数值验证脚本规模（data/results/paper_verify.json）

只读：不写 data/ 产物。默认打印；--json 给出落盘路径时才写一份 JSON 到
data/results/human_review/round2/（该目录允许写入）。

用法（Windows 上 fapn python 的 subprocess 被沙箱拒绝，本脚本不用 subprocess）：
    python code/round2_method_facts.py --root .
    python3 code/round2_method_facts.py --root .   # WSL 下亦可
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MAIN_LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto"]
SIDES = ["qtest", "qpool", "r", "c"]


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def side_facts(root, lib):
    """B：两侧录音重叠、同 hash 跨侧、组规模、跨录音/跨类组。"""
    rows = {s: load_csv(root / "data" / "splits" / "v2" / ("%s_%s.csv" % (lib, s)))
            for s in SIDES}
    rec = {s: {r["recording_canonical"] for r in rows[s]} for s in SIDES}

    qtest_pool_overlap = sorted(rec["qtest"] & rec["qpool"])
    overlap_slices = {"qtest": 0, "qpool": 0}
    for s in ("qtest", "qpool"):
        for r in rows[s]:
            if r["recording_canonical"] in (rec["qtest"] & rec["qpool"]):
                overlap_slices[s] += 1

    leak = sorted(rec["qtest"] & (rec["r"] | rec["c"]))

    hash_sides = defaultdict(set)
    for s in SIDES:
        for r in rows[s]:
            if r["content_hash"]:
                hash_sides[r["content_hash"]].add(s)
    hash_multi = {h: sorted(v) for h, v in hash_sides.items() if len(v) > 1}

    # 按 splits_v2.py:258-288 的规则重建组：Q 内每条录音按 slice_index 排序，
    # 相邻两片（偶数位与后一片）并集；再按同一 label 内相同 content_hash 并集。
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    q_rows = []
    for s in ("qtest", "qpool"):
        for r in rows[s]:
            r = dict(r)
            r["_side"] = s
            q_rows.append(r)

    by_rec = defaultdict(list)
    for r in q_rows:
        by_rec[(r["label"], r["recording_canonical"])].append(r)
    pair_split = 0
    for rs in by_rec.values():
        rs.sort(key=lambda r: int(r["slice_index"]))
        for i, r in enumerate(rs):
            find(r["sample_id"])
            if i % 2 == 1:
                union(rs[i - 1]["sample_id"], r["sample_id"])
                if rs[i - 1]["_side"] != r["_side"]:
                    pair_split += 1
    by_hash = defaultdict(list)
    for r in q_rows:
        if r["content_hash"]:
            by_hash[(r["label"], r["content_hash"])].append(r["sample_id"])
    for ids in by_hash.values():
        for x in ids[1:]:
            union(ids[0], x)

    comp = defaultdict(list)
    for r in q_rows:
        comp[find(r["sample_id"])].append(r)
    sizes = Counter(len(v) for v in comp.values())
    cross_rec = sum(1 for v in comp.values()
                    if len({r["recording_canonical"] for r in v}) > 1)
    cross_lab = sum(1 for v in comp.values() if len({r["label"] for r in v}) > 1)
    side_split = sum(1 for v in comp.values() if len({r["_side"] for r in v}) > 1)

    # 整条录音在 Q 内的总片数（qtest+qpool 即该录音全部 Q 片）
    per_rec_slices = Counter(r["recording_canonical"]
                             for s in ("qtest", "qpool") for r in rows[s])
    singleton = {k for k, v in per_rec_slices.items() if v == 1}

    return {
        "slices": {s: len(rows[s]) for s in SIDES},
        "recordings": {s: len(rec[s]) for s in SIDES},
        "classes": {s: len({r["label"] for r in rows[s]}) for s in SIDES},
        "qtest_qpool_recording_overlap": len(qtest_pool_overlap),
        "qtest_qpool_overlap_slices": overlap_slices,
        "qtest_r_c_recording_overlap": len(leak),
        "hash_multi_side": len(hash_multi),
        "groups": len(comp),
        "group_size_hist": dict(sorted(sizes.items())),
        "pairs_split_across_sides": pair_split,
        "groups_split_across_sides": side_split,
        "groups_cross_recording": cross_rec,
        "groups_cross_class": cross_lab,
        "singleton_recordings": len(singleton),
        "singleton_all_in_qtest": all(any(r["recording_canonical"] == k for r in rows["qtest"])
                                      for k in singleton),
    }


def class_facts(root, lib, qtest_rows, n_classes_csv):
    """C：Q_test 的类集合、逐类录音数、重抽样缺类概率。"""
    labels = [r["label"] for r in qtest_rows]
    uniq = sorted(set(labels))
    per_cls = Counter()
    seen = defaultdict(set)
    for r in qtest_rows:
        seen[r["label"]].add(r["recording_canonical"])
    for lab, recs in seen.items():
        per_cls[lab] = len(recs)
    n_rec = len({r["recording_canonical"] for r in qtest_rows})
    p_miss = {lab: (1.0 - n / n_rec) ** n_rec for lab, n in per_cls.items()}
    exp_missing = sum(p_miss.values())
    p_any = 1.0 - math.prod(1.0 - p for p in p_miss.values())
    return {
        "n_classes_in_qtest": len(uniq),
        "n_classes_in_e1_metrics": n_classes_csv,
        "classes_match": (n_classes_csv is None or n_classes_csv == len(uniq)),
        "n_recordings": n_rec,
        "min_recs_per_class": min(per_cls.values()) if per_cls else 0,
        "max_recs_per_class": max(per_cls.values()) if per_cls else 0,
        "classes_with_lt5_recordings": sum(1 for n in per_cls.values() if n < 5),
        "expected_classes_missing_per_resample": round(exp_missing, 3),
        "p_at_least_one_class_missing": round(p_any, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="项目根目录")
    ap.add_argument("--json", default="", help="可选：结果 JSON 落盘路径（须在 round2 目录内）")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    summary = load_json(root / "data" / "manifest" / "summary.json")["libraries"]
    pairs = {}
    for name in ("pairs_summary.json", "ctis_pairs.json"):
        p = root / "data" / "manifest" / name
        if p.exists():
            pairs = load_json(p)
            break
    inv = load_json(root / "data" / "datasets_inventory.json")
    inv_by = {d["dataset"]: d for d in inv["datasets"]}

    out = {"root": str(root), "A": {}, "B": {}, "C": {}, "D": {}}

    # ---------- A
    slices_sum = 0
    for lib in MAIN_LIBS:
        st = summary[lib]
        slices_sum += st["slices"]
        out["A"][lib] = {
            "classes": st.get("classes"),
            "recordings": st.get("recordings_in_manifest") or st.get("recordings"),
            "slices": st["slices"],
            "audio_files_on_disk": inv_by.get(lib, {}).get("audio"),
            "audio_hash_done": st.get("audio_hash_done"),
            "duplicate_audio_recordings": st.get("duplicate_audio_recordings", 0),
        }
    out["A"]["_slices_total"] = slices_sum
    out["A"]["_pairs"] = pairs.get("pairs", []) if isinstance(pairs, dict) else pairs

    # ---------- B / C
    metrics = load_csv(root / "data" / "results" / "e1_metrics.csv")
    ncls = defaultdict(set)
    for r in metrics:
        ncls[r["library"]].add(int(r["n_classes"]))

    for lib in MAIN_LIBS:
        out["B"][lib] = side_facts(root, lib)
        qtest = load_csv(root / "data" / "splits" / "v2" / ("%s_qtest.csv" % lib))
        nc = sorted(ncls[lib])[0] if ncls.get(lib) else None
        out["C"][lib] = class_facts(root, lib, qtest, nc)
        out["C"][lib]["n_classes_consistency"] = sorted(ncls.get(lib, []))

    # ---------- D
    pv = load_json(root / "data" / "results" / "paper_verify.json")
    out["D"] = {"passes": pv.get("passes"), "fails": pv.get("fails"),
                "panels": pv.get("panels"), "totals": pv.get("totals")}

    # ---------- 打印
    print("== A 身份核对可用面 ==")
    print("lib | classes | recordings | slices | audio_on_disk | audio_hash_done | dup_audio_recs")
    for lib in MAIN_LIBS:
        d = out["A"][lib]
        print("%s | %s | %s | %s | %s | %s | %s" % (
            lib, d["classes"], d["recordings"], d["slices"],
            d["audio_files_on_disk"], d["audio_hash_done"], d["duplicate_audio_recordings"]))
    print("slices_total = %d" % slices_sum)

    print("\n== B 两臂分配（qtest/qpool/r/c） ==")
    for lib in MAIN_LIBS:
        b = out["B"][lib]
        print("[%s] slices=%s recs=%s classes=%s" % (lib, b["slices"], b["recordings"], b["classes"]))
        print("    qtest∩qpool 录音=%d（qtest 片 %d / qpool 片 %d）；qtest∩(r∪c) 录音=%d" % (
            b["qtest_qpool_recording_overlap"], b["qtest_qpool_overlap_slices"]["qtest"],
            b["qtest_qpool_overlap_slices"]["qpool"], b["qtest_r_c_recording_overlap"]))
        print("    同 hash 跨侧=%d；组数=%d 规模分布=%s；跨录音组=%d；跨类组=%d；跨侧组=%d；相邻对被拆=%d；单记录=%d 全在 qtest=%s" % (
            b["hash_multi_side"], b["groups"], b["group_size_hist"],
            b["groups_cross_recording"], b["groups_cross_class"],
            b["groups_split_across_sides"], b["pairs_split_across_sides"],
            b["singleton_recordings"], b["singleton_all_in_qtest"]))

    print("\n== C 类集合与重抽样缺类 ==")
    for lib in MAIN_LIBS:
        c = out["C"][lib]
        print("[%s] n_classes(qtest)=%d n_classes(e1_metrics)=%s 一致=%s | recs=%d 每类 %d–%d 片录音 | <5 录音的类=%d | E[缺类/次]=%.3f P(≥1 类缺)=%.4f" % (
            lib, c["n_classes_in_qtest"], c["n_classes_in_e1_metrics"], c["classes_match"],
            c["n_recordings"], c["min_recs_per_class"], c["max_recs_per_class"],
            c["classes_with_lt5_recordings"], c["expected_classes_missing_per_resample"],
            c["p_at_least_one_class_missing"]))

    print("\n== D 验证脚本 ==")
    print("paper_verify.json: passes=%s fails=%s" % (out["D"]["passes"], out["D"]["fails"]))
    print("panels totals=%s" % json.dumps(out["D"]["totals"], ensure_ascii=False))

    if a.json:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n→ %s" % p)


if __name__ == "__main__":
    main()
