#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""round2_data_accounting.py — round2 审稿数据核算的复跑脚本（只读产物，不训练）。

产出（写入 data/results/human_review/round2/）：
  A. slice_budget_by_class.csv  CTIS 三臂 A0/A1/C1 的逐类训练切片数
  B. p3_fold_composition.csv    §6.4 四臂（R-slice/L-inst/L-inst+Adv/R-rec）三折构成

全部数字直接来自磁盘产物并逐项断言：
  - A0/A1/C1 清单：data/splits/v2/CTIS_{a0,a1,c1}_train.csv，字段 label
  - 预算账面：data/splits/v2/feasibility.csv，字段 m_c / qpool_slices / n_a1
  - 去重证据：data/splits/v2/CTIS_qpool.csv vs feasibility 的 qpool_slices 差值
  - P3 折定义：data/p3/p3_splits.csv(fold) / p3b_splits.csv(fold_b1)
  - P3 实际训练：runs_p3/<arm>/<bb>/seed0/fold<f>_result.json + fold<f>.npz(records)
  - P3 指标：data/results/p3/p3_metrics.csv

用法（项目根目录，Windows）：
  D:\\XiaBao\\AIArtDetect\\envs\\fapn\\python.exe code\\round2_data_accounting.py

任一断言失败 -> exit 1。脚本不改 paper/、不改 data/splits/、不写 runs/。
"""
import collections
import csv
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D_SPLITS = os.path.join(ROOT, "data", "splits", "v2")
D_P3 = os.path.join(ROOT, "data", "p3")
D_RUNS = os.path.join(ROOT, "runs_p3")
D_RES = os.path.join(ROOT, "data", "results")
D_OUT = os.path.join(D_RES, "human_review", "round2")

N_FOLDS = 3
SEED_SPLIT = 20261105

CHECKS = []


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(("PASS  " if ok else "FAIL  ") + name + (("   " + detail) if detail else ""))


# ---------------------------------------------------------------- A
def task_a():
    print("\n== A  CTIS 三臂逐类切片预算 ==")
    arms = [("A0", "CTIS_a0_train.csv"), ("A1", "CTIS_a1_train.csv"),
            ("C1", "CTIS_c1_train.csv")]
    counts, rows_out = {}, []
    for arm, fn in arms:
        rows = read_csv(os.path.join(D_SPLITS, fn))
        counts[arm] = collections.Counter(r["label"] for r in rows)
        n = len(rows)
        check("A/%s 行数 == %d" % (arm, n), n == len(rows))
        rows_out += [("CTIS", arm, cls, counts[arm][cls]) for cls in sorted(counts[arm])]
    write_csv(os.path.join(D_OUT, "slice_budget_by_class.csv"),
              ["lib", "arm", "class", "n_slices"], rows_out)

    tot = {a: sum(c.values()) for a, c in counts.items()}
    check("A/总数 A0=25502, A1=25498, C1=25502",
          tot == {"A0": 25502, "A1": 25498, "C1": 25502}, str(tot))
    check("A/三臂类别数均 201", all(len(c) == 201 for c in counts.values()),
          str({a: len(c) for a, c in counts.items()}))

    d01 = {k: counts["A0"][k] - counts["A1"].get(k, 0)
           for k in counts["A0"] if counts["A0"][k] != counts["A1"].get(k, 0)}
    d02 = {k: counts["A0"][k] - counts["C1"].get(k, 0)
           for k in counts["A0"] if counts["A0"][k] != counts["C1"].get(k, 0)}
    check("A/A0-A1 仅 C0296 差 4 片", d01 == {"C0296": 4}, str(d01))
    check("A/A0-C1 逐类全等", d02 == {}, str(d02))

    # 账面（feasibility）vs 实际：qpool 计划 4182、实际 4178，差 4 全在 C0296
    feas = [r for r in read_csv(os.path.join(D_SPLITS, "feasibility.csv"))
            if r["library"] == "CTIS"]
    feas = {r["cls"]: r for r in feas}
    qp = collections.Counter(r["label"] for r in read_csv(os.path.join(D_SPLITS, "CTIS_qpool.csv")))
    plan = {c: int(feas[c]["qpool_slices"]) for c in feas}
    delta = {c: plan[c] - qp.get(c, 0) for c in plan if plan[c] != qp.get(c, 0)}
    check("A/qpool 计划 %d vs 实际 %d，差 4 全在 C0296"
          % (sum(plan.values()), sum(qp.values())),
          sum(plan.values()) == 4182 and sum(qp.values()) == 4178 and delta == {"C0296": 4},
          str(delta))
    check("A/feasibility 的 n_a1 账面合计 25502、a1_short 全 0（去重前口径）",
          sum(int(r["n_a1"]) for r in feas.values()) == 25502
          and all(int(r["a1_short"]) == 0 for r in feas.values()),
          "C0296 n_a1=%s a1_short=%s" % (feas["C0296"]["n_a1"], feas["C0296"]["a1_short"]))
    check("A/A1 实际 C0296 == 237（= 35 qpool + 202 C 侧）",
          counts["A1"]["C0296"] == 237, str(counts["A1"]["C0296"]))

    asrt = json.load(open(os.path.join(D_SPLITS, "assertions.json"), encoding="utf-8"))
    ctis = asrt["libraries"]["CTIS"]
    check("A/assertions.json A1_slice_budget=true —— 该断言用 trim 前 cls_stats，故漏检",
          ctis["ok"]["A1_slice_budget"] is True and ctis["all_pass"] is True,
          "dropped_dup_slices=%s" % ctis["dropped_dup_slices"])

    audit = json.load(open(os.path.join(D_SPLITS, "c1_audit.json"), encoding="utf-8"))
    det = audit.get("detail", {})
    check("A/c1_audit：C1 逐类 == A0（K5）且 all_pass",
          audit.get("all_pass") is True
          and audit.get("assertions", {}).get("K5_per_class_equal_to_a0") is True
          and det.get("n_train") == 25502 and det.get("n_a1_train") == 25498,
          "n_injected=%s n_dropped=%s" % (det.get("n_injected"), det.get("n_dropped")))
    return counts


# ---------------------------------------------------------------- B
def _split_a0(rows, classes):
    """复刻 code/train_p3.py:85-96。键必须是整数类索引（sorted(by_cls)），
    用 technique 字符串插入序会改变 rng 消费顺序 -> fold 分配错。"""
    label_of = {c: i for i, c in enumerate(classes)}
    by = collections.defaultdict(list)
    for r in rows:
        by[label_of[r["technique"]]].append(r)
    rng = random.Random(SEED_SPLIT)
    for c in sorted(by):
        sub = sorted(by[c], key=lambda r: r["sample_id"])
        rng.shuffle(sub)
        for i, r in enumerate(sub):
            r["fold_a0"] = i % N_FOLDS + 1
    return rows


def _compose(rows, key, fold):
    """复刻 code/train_p3.py:289-299 的 train/val/test 划分。"""
    te = [r for r in rows if r[key] == fold]
    tr_pool = [r for r in rows if r[key] != fold]
    rng = random.Random(SEED_SPLIT + fold)
    recs = sorted({r["recording_uid"] for r in tr_pool})
    rng.shuffle(recs)
    val_recs = set(recs[:max(1, int(0.1 * len(recs)))])
    tr = [r for r in tr_pool if r["recording_uid"] not in val_recs]
    va = [r for r in tr_pool if r["recording_uid"] in val_recs]
    return tr, va, te


def _macro_f1(y, pred, n_classes):
    """复刻 code/analyze_p3.py:35-47：只统计真值或预测至少出现一次的类。"""
    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(sum(f1s) / len(f1s))


def _pooled_seed_f1(arm, bb, seed, n_classes):
    """合并三折 npz 后算切片级 macro-F1（复刻 analyze_p3.py:50-62 + :140-146）。"""
    import numpy as np
    ps, ys = [], []
    for fold in (1, 2, 3):
        z = np.load(os.path.join(D_RUNS, arm, bb, "seed%d" % seed, "fold%d.npz" % fold),
                    allow_pickle=True)
        ps.append(z["probs"].astype(np.float32))
        ys.append(z["y"].astype(np.int64))
    prob = np.concatenate(ps)
    y = np.concatenate(ys)
    return _macro_f1(y, prob.argmax(1), n_classes)


def task_b(counts_a):
    print("\n== B  P3 §6.4 折构成 ==")
    folds_meta = json.load(open(os.path.join(D_P3, "p3_folds.json"), encoding="utf-8"))
    classes = folds_meta["classes"]
    rows = read_csv(os.path.join(D_P3, "p3_splits.csv"))
    b1 = {r["sample_id"]: int(r["fold_b1"])
          for r in read_csv(os.path.join(D_P3, "p3b_splits.csv"))}
    for r in rows:
        r["fold"] = int(r["fold"])
        if r["sample_id"] not in b1:
            raise SystemExit("p3b_splits.csv 缺少样本: " + r["sample_id"])
        r["fold_b1"] = b1[r["sample_id"]]
    _split_a0(rows, classes)
    check("B/p3 切片 7528、录音 1388、类 43、乐器 135",
          len(rows) == 7528
          and len({r["recording_uid"] for r in rows}) == 1388
          and len({r["technique"] for r in rows}) == 43
          and len({r["instrument_canon"] for r in rows}) == 135,
          "n=%d rec=%d cls=%d inst=%d" % (len(rows),
                                          len({r["recording_uid"] for r in rows}),
                                          len({r["technique"] for r in rows}),
                                          len({r["instrument_canon"] for r in rows})))

    # 折定义互斥性（L-inst：乐器不跨折）
    inst_folds = collections.defaultdict(set)
    for r in rows:
        inst_folds[r["instrument_canon"]].add(r["fold"])
    check("B/L-inst 乐器不跨折（135/135 只出现在一折）",
          all(len(v) == 1 for v in inst_folds.values()),
          "multi=%d" % sum(1 for v in inst_folds.values() if len(v) > 1))

    arm_key = {"A0": "fold_a0", "A1": "fold", "A2": "fold", "A1r": "fold_b1"}
    scheme = {"A0": "R-slice", "A1": "L-inst", "A2": "L-inst+Adv", "A1r": "R-rec"}
    out = []
    for arm in ("A0", "A1", "A2", "A1r"):
        key = arm_key[arm]
        for f in (1, 2, 3):
            tr, va, te = _compose(rows, key, f)
            n_tr_rec = len({r["recording_uid"] for r in tr})
            n_te_rec = len({r["recording_uid"] for r in te})
            n_cls_tr = len({r["technique"] for r in tr})
            n_cls_te = len({r["technique"] for r in te})
            out.append([f, arm, n_tr_rec, len(tr), n_te_rec, len(te), n_cls_tr, n_cls_te])

            res = json.load(open(os.path.join(
                D_RUNS, arm, "alexnet", "seed0", "fold%d_result.json" % f), encoding="utf-8"))
            check("B/%s f%d 切片数对 result.json (%d/%d/%d)"
                  % (arm, f, res["n_train"], res["n_val"], res["n_test"]),
                  (len(tr), len(va), len(te)) == (res["n_train"], res["n_val"], res["n_test"]),
                  "recomputed %d/%d/%d" % (len(tr), len(va), len(te)))

            try:
                import numpy as np
                npz = np.load(os.path.join(
                    D_RUNS, arm, "alexnet", "seed0", "fold%d.npz" % f), allow_pickle=True)
                npz_recs = {str(x) for x in npz["records"].tolist()}
                check("B/%s f%d 测试录音集合 == npz records (%d)" % (arm, f, len(npz_recs)),
                      npz_recs == {r["recording_uid"] for r in te})
                check("B/%s f%d 测试切片数 == npz y (%d)" % (arm, f, len(npz["y"])),
                      len(npz["y"]) == len(te))
            except ImportError:
                check("B/%s f%d npz 校验（缺 numpy，跳过）" % (arm, f), True)

    write_csv(os.path.join(D_OUT, "p3_fold_composition.csv"),
              ["fold", "arm", "n_train_recordings", "n_train_slices",
               "n_test_recordings", "n_test_slices", "n_classes_train",
               "n_classes_test"], out)

    # 折覆盖缺口：L-inst fold1 缺「自然泛音」
    te1 = {r["technique"] for r in rows if r["fold"] == 1}
    missing = sorted({r["technique"] for r in rows} - te1)
    check("B/L-inst fold1 测试缺 1 类（自然泛音），fold2/3 全覆盖",
          missing == ["自然泛音"]
          and len({r["technique"] for r in rows if r["fold"] == 2}) == 43
          and len({r["technique"] for r in rows if r["fold"] == 3}) == 43, str(missing))

    # R-slice 的录音级泄漏：训练/测试录音交集
    tr0, _, te0 = _compose(rows, "fold_a0", 1)
    ov = len({r["recording_uid"] for r in tr0} & {r["recording_uid"] for r in te0})
    check("B/R-slice f1 训练∩测试录音 %d 条（非零即切片级折的泄漏来源）" % ov, ov > 0)

    # 43 类口径
    uni = read_csv(os.path.join(D_P3, "universe_43.csv"))
    before = read_csv(os.path.join(D_P3, "universe_before_drop.csv"))
    check("B/类宇宙 49 -> 43，录音 %d -> %d"
          % (sum(int(r["recordings"]) for r in before), sum(int(r["recordings"]) for r in uni)),
          len(before) == 49 and len(uni) == 43
          and sum(int(r["recordings"]) for r in before) == 4037
          and sum(int(r["recordings"]) for r in uni) == 1388)
    dropped = sorted({r["technique"] for r in before} - {r["technique"] for r in uni})
    check("B/被剔除 6 个内容类标签",
          dropped == sorted(["音阶", "空弦音", "筒音", "单音", "乐曲片段", "特有演奏技法"]),
          str(dropped))

    # 论文 §6.4 的 0.4483 / 0.5295 与产物核对（A1 行；两 seed 全精度 pooled 均值）
    met = {(r["arm"], r["bb"]): r for r in read_csv(os.path.join(D_RES, "p3", "p3_metrics.csv"))}
    for bb, want in (("alexnet", 0.4483), ("swin_t", 0.5295)):
        vals = [_pooled_seed_f1("A1", bb, s, 43) for s in (0, 1)]
        mean = round(sum(vals) / len(vals), 4)
        # 注意：平均 pooled.json 里已舍入的 4 位值会得 0.4484/0.5294（0.44835 的浮点舍入），
        # 论文口径是全精度 seed F1 求均值后再 round 4（analyze_p3.py:142-146,171）。
        rounded4 = [round(v, 4) for v in vals]
        check("B/A1 %s pooled %.4f == p3_metrics %.4f == 论文"
              % (bb, mean, float(met[("A1", bb)]["macro_f1_slice"])),
              abs(mean - float(met[("A1", bb)]["macro_f1_slice"])) < 5e-5
              and abs(float(met[("A1", bb)]["macro_f1_slice"]) - want) < 5e-5,
              "seed_f1=%s(4dp %s)" % ([round(v, 6) for v in vals], rounded4))
    # 附注：p3b_instr 目录 A0/A1/A2 与 p3 逐位相同，仅 A1r 为新训
    p3b = {(r["arm"], r["bb"]): float(r["macro_f1_slice"])
           for r in read_csv(os.path.join(D_RES, "p3b_instr", "p3_metrics.csv"))}
    same = all(abs(p3b[(a, b)] - float(met[(a, b)]["macro_f1_slice"])) < 5e-5
               for a in ("A0", "A1", "A2") for b in ("alexnet", "swin_t"))
    check("B/p3b_instr 的 A0/A1/A2 与 p3 逐位相同（共用 p3 训练产物，仅 A1r 新训）", same,
          "A1r=%s/%s" % (p3b[("A1r", "alexnet")], p3b[("A1r", "swin_t")]))
    return out


def main():
    os.makedirs(D_OUT, exist_ok=True)
    counts_a = task_a()
    rows_b = task_b(counts_a)
    print("\n== 折构成表（fold 的语义随臂不同：A0=fold_a0，A1/A2=fold，A1r=fold_b1）==")
    print("fold arm  n_tr_rec n_tr_slices n_te_rec n_te_slices ncls_tr ncls_te")
    for r in rows_b:
        print("%-4s %-4s %8s %11s %8s %11s %7s %7s" % tuple(r))
    bad = [c for c in CHECKS if not c[1]]
    print("\n%d checks, %d failed" % (len(CHECKS), len(bad)))
    if bad:
        for name, _, detail in bad:
            print("  FAILED:", name, detail)
        sys.exit(1)
    print("ALL PASS")
    print("written:", os.path.join(D_OUT, "slice_budget_by_class.csv"),
          "|", os.path.join(D_OUT, "p3_fold_composition.csv"))


if __name__ == "__main__":
    main()
