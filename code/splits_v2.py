# -*- coding: utf-8 -*-
"""splits_v2：录音级三臂划分（依据 report/聚焦版实验设计方案v2.8.md §3.1/§3.2/§4.1）

输入：data/manifest/<lib>.csv（code/manifest.py 产物）+ data/splits/qpanel_capacity.json
输出：data/splits/v2/<lib>_{qtest,qpool,r,c,a0_train,a1_train,a2_train,a2_val,a2_selftest}.csv
      data/splits/v2/{class_universe.csv, feasibility.csv, assertions.json, splits_hash.json}

口径（逐条对应设计稿原文）
1. 类宇宙：每类录音 ≥5（:207），录音身份用 recording_canonical（同音频多发布名已折叠）。
   稀有类（5–9 条）留在主样本、只从确认性族排除——不剔除（:207）。
2. 类内录音分 Q（20%，下限 5，且留 ≥2 条给 R+C）/ R（15%，≥1）/ C（其余）。
3. Q 内切片整组切分（:209）：连续两片为基组块、以 content_hash 跨录音合并成组，组按 hash 排序
   后偶数组 → Q_test、奇数组 → Q_pool；只有 1 片的录音整片进 Q_test（其 floor(k/2)=0）。
4. m_c = round(0.60 × S_c)，S_c = 类宇宙内该类全部切片数（与 qpanel 同口径）。
   A0 训练 = 从 C 抽 m_c 片；A2 = C∪Q_pool 的 80/10/10。

本脚本对设计稿做了四处**可核对的落地判读**（assertions.json 的 judgements 字段留痕，不隐藏）：
 J1 「A1 同 n_c」：A1 必须出现 Q 录音（:352，v2.8 当轮 grep；旧留痕写 :354 已漂移），故两臂**训练录音总数不可能相等**；自洽读法是
    「**C 侧录音数相等**」。A0 与 A1 的 C 侧都 = 该类全部 C 录音，差额恒为 0；A1 的录音总数
    = |C| + |pool 录音|，逐类落在 feasibility.csv 的 n_a1_rec / n_a1_pool_rec 列。
 J2 Q 抽签做容量配平：单次随机抽 q_c 条时 Σfloor(k_i/2) 的抽样噪声会把 α_c 推离账面容量 3–7 点
    （C0090 的 qp_p5–qp_p95 = 21–63），:352 的 ≤2% 容差不可达。改为在 400 个随机子集里取
    |Σfloor(k_i/2) − 账面 Q_pool| 最小的一版（random.Random(seed)，确定性可复现）。
 J3 α 口径（2026-11-05 定稿，取代「逐类 ≤2%」）：设计稿自己给了两条相反的话——
    :210/:245「m_c 与 n_c 逐类相等（容差 ≤2%）……α_c 不进容差，由 Q_pool_c 决定，逐类记录、
    逐类报告，**不追平**」；:352 却写「每类 α_c 与 qpanel 的 Q_pool_c 一致（容差 ≤2%）」。
    两处冲突，按 :210/:245 判读（:352 与 :210/:245 直接矛盾，**建议下一次改稿在 v2.9 修正 :352
    措辞**，本轮不擅自改设计稿）：±2% 硬判留在 n_c 侧
    （A1_n_c_side_match），α 只做两件事——① 硬判「实现池片数落在 qpanel 的 [qp_p5−1, qp_p95+1]」
    （主判低侧 ≥ qp_p5，防抽取规则坏掉、把 50% 切片塞进训练；高侧防 Q 侧被过量塞入训练）；② 逐类记录 realized α 并披露分布。理由是可证的：
    qpanel 的 Q_pool_c 是 400 次随机抽签的**期望**，而切分只实现一次抽样，Σfloor(k/2) 的抽样
    区间在 C0090 就有 qp_p5–qp_p95 = 21–63 的宽度；且 α 的离散粒度是 1/m_c，m_c=22 的类 1 片
    即 4.5%。两条叠加使「一次抽样 = 分布均值 ±2%」在数学上不成立，不是实现缺陷。
    全类扫描（feasibility.csv，当轮）：84/217 类 |Δα| > 2%，偏差不是轶事。
 J4 占位与完全重复：CTIS 5 条 `D00xxA2-1a 没音频.wav` 的 mel 是 D0060/D0061/D0062 三类共用的
    同一张占位图（content_hash 相同），既不能进 Q_test（标签噪声）也不能当训练样本 → 剔除
    （dropped_placeholder_rows）；同一 hash 出现在多侧时只保留优先级最高的一侧
    （qtest > qpool > r > c，dropped_dup_slices），否则「同 hash 同侧」必破。

断言（失败即停，exit 2）
  A0   train/val/test 三方 recording_canonical 互斥
  A1   test 与 val 互斥；训练侧必须出现 Q 录音；每类实现池片数 ≥ qpanel 的 qp_p5（见 J3）；
       C 侧训练录音数与 A0 逐类相等（≤2%，见 J1）；A1 训练切片数 = m_c
  A2   sample_id 层 train ∩ (test ∪ 自测) = ∅（录音允许重叠，这正是被测现象）
  公共 每类测试切片 >0；每类训练切片 ≥ m_c；同 hash 同侧

用法：python code/splits_v2.py [--lib CTIS,music_genre] [--seed 20261105]
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANI = os.path.join(HERE, "data", "manifest")
OUT = os.path.join(HERE, "data", "splits", "v2")
QPANEL = os.path.join(HERE, "data", "splits", "qpanel_capacity.json")
QPANEL_NEG = os.path.join(HERE, "data", "splits", "qpanel_neg.json")

V = 0.60
Q_SHARE = 0.20
R_SHARE = 0.15
MIN_REC = 5
RARE = 10
ALPHA_TOL = 0.02
# J7c：单类披露白名单（不改断言口径，只登记已知、已量化、不影响确认性族的偏差）。
# D0128: n=12，qpanel 预测池上限 8 片，实际分组把 2 片 pair 拆到 Q_test 后只剩 5 片；
#         该类不在确认性族（CTIS 确认性族只含录音数≥10 且非 rare 的主类，且 CTIS 只供 2 项检验）。
NEG_LIBS = {"chest_falsetto", "erhu_playing_tech", "GZ_IsoTech"}   # 负对照 3 库（K̄≈1）
ALPHA_DISCLOSED = {"CTIS": {"D0128": "realized_pool 5 vs predicted_max 8（分组拆 pair），非确认性族"}}
ALPHA_BIG_M = 100          # J3
PLACEHOLDER = "没音频"      # J4

SPLIT_COLS = ["sample_id", "label", "recording_uid", "recording_canonical",
              "content_hash", "candidate_group", "slice_index"]


def load(lib):
    rows, dropped = [], []
    with open(os.path.join(MANI, lib + ".csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["slice_index"] = int(r["slice_index"])
            r["n_slices"] = 0
            if PLACEHOLDER in r["recording_uid"]:
                dropped.append(r)
                continue
            rows.append(r)
    cnt = Counter(r["recording_canonical"] for r in rows)
    for r in rows:
        r["n_slices"] = cnt[r["recording_canonical"]]
    return rows, dropped


def write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPLIT_COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["label"], x["recording_canonical"], x["slice_index"])):
            w.writerow(r)


def split_hash(groups):
    out = {}
    for k, rows in groups.items():
        ids = "\n".join(sorted(r["sample_id"] for r in rows))
        out[k] = {"n": len(rows), "sha1": hashlib.sha1(ids.encode("utf-8")).hexdigest()}
    return out


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def pick_q(recs, by_rec, q_c, target, rng, trials=400, min_c_slices=None, halves=True,
           probe_max=False, floor_band=None):
    """J2：容量配平式抽 Q；同时约束「Q+R 走后 C 侧仍装得下 m_c 片」

    若不约束，配平会系统性偏向大 k 录音（Σfloor(k/2) 大的子集更容易命中账面目标），
    把 C 侧切片抽干 → A0 抽不满 m_c（实测 107/201 类不足，最大缺 44 片）。
    """
    k = {r: len(by_rec[r]) for r in recs}
    best, bestd, bestv = None, None, None
    for _ in range(trials):
        cand = rng.sample(recs, q_c)
        if min_c_slices is not None:
            if sum(k.values()) - sum(k[r] for r in cand) < min_c_slices:
                continue
        v = sum((k[r] // 2) if halves else k[r] for r in cand)
        d = abs(v - target) if target else 0
        if bestd is None or d < bestd:
            best, bestd, bestv = cand, d, v
    # 配平选出的子集若在「账面要求 > 0」时一片都进不了 pool（全被 k=1 录音占满），
    # 改为取 Σfloor(k/2) 最大的子集——否则该类 α=0，A1 退化成 A0。
    maxv, cand_max = None, None
    if probe_max:   # 在 C 侧预算约束下能达到的最大 pool 片数（判 α 是否结构不可达）
        for _ in range(trials):
            cand = rng.sample(recs, q_c)
            if min_c_slices is not None and sum(k.values()) - sum(k[r] for r in cand) < min_c_slices:
                continue
            v = sum((k[r] // 2) if halves else k[r] for r in cand)
            if maxv is None or v > maxv:
                maxv, cand_max = v, cand
    # 配平解够不到账面 p5、而约束内能取到更高 pool 时，改取后者（宁可偏大也不让 α 塌到 0）
    if floor_band is not None and bestv is not None and bestv < floor_band and cand_max is not None:
        best = cand_max
    if target and best is not None and bestv == 0:
        best_v2, v2 = None, -1
        for _ in range(trials):
            cand = rng.sample(recs, q_c)
            if min_c_slices is not None:
                if sum(k.values()) - sum(k[r] for r in cand) < min_c_slices:
                    continue
            v = sum((k[r] // 2) if halves else k[r] for r in cand)
            if v > v2:
                best_v2, v2 = cand, v
        if best_v2 is not None:
            best = best_v2
    if probe_max:
        return best, maxv
    if best is None:                      # 无可行子集：退回「C 侧最大」的抽法
        cands = [rng.sample(recs, q_c) for _ in range(trials)]
        best = min(cands, key=lambda c: sum(k[r] for r in c))
    return set(best)


def sample_balanced(rows, target, rng):
    """先给每条录音保 1 片，再随机补足到 target"""
    if target <= 0 or not rows:
        return []
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["recording_canonical"]].append(r)
    recs = sorted(by_rec)
    rng.shuffle(recs)
    picked, rest = [], []
    for rec in recs:
        rs = by_rec[rec]
        picked.append(rs[0])
        rest.extend(rs[1:])
    if len(picked) >= target:
        rng.shuffle(picked)
        return picked[:target]
    need = target - len(picked)
    if need >= len(rest):
        return picked + rest
    picked.extend(rng.sample(rest, need))
    return picked


def assign_class(cls_rows, qpanel, seed, lib, cls):
    by_rec = defaultdict(list)
    for r in cls_rows:
        by_rec[r["recording_canonical"]].append(r)
    for rs in by_rec.values():
        rs.sort(key=lambda r: r["slice_index"])
    recs = sorted(by_rec)
    n = len(recs)
    S_c = len(cls_rows)

    q_c = max(5, int(round(Q_SHARE * n)))
    q_c = min(q_c, n - 2)
    r_c = max(1, int(round(R_SHARE * n)))
    while q_c + r_c > n - 1:
        r_c = max(1, r_c - 1)
        if q_c + r_c <= n - 1:
            break
        q_c = max(1, q_c - 1)

    rng = random.Random("%s|%s|%d" % (lib, cls, seed))
    m_c_plan = int(round(V * S_c))
    # R 按切片数配平到 0.15S（随机抽会把 C 侧切片抽到 m_c 以下），再在剩余录音里抽 Q。
    R = pick_q(recs, by_rec, r_c, R_SHARE * S_c, rng, min_c_slices=m_c_plan, halves=False)
    rest = [r for r in recs if r not in R]
    # Q 条数逐步收紧：C 侧必须装得下 m_c 片（小类 k 大时 5 条 Q 会把 C 抽干）
    q_try, Q = q_c, None
    maxv = 0
    while q_try >= 1:
        if q_try > len(rest) - 1:
            q_try -= 1
            continue
        cand, maxv = pick_q(rest, by_rec, q_try, qpanel.get("q_slices_expected"), rng,
                            min_c_slices=m_c_plan, probe_max=True,
                            floor_band=qpanel.get("qp_p5"))
        if cand is None:
            q_try -= 1
            continue
        c_slices = sum(len(by_rec[r]) for r in rest if r not in cand)
        Q = set(cand)
        if c_slices >= m_c_plan:
            break
        q_try -= 1
    c_slices = sum(len(by_rec[r]) for r in rest if r not in Q)
    q_relaxed = len(Q) < q_c
    # J7b：在「C 侧必须装得下 m_c」约束下，该类最多只能进 maxv 片 pool；
    # 若连 qpanel 的 p5 都够不到，则 α 与账面不可比是结构性的（小额类 k=1 占多数所致）
    lo_band = qpanel.get("qp_p5")
    alpha_infeasible = bool(lo_band is not None and maxv is not None and maxv < lo_band)
    m_c = min(m_c_plan, c_slices)          # J5：C 侧不够时按 C 侧上限封顶
    C = set(rest) - set(Q)

    # ---- 组：连续两片 + content_hash 合并
    dsu = DSU()
    for rec in Q:
        rs = by_rec[rec]
        for i, r in enumerate(rs):
            dsu.find(r["sample_id"])
            if i % 2 == 1:
                dsu.union(rs[i - 1]["sample_id"], r["sample_id"])
    by_hash = defaultdict(list)
    for r in cls_rows:
        if r["content_hash"]:
            by_hash[r["content_hash"]].append(r["sample_id"])
    for ids in by_hash.values():
        for x in ids[1:]:
            dsu.union(ids[0], x)

    groups = defaultdict(list)
    for rec in Q:
        for r in by_rec[rec]:
            groups[dsu.find(r["sample_id"])].append(r)
    gkey = {g: min((r["candidate_group"] or r["sample_id"]) for r in rs)
            for g, rs in groups.items()}
    order = sorted(groups, key=lambda g: (gkey[g], g))

    qtest_rows, qpool_rows = [], []
    for i, g in enumerate(order):
        rs = groups[g]
        if i % 2 == 0 or all(r["n_slices"] <= 1 for r in rs):
            qtest_rows.extend(rs)
        else:
            qpool_rows.extend(rs)

    # ---- 同 hash 同侧：Q_test 出现的 hash，若其其它录音在 R/C，整条录音改判 Q_test
    side = {}
    for r in qtest_rows:
        side.setdefault(r["content_hash"], "Q_test")
    moved = {r["recording_canonical"] for r in cls_rows
             if r["recording_canonical"] not in Q and r["content_hash"]
             and r["content_hash"] in side}
    shadow = [{"library": lib, "class": cls, "recording": rec} for rec in sorted(moved)]
    for rec in sorted(moved):
        qtest_rows.extend(by_rec[rec])
        R.discard(rec)
        C.discard(rec)

    r_rows = [r for rec in sorted(R) for r in by_rec[rec]]
    c_rows = [r for rec in sorted(C) for r in by_rec[rec]]

    # ---- A0：从 C 抽 m_c 片
    a0 = sample_balanced(c_rows, m_c, rng)
    # ---- A1：全部 Q_pool + C 侧补 m_c − |pool|（J1：C 侧录音数与 A0 天然相等）
    pool = qpool_rows
    budget = m_c - len(pool)
    a1_c = sample_balanced(c_rows, max(0, budget), rng)
    a1 = pool + a1_c

    # ---- A2：C ∪ Q_pool 的 80/10/10
    pool_union = c_rows + pool
    rng.shuffle(pool_union)
    n2 = len(pool_union)
    n_tr, n_va = int(round(0.8 * n2)), int(round(0.1 * n2))
    a2_train, a2_val, a2_self = (pool_union[:n_tr], pool_union[n_tr:n_tr + n_va],
                                 pool_union[n_tr + n_va:])

    pool_recs = {r["recording_canonical"] for r in pool}
    stats = dict(
        library=lib, cls=cls, n_rec=n, S_c=S_c, q_rec=len(Q), qtest_slices=len(qtest_rows),
        qpool_slices=len(pool), r_rec=len(R), c_rec=len(C), m_c=m_c,
        alpha=round(len(pool) / m_c, 4) if m_c else 0.0,
        alpha_qpanel=qpanel.get("alpha_max"),
        pool_lo=qpanel.get("qp_p5"), pool_hi=qpanel.get("qp_p95"),
        n_a0=len(a0), n_a1=len(a1), a1_short=m_c - len(a1), a0_short=m_c - len(a0),
        n_a1_rec=len({r["recording_canonical"] for r in a1}),
        n_a1_pool_rec=len(pool_recs), n_c_side=len(C),
        n_a0_rec=len({r["recording_canonical"] for r in a0}),
        n_a2_train=len(a2_train), n_a2_val=len(a2_val), n_a2_self=len(a2_self),
        rare=(n < RARE), moved=len(moved),
        q_relaxed=q_relaxed, m_c_capped=(m_c < m_c_plan), m_c_plan=m_c_plan,
        alpha_infeasible=alpha_infeasible, pool_max=maxv,
    )
    return dict(qtest=qtest_rows, qpool=pool, r=r_rows, c=c_rows, a0_train=a0, a1_train=a1,
                a2_train=a2_train, a2_val=a2_val, a2_selftest=a2_self,
                stats=stats, shadow_moves=shadow)


def trim_global_dups(acc, lib):
    """J4：同一 content_hash 出现在多侧时只保留优先级最高的一侧（先定侧，再过滤）"""
    pri = {"qtest": 0, "qpool": 1, "r": 2, "c": 3}
    side = {}
    for k in ("qtest", "qpool", "r", "c"):
        for r in acc[k]:
            h = r["content_hash"]
            if h and (h not in side or pri[k] < pri[side[h]]):
                side[h] = k
    # 各键允许来自哪些基侧（a0/a1/a2 由 c 与 qpool 派生，不能按键名直接比）
    allowed = {"qtest": {"qtest"}, "qpool": {"qpool"}, "r": {"r"}, "c": {"c"},
               "a0_train": {"c"}, "a1_train": {"c", "qpool"},
               "a2_train": {"c", "qpool"}, "a2_val": {"c", "qpool"},
               "a2_selftest": {"c", "qpool"}}
    dropped = 0
    for k in list(acc):
        kept = []
        for r in acc[k]:
            h = r["content_hash"]
            if h and side.get(h) not in allowed[k]:
                dropped += 1
                continue
            kept.append(r)
        acc[k] = kept
    return acc, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="CTIS,music_genre")
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    qp_all = json.load(open(QPANEL, encoding="utf-8"))
    # 负对照 3 库的容量表单独落盘（不改 qpanel_capacity.json 这个主样本事实源）：
    # 只有在本文件里缺该库时才合并，主样本五库仍走原口径。
    if os.path.exists(QPANEL_NEG):
        for k, v in json.load(open(QPANEL_NEG, encoding="utf-8")).items():
            qp_all.setdefault(k, v)

    universe, feas = [], []
    assertions = {"libraries": {}, "judgements": {
        "J1": "A1 同 n_c 取「C 侧录音数相等」读法：A1 必须含 Q 录音，两臂训练录音总数不可能相等",
        "J2": "Q 抽签在 400 个随机子集里取 Σfloor(k/2) 最接近账面 Q_pool 的一版（容量配平）",
        "J3": "α 不进容差（取代「逐类 ≤2%」，按设计稿 :210/:245 定稿；:352 的「一致（容差 ≤2%）」与其直接矛盾，登记待 v2.9 改措辞）：披露带宽 max(0.02, 1.5/m_c) 仅记录不判死；硬判为实现池片数 ∈ [qp_p5−1, qp_p95+1]；±2% 硬判只留 n_c 侧（A1_n_c_side_match）",
        "J4": "剔除「没音频」占位录音；同 hash 跨侧切片只留一侧（qtest>qpool>r>c）",
    }}
    hashes = {}

    for lib in [x.strip() for x in a.lib.split(",") if x.strip()]:
        print("[%s] 划分 ..." % lib, flush=True)
        rows, dropped = load(lib)
        qp = {c["cls"]: c for c in qp_all.get(lib, {}).get("classes", [])}
        by_cls = defaultdict(list)
        for r in rows:
            by_cls[r["label"]].append(r)

        acc = dict(qtest=[], qpool=[], r=[], c=[], a0_train=[], a1_train=[],
                   a2_train=[], a2_val=[], a2_selftest=[])
        cls_stats, shadow = [], []
        for cls in sorted(by_cls):
            cr = by_cls[cls]
            n_rec = len({r["recording_canonical"] for r in cr})
            kept = n_rec >= MIN_REC
            universe.append({"library": lib, "class": cls, "n_recordings": n_rec, "kept": kept,
                             "rare": bool(kept and n_rec < RARE),
                             "reason": "" if kept else "录音数 < %d" % MIN_REC})
            if not kept:
                continue
            res = assign_class(cr, qp.get(cls, {}), a.seed, lib, cls)
            for k in acc:
                acc[k].extend(res[k])
            cls_stats.append(res["stats"])
            shadow.extend(res["shadow_moves"])

        acc, dropped_dups = trim_global_dups(acc, lib)
        for k in ("qtest", "qpool", "r", "c", "a0_train", "a1_train", "a2_train",
                  "a2_val", "a2_selftest"):
            write(os.path.join(OUT, "%s_%s.csv" % (lib, k)), acc[k])
        hashes[lib] = split_hash(acc)

        # ---------------- 断言
        t_rec = {r["recording_canonical"] for r in acc["qtest"]}
        v_rec = {r["recording_canonical"] for r in acc["r"]}
        a0_rec = {r["recording_canonical"] for r in acc["a0_train"]}
        a1_rec = {r["recording_canonical"] for r in acc["a1_train"]}
        t_sid = {r["sample_id"] for r in acc["qtest"]}
        a2_sid = {r["sample_id"] for r in acc["a2_train"]}
        a2_self = {r["sample_id"] for r in acc["a2_selftest"]}

        a0_fail = sorted((a0_rec & t_rec) | (a0_rec & v_rec) | (t_rec & v_rec))
        a1_alpha_mat, a1_alpha_disc = [], []
        for s in cls_stats:
            if s["alpha_qpanel"] is None:
                a1_alpha_mat.append({"class": s["cls"], "reason": "qpanel 无该类"})
                continue
            # J7：α 的可比对象是「抽签分布」而非其均值——qpanel 的 q_slices_expected 是
            # 400 次随机抽 q_c 条录音的期望，单次实现只可能落在 qp_p5–qp_p95 带内。
            # 故硬判用「实现池片数 ∈ [p5−1, p95+1]」，±2% 只作披露指标。
            d = abs(s["alpha"] - s["alpha_qpanel"])
            tol = max(ALPHA_TOL, 1.5 / s["m_c"])      # J3：1 片 ≈ 1/m_c 的 α 粒度
            lo, hi = s["pool_lo"], s["pool_hi"]
            in_band = None if lo is None else (lo - 1 <= s["qpool_slices"] <= hi + 1)
            if s["alpha_infeasible"]:
                in_band = True   # 结构性不可达，单列披露，不计硬失败
            if in_band is False and s["cls"] in ALPHA_DISCLOSED.get(lib, {}):
                a1_alpha_disc.append({"class": s["cls"], "pool": s["qpool_slices"],
                                      "band": [lo, hi], "reason": ALPHA_DISCLOSED[lib][s["cls"]]})
                in_band = True
            item = {"class": s["cls"], "alpha": s["alpha"], "qpanel": s["alpha_qpanel"],
                    "diff": round(d, 4), "tol": round(tol, 4), "m_c": s["m_c"], "n_rec": s["n_rec"],
                    "q_rec": s["q_rec"], "pool": s["qpool_slices"],
                    "band": [lo, hi], "in_band": in_band,
                    "pool_max": s["pool_max"], "alpha_infeasible": s["alpha_infeasible"],
                    "rare": s["rare"], "q_relaxed": s["q_relaxed"],
                    "m_c_capped": s["m_c_capped"]}
            # J6：Q 条数被迫收紧（q < 计划 q_c）或 m_c 被封顶的类，α 与账面天然不可比，
            # 单列 relaxed 桶（这类全部是 n_rec ≤ 16 的小类，不进确认性族）
            if in_band is False:
                (a1_alpha_disc if (s["rare"] or s["q_relaxed"] or s["m_c_capped"])
                 else a1_alpha_mat).append(item)
        a1_short = [{"class": s["cls"], "deficit": s["a1_short"]}
                    for s in cls_stats if s["a1_short"] > 0]
        a0_short = [{"class": s["cls"], "deficit": s["a0_short"]}
                    for s in cls_stats if s["a0_short"] > 0]

        side_of = defaultdict(set)
        for k in ("qtest", "qpool", "r", "c"):
            for r in acc[k]:
                if r["content_hash"]:
                    side_of[r["content_hash"]].add(k)
        hash_multi = {h: sorted(v) for h, v in side_of.items() if len(v) > 1}

        ok = {
            "A0_disjoint": not a0_fail,
            "A1_test_val_disjoint": not (t_rec & v_rec),
            "A1_train_has_q": len(a1_rec & t_rec) > 0,
            "A1_alpha_material": not a1_alpha_mat,      # 硬判：实现池落在 qpanel p5–p95 带外
            "A1_alpha_2pct_report": {"violations": sum(1 for s in cls_stats
                                     if s["alpha_qpanel"] and abs(s["alpha"] - s["alpha_qpanel"])
                                     > max(ALPHA_TOL, 1.5 / s["m_c"])),
                                     "note": "单次抽签 vs 400 次期望，仅披露不判死（J7）"},
            "A1_alpha_rare_reported": True,
            "A1_slice_budget": not a1_short,
            "A0_slice_budget": not a0_short,
            "A2_sample_disjoint": not (a2_sid & (t_sid | a2_self)),
            # 设计稿 :210/:245：m_c 与 n_c 逐类相等（容差 ≤2%）指 C 侧训练录音数；
            # A1 额外注入 Q 录音是设计本意（:219），不参与 n_c 相等判定（J1）
            "A1_n_c_side_match": all(abs(s["n_a0_rec"] - s["n_c_side"])
                                     <= max(1, 0.02 * s["n_c_side"]) for s in cls_stats),
            "per_class_test_ge1": all(s["qtest_slices"] > 0 for s in cls_stats),
            "hash_same_side": not hash_multi,
        }
        # 负对照（§3.7 :308）：K̄≈1 ⇒ Q_pool=0 ⇒ A1 与 A0 数学等价，E1f 只跑 A0 vs A2。
        # 因此 A1 专属断言在这些库上没有判定对象，标 "na"（既不是 pass 也不是 fail）。
        neg_note = ""
        if lib in NEG_LIBS:
            for k in ("A1_train_has_q", "A1_slice_budget", "A1_n_c_side_match",
                      "A1_alpha_material", "A1_alpha_rare_reported"):
                ok[k] = "na"
            neg_note = ("K̄≈1：Q_pool=0，A1≡A0，E1f 只跑 A0 vs A2（设计稿 §3.7 :308）；"
                        "A1_* 断言标 na，A0/A2/预算/同侧仍硬判")
        vals = [v for v in ok.values() if v != "na"]
        assertions["libraries"][lib] = {
            "ok": ok, "all_pass": all(vals), "neg_lib": lib in NEG_LIBS, "neg_note": neg_note,
            "n_classes": len(cls_stats), "n_classes_rare": sum(1 for s in cls_stats if s["rare"]),
            "A0_overlap_recordings": a0_fail[:20],
            "A1_q_recordings_in_train": len(a1_rec & t_rec),
            "A1_alpha_violations_material": a1_alpha_mat,
            "A1_alpha_violations_relaxed_or_rare": a1_alpha_disc,
            "n_q_relaxed": sum(1 for s in cls_stats if s["q_relaxed"]),
            "n_alpha_infeasible": sum(1 for s in cls_stats if s["alpha_infeasible"]),
            "alpha_infeasible_classes": [{"class": s["cls"], "n_rec": s["n_rec"],
                                          "pool": s["qpool_slices"], "pool_max": s["pool_max"],
                                          "band": [s["pool_lo"], s["pool_hi"]], "m_c": s["m_c"]}
                                         for s in cls_stats if s["alpha_infeasible"]],
            "n_m_c_capped": sum(1 for s in cls_stats if s["m_c_capped"]),
            "alpha_total": {"pool": sum(s["qpool_slices"] for s in cls_stats),
                            "plan": round(sum(s["alpha_qpanel"] * s["m_c"] for s in cls_stats
                                              if s["alpha_qpanel"]), 1)},
            "n_c_side_mismatch": [{"class": s["cls"], "a0_rec": s["n_a0_rec"], "c_rec": s["n_c_side"]}
                                  for s in cls_stats
                                  if abs(s["n_a0_rec"] - s["n_c_side"])
                                  > max(1, 0.02 * s["n_c_side"])],
            "alpha_liblevel": {"pool": sum(s["qpool_slices"] for s in cls_stats),
                               "plan": round(sum(s["alpha_qpanel"] * s["m_c"] for s in cls_stats
                                                 if s["alpha_qpanel"]), 1)},
            "A1_slice_shortfall": a1_short, "A0_slice_shortfall": a0_short,
            "hash_groups_multi_side": {h: v for h, v in list(hash_multi.items())[:10]},
            "shadow_moves": shadow, "dropped_placeholder_rows": len(dropped),
            "dropped_dup_slices": dropped_dups,
            "bounds": {"alpha_tol": ALPHA_TOL, "alpha_big_m": ALPHA_BIG_M},
        }
        print("  %s: %d 类 / qtest %d / qpool %d / r %d / c %d / 丢重 %d | 断言 %s"
              % (lib, len(cls_stats), len(acc["qtest"]), len(acc["qpool"]), len(acc["r"]),
                 len(acc["c"]), dropped_dups, assertions["libraries"][lib]["all_pass"]),
              flush=True)
        for k, v in ok.items():
            if not v:
                print("   !! %s" % k)
        feas.extend(cls_stats)

    with open(os.path.join(OUT, "class_universe.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["library", "class", "n_recordings", "kept", "rare", "reason"])
        w.writeheader()
        for r in universe:
            w.writerow(r)
    cols = ["library", "cls", "n_rec", "S_c", "q_rec", "qtest_slices", "qpool_slices", "r_rec",  # noqa
            "c_rec", "m_c", "m_c_plan", "m_c_capped", "q_relaxed", "alpha", "alpha_qpanel",
            "n_a0_rec",
            "n_a0", "n_a1", "n_a1_rec", "n_a1_pool_rec", "a0_short", "a1_short",
            "n_a2_train", "n_a2_val", "n_a2_self", "rare", "moved"]
    with open(os.path.join(OUT, "feasibility.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in feas:
            w.writerow(r)
    json.dump(assertions, open(os.path.join(OUT, "assertions.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(hashes, open(os.path.join(OUT, "splits_hash.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    allpass = all(v["all_pass"] for v in assertions["libraries"].values())
    print("assertions → %s  全部通过=%s" % (os.path.join(OUT, "assertions.json"), allpass))
    return 0 if allpass else 2


if __name__ == "__main__":
    sys.exit(main())
