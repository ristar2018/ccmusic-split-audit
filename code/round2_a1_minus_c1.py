#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""round2_a1_minus_c1.py — CTIS 上 A1−C1 的直接配对区间（切片级/录音级），
并复算 A1−A0 / C1−A0 与已发表 CSV 逐字段对账（忠实性门禁）。

统计口径逐条对齐 code/analyze_e1.py（不另造一套）：
  * 宏 F1 = 混淆矩阵口径：tp=对角，fp=列和−tp，fn=行和−tp；keep = 该类在**本次（重采样）样本**里
    出现过（行和+列和>0），空类不参与平均（analyze_e1.py:96-106）。与 sklearn
    average='macro', zero_division=0 同口径。
  * 切片级：pred = probs.argmax(1)；录音级：同录音 probs 均值后 argmax 一票
    （analyze_e1.py:114-126）。
  * 录音簇自助 2000 次：重采样单位 = 录音簇（整簇保留/剔除，不拆簇）；RNG = default_rng(20261105)；
    按 sorted(库) 顺序、每库抽一次 size=(2000, n_rec)（analyze_e1.py:205-210），因此 CTIS 的
    自助索引矩阵与已发表 e1_delta.csv **同源**，A1−C1 与 A1−A0 共享同一组簇索引。
  * Δ = 每次重采样内先逐 seed 配对求差，再对 3 个 seed 取均值；Δ = F1_臂 − F1_参照，单位百分点。
    点估计同口径（3 seed 平均的逐 seed 配对差）。
  * 95% CI = Δ 自助分布的 2.5/97.5 百分位。
  * boot p 两个口径都给：boot_p = analyze_e1 现行双侧 +1（2*(k+1)/(B+1)，analyze_e1.py:258-260）；
    boot_p_legacy = 已发表 e1_delta.csv 的口径 2*min(k_le0,k_ge0)/B、下限 1/B（见函数注释）。
  * 忠实性：切片级与 e1_delta.csv 逐字段对账；录音级 CI 对 e1_delta.csv，**点估计**对
    human_review/corrected_recording_delta.csv 的 point_unw_pt（e1_delta.csv 的 delta_rec_pt 是已取代的
    legacy slice-weighted 值，稿件正文已改用 point_unw）；boot_p 按 legacy 口径复算。
  * 传递性：同一重抽样下 Δ(A1−C1) 必精确等于 Δ(A1−A0) − Δ(C1−A0)（脚本内断言）。
  * seed 极差 = 逐 seed Δ 的 max−min（只报数，不把 3 seed 当独立样本检验）。
  * 配对完整性（本脚本第 2 节）：同骨干同 seed 下 A0/A1/A2/C1 的 y 与 records 必须逐元素相同；
    跨 seed 的 y/records 也必须相同；每格宏 F1 必须与 seed{k}.json 的 qtest 记录一致（tol 5e-4）。

探索性分组（第 5 节）：按「测试录音是否出现在 A1 注入池」二分。注入暴露清单不另算，直接取冻结切分
  data/splits/v2/CTIS_{qpool,qtest,a1_train}.csv：a1_train 中 sample_id ∈ qpool 的行 = 实际注入切片，
  其 recording_canonical 与 qtest 的 recording_canonical 取交集 = 暴露测试录音（与 npz 的 records 列
  同名同值，脚本内断言；用 recording_uid 得到的交集数相同）。该口径与
  code/audit_review_exposure_20261108.py 相同，且与 paper/sections/10_supplement.tex 表
  tab:realised-injection 的 578/963 对得上（脚本内断言）。分组 Δ 的簇自助在组内重新抽簇，
  使用独立随机流 default_rng(20261105)（与原表的流不同源，仅组内可解释）。

只读：runs/sealed_eval/**、data/splits/v2/**、data/results/{e1_delta.csv,c1/e1_delta.csv}。
只写：data/results/human_review/round2/a1_minus_c1.{csv,json}。

用法：python code/round2_a1_minus_c1.py
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_e1 as ae  # noqa: E402  复用其 load_npz / macro_f1 / conf_slice / per_recording / discover

RUNS = os.path.join(ROOT, "runs", "sealed_eval")
SPLITS = os.path.join(ROOT, "data", "splits", "v2")
OUTDIR = os.path.join(ROOT, "data", "results", "human_review", "round2")
OUT_CSV = os.path.join(OUTDIR, "a1_minus_c1.csv")
OUT_JSON = os.path.join(OUTDIR, "a1_minus_c1.json")
PUB_A1A0 = os.path.join(ROOT, "data", "results", "e1_delta.csv")
PUB_C1A0 = os.path.join(ROOT, "data", "results", "c1", "e1_delta.csv")

BOOT = 2000
BOOT_SEED = 20261105
SESOI = 3.0
ARMS = ("A0", "A1", "A2", "C1")
SEEDS = (0, 1, 2)
BBS = ("alexnet", "swin_t")
CHUNK = 200

CSV_COLS = ["lib", "backbone", "level", "contrast", "delta_pp", "ci_lo", "ci_hi",
            "seed_deltas", "seed_range", "n_slices", "n_recordings"]


def die(msg):
    print("[PAIRING-FAIL] " + msg)
    sys.exit(2)


def macro_f1_batch(conf):
    """(B,C,C) 混淆矩阵 → (B,) 宏F1；与 ae.macro_f1 逐元素同口径（空类剔除）。"""
    tp = np.diagonal(conf, axis1=1, axis2=2).astype(np.float64)
    fp = conf.sum(axis=1) - tp          # 列和（pred=j 的总数，含 tp）
    fn = conf.sum(axis=2) - tp          # 行和（true=i 的总数，含 tp）
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(den > 0, 2 * tp / den, np.nan)
    keep = (conf.sum(axis=2) + conf.sum(axis=1)) > 0
    f1 = np.where(keep, f1, np.nan)
    return np.nanmean(f1, axis=1)


def batch_f1_slice(pred, y, nc, W):
    """切片级自助：W (B,n) 为每切片的簇重数权重。"""
    flat = y * nc + pred
    out = np.empty(W.shape[0])
    for st in range(0, W.shape[0], CHUNK):
        en = min(W.shape[0], st + CHUNK)
        wsum = np.zeros((en - st, nc * nc))
        for i, b in enumerate(range(st, en)):
            wsum[i] = np.bincount(flat, weights=W[b], minlength=nc * nc)
        out[st:en] = macro_f1_batch(wsum.reshape(en - st, nc, nc))
    return out


def batch_f1_rec(rec_y, rec_pred, nc, mult):
    """录音级自助：mult (B,R) 为每录音簇的重数权重。"""
    flat = rec_y * nc + rec_pred
    out = np.empty(mult.shape[0])
    for st in range(0, mult.shape[0], CHUNK):
        en = min(mult.shape[0], st + CHUNK)
        wsum = np.zeros((en - st, nc * nc))
        for i, b in enumerate(range(st, en)):
            wsum[i] = np.bincount(flat, weights=mult[b], minlength=nc * nc)
        out[st:en] = macro_f1_batch(wsum.reshape(en - st, nc, nc))
    return out


def boot_p(d):
    """双侧百分位 p，加 1 校正（analyze_e1.py:258-260 现行实现）：2*(k+1)/(B+1)。"""
    smaller = min(int(np.count_nonzero(d <= 0)), int(np.count_nonzero(d >= 0)))
    return min(1.0, 2.0 * (smaller + 1) / (d.size + 1))


def boot_p_legacy(d):
    """已发表 e1_delta.csv 的 boot_p 口径：2*min(#<=0,#>=0)/B，下限 1/B。

    依据：human_review/slice_p_addone.csv 的 n_draws_le0 与 p_legacy_clamped 反推
    （k=0 -> 2*0/2000 被下限 1/2000=0.0005 兜住；k=56 -> 0.056；k=462 -> 0.462；
    k=731 -> 0.731；k=20 -> 0.02），与 e1_delta.csv 四个 CTIS 单元逐一吻合。
    """
    k = min(int(np.count_nonzero(d <= 0)), int(np.count_nonzero(d >= 0)))
    return max(1.0 / d.size, min(1.0, 2.0 * k / d.size))


def read_csv_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    ae.RUNS = RUNS
    data = ae.discover()

    # ================= 1. 配对完整性断言 =================
    checks = []
    n_y = n_r = n_f1 = 0
    for bb in BBS:
        for s in SEEDS:
            ref = None
            for arm in ARMS:
                p = os.path.join(RUNS, "CTIS_%s_%s" % (arm, bb), "seed%d_qtest.npz" % s)
                if not os.path.isfile(p):
                    die("缺封存文件 %s" % p)
                probs, y, rec = ae.load_npz(p)
                if ref is None:
                    ref = (y, rec, arm)
                    checks.append({"check": "y/records 参照", "bb": bb, "seed": s, "arm": arm,
                                   "n": int(y.size), "n_recordings": int(np.unique(rec).size),
                                   "ok": True})
                else:
                    ok_y = bool(np.array_equal(y, ref[0]))
                    ok_r = bool(np.array_equal(rec, ref[1]))
                    n_y += 1
                    n_r += 1
                    if not ok_y:
                        die("%s/seed%d: %s 的 y 与 %s 不逐元素相同" % (bb, s, arm, ref[2]))
                    if not ok_r:
                        die("%s/seed%d: %s 的 records 与 %s 不逐元素相同" % (bb, s, arm, ref[2]))
                    checks.append({"check": "y/records 配对", "bb": bb, "seed": s, "arm": arm,
                                   "vs": ref[2], "ok": True})
                # 指标复算对账（对 seed{k}.json 的 qtest 记录）
                nc = int(y.max()) + 1
                f1s = ae.macro_f1(ae.conf_slice(y, probs.argmax(1), nc))[0]
                rec_y, rec_pred, _ = ae.per_recording(probs, y, np.unique(rec, return_inverse=True)[1],
                                                      int(np.unique(rec).size), nc)
                f1r = ae.macro_f1(ae.conf_slice(rec_y, rec_pred, nc))[0]
                st = json.loads(open(os.path.join(RUNS, "CTIS_%s_%s" % (arm, bb),
                                                  "seed%d.json" % s), encoding="utf-8").read())["qtest"]
                n_f1 += 2
                if abs(f1s - st["macro_f1_slice"]) > 5e-4 or abs(f1r - st["macro_f1_recording"]) > 5e-4:
                    die("%s/%s/seed%d: 复算 F1 (%.6f/%.6f) 与 seed json (%.6f/%.6f) 不符"
                        % (bb, arm, s, f1s, f1r, st["macro_f1_slice"], st["macro_f1_recording"]))
                checks.append({"check": "F1 复算对账", "bb": bb, "seed": s, "arm": arm,
                               "f1_slice": round(f1s, 6), "f1_recording": round(f1r, 6),
                               "json_slice": st["macro_f1_slice"],
                               "json_recording": st["macro_f1_recording"], "ok": True})
    # 跨 seed：切片集合必须稳定
    for bb in BBS:
        for s in SEEDS[1:]:
            _, y0, r0 = ae.load_npz(os.path.join(RUNS, "CTIS_A0_%s" % bb, "seed0_qtest.npz"))
            _, y, r = ae.load_npz(os.path.join(RUNS, "CTIS_A0_%s" % bb, "seed%d_qtest.npz" % s))
            if not (np.array_equal(y, y0) and np.array_equal(r, r0)):
                die("%s: seed%d 与 seed0 的切片集合/顺序漂移" % (bb, s))
            checks.append({"check": "跨 seed 切片集合", "bb": bb, "seed": s, "ok": True})
    print("[断言] 配对完整性 PASS：%d 个 y 比对 + %d 个 records 比对 + %d 项 F1 复算对账"
          % (n_y, n_r, n_f1))

    # ================= 2. 重放 analyze_e1 的自助随机流 =================
    rng = np.random.default_rng(BOOT_SEED)
    boot, pt = {}, {}   # (arm,bb,seed) -> (f1s_boot, f1r_boot) / (f1s, f1r)
    y_ref = rec_ref = None
    nrec = None
    for lib in sorted(data):
        per_arm = data[lib]
        if ("A0", "alexnet") not in per_arm:      # analyze_e1:155-158 同一跳库条件，保证流对齐
            continue
        z0 = sorted(per_arm[("A0", "alexnet")])[0]
        _, y_l, r_l = ae.load_npz(per_arm[("A0", "alexnet")][z0])
        recs, rec_idx = np.unique(r_l, return_inverse=True)
        nrec_l = recs.size
        pick = rng.integers(0, nrec_l, size=(BOOT, nrec_l))
        mult = np.stack([np.bincount(p, minlength=nrec_l) for p in pick])
        if lib != "CTIS":
            continue
        y_ref, rec_ref, nrec = y_l, r_l, nrec_l
        nc = int(y_ref.max()) + 1
        W = mult[:, rec_idx]                       # (B, n) 切片级簇权重

        for (arm, bb), seeds in sorted(per_arm.items()):
            for s in sorted(seeds):
                probs, y, rec = ae.load_npz(seeds[s])
                if not (np.array_equal(y, y_ref) and np.array_equal(rec, rec_ref)):
                    die("CTIS/%s/%s/seed%d 与 A0 参照切片顺序不一致" % (arm, bb, s))
                rec_y_s, rec_pred_s, _ = ae.per_recording(probs, y, rec_idx, nrec, nc)
                pred = probs.argmax(1)
                pt[(arm, bb, s)] = (ae.macro_f1(ae.conf_slice(y, pred, nc))[0],
                                    ae.macro_f1(ae.conf_slice(rec_y_s, rec_pred_s, nc))[0])
                boot[(arm, bb, s)] = (batch_f1_slice(pred, y, nc, W),
                                      batch_f1_rec(rec_y_s, rec_pred_s, nc, mult))
        # 批式实现与标量实现的一致性（抽 3 个重抽样核对）
        for (arm, bb, s) in [(k) for k in boot if k[0] == "A0" and k[1] == "alexnet"]:
            probs, y, _ = ae.load_npz(per_arm[(arm, bb)][s])
            flat = y * nc + probs.argmax(1)
            for b in (0, 7, 999):
                c = np.bincount(flat, weights=W[b], minlength=nc * nc).reshape(nc, nc)
                if abs(ae.macro_f1(c)[0] - boot[(arm, bb, s)][0][b]) > 1e-12:
                    die("批式/标量宏F1 实现不一致 (%s/%s/seed%d, b=%d)" % (arm, bb, s, b))
        print("[断言] 自助实现一致：CTIS 簇自助索引已按 analyze_e1 随机流重放，切片级批式=标量")

    # ================= 3. 对比 A1−C1 / A1−A0 / C1−A0 =================
    def contrast(num, ref, bb):
        seeds = sorted(set(k[2] for k in boot if k[0] == num and k[1] == bb)
                       & set(k[2] for k in boot if k[0] == ref and k[1] == bb))
        out = {}
        for i, level in ((0, "slice"), (1, "recording")):
            d_b = np.mean(np.stack([boot[(num, bb, s)][i] - boot[(ref, bb, s)][i] for s in seeds]), axis=0)
            pd = [100.0 * (pt[(num, bb, s)][i] - pt[(ref, bb, s)][i]) for s in seeds]
            lo, hi = np.percentile(100.0 * d_b, [2.5, 97.5])
            out[level] = {"delta_pp": float(np.mean(pd)), "ci_lo": float(lo), "ci_hi": float(hi),
                          "seed_deltas": [float(v) for v in pd],
                          "seed_range": float(max(pd) - min(pd)),
                          "boot_p": boot_p(d_b), "boot_p_legacy": boot_p_legacy(d_b),
                          "n_le0": int(np.count_nonzero(d_b <= 0)),
                          "n_ge0": int(np.count_nonzero(d_b >= 0)),
                          "boot": int(d_b.size), "n_seeds": len(seeds),
                          "seeds": seeds}
        return out

    contrasts = {}
    for bb in BBS:
        for num, ref in (("A1", "C1"), ("A1", "A0"), ("C1", "A0")):
            contrasts[("%s-%s" % (num, ref), bb)] = contrast(num, ref, bb)

    # 传递性自检：同一重抽样、同一 seed 配对下，Δ(A1−C1) 必精确等于 Δ(A1−A0) − Δ(C1−A0)
    tri = []
    for bb in BBS:
        for lv in ("slice", "recording"):
            a = contrasts[("A1-C1", bb)][lv]["delta_pp"]
            b = contrasts[("A1-A0", bb)][lv]["delta_pp"] - contrasts[("C1-A0", bb)][lv]["delta_pp"]
            tri.append({"backbone": bb, "level": lv, "a1_c1_pp": round(a, 6),
                        "a1_a0_minus_c1_a0_pp": round(b, 6), "abs_diff": abs(a - b)})
    assert all(t["abs_diff"] < 1e-9 for t in tri), tri

    # ================= 4. 忠实性门禁：与已发表 CSV 逐字段比对 =================
    # 口径说明（2026-11-11 复核）：
    #  (a) 切片级 Δ/CI/seed 极差/逐 seed Δ —— 直接对 e1_delta.csv（c1/e1_delta.csv）。
    #  (b) 录音级 CI —— 对 e1_delta.csv 的 ci_lo_rec_pt/ci_hi_rec_pt（簇自助分位，未受影响）；
    #      录音级**点估计**则对 human_review/corrected_recording_delta.csv 的 point_unw_pt：
    #      e1_delta.csv 里的 delta_rec_pt 是已登记的 legacy slice-weighted 点估计
    #      （= corrected_recording_delta.csv 的 legacy_point_sw_pt），稿件正文已改用 point_unw。
    #      因此 legacy 差异单列为信息项，不计入门禁。
    #  (c) boot_p —— e1_delta.csv 用 2*min(k_le0,k_ge0)/B、下限 1/B（见 boot_p_legacy 注释），
    #      与 analyze_e1.py 现行的双侧+1 口径不同；门禁按 CSV 口径比对，JSON 两个值都给。
    CORR = os.path.join(ROOT, "data", "results", "human_review", "corrected_recording_delta.csv")
    corr_rows = {(r["library"], r["bb"], r["contrast"]): r for r in read_csv_dicts(CORR)}

    def cmp_block(path, arm_pair):
        rows = {(r["library"], r["bb"]): r for r in read_csv_dicts(path)}
        det, ok_all = {}, True
        for bb in BBS:
            pub = rows.get(("CTIS", bb))
            corr = corr_rows.get(("CTIS", bb, arm_pair))
            if pub is None or corr is None:
                det["%s/%s" % (arm_pair, bb)] = {"missing": True}
                ok_all = False
                continue
            mine = contrasts[(arm_pair, bb)]
            fdet, ok = {}, True

            def put(name, pv, mv, tol, note=""):
                nonlocal ok
                same = abs(pv - mv) <= tol
                item = {"published": pv, "recomputed": round(float(mv), 4),
                        "abs_diff": round(abs(pv - mv), 4), "tol": tol, "match": bool(same)}
                if note:
                    item["note"] = note
                fdet[name] = item
                ok = ok and same

            put("delta_slice_pt", float(pub["delta_slice_pt"]), mine["slice"]["delta_pp"], 0.011)
            put("ci_lo_slice_pt", float(pub["ci_lo_slice_pt"]), mine["slice"]["ci_lo"], 0.011)
            put("ci_hi_slice_pt", float(pub["ci_hi_slice_pt"]), mine["slice"]["ci_hi"], 0.011)
            put("seed_range_pt", float(pub["seed_range_pt"]), mine["slice"]["seed_range"], 0.011)
            put("boot_p_slice", float(pub["boot_p_slice"]), mine["slice"]["boot_p_legacy"], 5e-4,
                "e1_delta.csv 口径 2*min(k_le0,k_ge0)/B；analyze_e1 现行双侧+1 值为 %.4f"
                % mine["slice"]["boot_p"])
            put("ci_lo_rec_pt", float(pub["ci_lo_rec_pt"]), mine["recording"]["ci_lo"], 0.011)
            put("ci_hi_rec_pt", float(pub["ci_hi_rec_pt"]), mine["recording"]["ci_hi"], 0.011)
            put("delta_rec_pt_unw_vs_corrected", float(corr["point_unw_pt"]),
                mine["recording"]["delta_pp"], 5e-3,
                "以 corrected_recording_delta.csv 的 point_unw_pt 为准（稿件正文用此值）")
            fdet["delta_rec_pt_legacy_in_e1_delta"] = {
                "published": float(pub["delta_rec_pt"]),
                "recomputed": round(mine["recording"]["delta_pp"], 4),
                "abs_diff": round(abs(float(pub["delta_rec_pt"]) - mine["recording"]["delta_pp"]), 4),
                "correction_pt": float(corr["correction_pt"]),
                "match": None,
                "note": "已知修正项，不计入门禁：e1_delta.csv 保留 legacy slice-weighted 点估计，"
                        "corrected_recording_delta.csv 的 correction_pt 即两者之差"}
            ps_mine = "|".join("%.2f" % v for v in mine["slice"]["seed_deltas"])
            same = ps_mine == pub["per_seed_delta_pt"]
            fdet["per_seed_delta_pt"] = {"published": pub["per_seed_delta_pt"],
                                         "recomputed": ps_mine, "match": bool(same)}
            ok = ok and same
            psr_mine = "|".join("%.2f" % v for v in mine["recording"]["seed_deltas"])
            same_r = psr_mine == corr["per_seed_unw_pt"]
            fdet["per_seed_unw_pt"] = {"published": corr["per_seed_unw_pt"],
                                       "recomputed": psr_mine, "match": bool(same_r),
                                       "note": "录音级逐 seed 对 corrected_recording_delta.csv"}
            ok = ok and same_r
            det["%s/%s" % (arm_pair, bb)] = {"fields": fdet, "all_match": bool(ok)}
            ok_all = ok_all and ok
        return {"source": os.path.relpath(path, ROOT).replace("\\", "/"),
                "recording_point_source": os.path.relpath(CORR, ROOT).replace("\\", "/"),
                "details": det, "all_fields_match": bool(ok_all)}

    fidelity = {"A1-A0": cmp_block(PUB_A1A0, "A1-A0"),
                "C1-A0": cmp_block(PUB_C1A0, "C1-A0")}
    fidelity["verdict"] = (
        "一致：切片级 Δ/CI/seed 与 e1_delta.csv 逐字段相同；录音级 CI 相同、点估计与 "
        "corrected_recording_delta.csv 的 point_unw_pt 相同；boot_p 按 e1_delta.csv 的 2k/B 口径复现相同"
        if fidelity["A1-A0"]["all_fields_match"] and fidelity["C1-A0"]["all_fields_match"]
        else "存在不一致字段：先查本脚本实现，不得改已发表数字")

    # ================= 5. 探索性分组：A1 注入池暴露 vs 未暴露 =================
    test_rows = read_csv_dicts(os.path.join(SPLITS, "CTIS_qtest.csv"))
    pool_rows = read_csv_dicts(os.path.join(SPLITS, "CTIS_qpool.csv"))
    a1_rows = read_csv_dicts(os.path.join(SPLITS, "CTIS_a1_train.csv"))
    pool_ids = {r["sample_id"] for r in pool_rows}
    injected = [r for r in a1_rows if r["sample_id"] in pool_ids]
    # 暴露口径与 code/audit_review_exposure_20261108.py、npz 的 records 列一致：用 recording_canonical
    inj_recs = {r["recording_canonical"] for r in injected}
    test_recs = {r["recording_canonical"] for r in test_rows}
    assert not ({r["sample_id"] for r in test_rows} & pool_ids), "qpool 与 qtest 的 sample_id 相交"
    assert set(rec_ref.tolist()) == test_recs, "npz 的 records 与 qtest recording_canonical 不一致"
    uid_x = {r["recording_uid"] for r in injected} & {r["recording_uid"] for r in test_rows}
    exposed = inj_recs & test_recs
    assert len(exposed) == 578, "暴露录音数应为 578（code/audit_review_exposure_20261108.py 实测）"
    group_info = {"source": "data/splits/v2/CTIS_{qpool,qtest,a1_train}.csv",
                  "rule": "a1_train 中 sample_id ∈ qpool 的行=实际注入切片；其 recording_canonical "
                          "∩ qtest recording_canonical 为暴露录音（与 npz records 列同名同值，已断言）",
                  "cross_check": {"audit_script": "code/audit_review_exposure_20261108.py",
                                  "audit_reported": "注入 4178 切片 / 578 暴露测试录音 / 963 测试录音",
                                  "uid_based_exposed_recordings": len(uid_x)},
                  "injected_slices": len(injected), "injected_recordings": len(inj_recs),
                  "test_recordings": len(test_recs), "exposed_test_recordings": len(exposed),
                  "unexposed_test_recordings": len(test_recs - exposed),
                  "paper_table_cell": "578/963 (paper/sections/10_supplement.tex:44)"}
    if not (len(exposed) == 578 and len(test_recs) == 963 and len(injected) == 4178):
        group_info["warn"] = "与 supplementary 表 578/963、4178 切片不一致，分组结果降级为未核验"
        print("[警告] 暴露清单与补充表数字不一致：%d/%d, injected=%d"
              % (len(exposed), len(test_recs), len(injected)))

    recs_all, rec_idx_all = np.unique(rec_ref, return_inverse=True)
    nc = int(y_ref.max()) + 1
    grp_rows = []
    rng_g = np.random.default_rng(BOOT_SEED)
    for gname, gset in (("exposed", exposed), ("unexposed", test_recs - exposed)):
        gmask = np.array([r in gset for r in recs_all])
        keep_slices = gmask[rec_idx_all]
        g_rec_idx = rec_idx_all[keep_slices]
        g_map = np.full(recs_all.size, -1)
        g_map[gmask] = np.arange(int(gmask.sum()))
        g_rec_idx = g_map[g_rec_idx]
        R_g = int(gmask.sum())
        y_g = y_ref[keep_slices]
        mult_g = np.stack([np.bincount(p, minlength=R_g)
                           for p in rng_g.integers(0, R_g, size=(BOOT, R_g))])
        W_g = mult_g[:, g_rec_idx]
        g_pt, g_boot = {}, {}
        for bb in BBS:
            for s in SEEDS:
                for arm in ("A0", "A1"):
                    probs, y, rec = ae.load_npz(os.path.join(
                        RUNS, "CTIS_%s_%s" % (arm, bb), "seed%d_qtest.npz" % s))
                    pred = probs[keep_slices].argmax(1)
                    rec_y_g, rec_pred_g, _ = ae.per_recording(probs[keep_slices], y_g, g_rec_idx, R_g, nc)
                    g_pt[(arm, bb, s)] = (ae.macro_f1(ae.conf_slice(y_g, pred, nc))[0],
                                          ae.macro_f1(ae.conf_slice(rec_y_g, rec_pred_g, nc))[0])
                    g_boot[(arm, bb, s)] = (batch_f1_slice(pred, y_g, nc, W_g),
                                            batch_f1_rec(rec_y_g, rec_pred_g, nc, mult_g))
            d_s = np.mean(np.stack([g_boot[("A1", bb, s)][0] - g_boot[("A0", bb, s)][0]
                                    for s in SEEDS]), axis=0)
            d_r = np.mean(np.stack([g_boot[("A1", bb, s)][1] - g_boot[("A0", bb, s)][1]
                                    for s in SEEDS]), axis=0)
            ps = [100.0 * (g_pt[("A1", bb, s)][0] - g_pt[("A0", bb, s)][0]) for s in SEEDS]
            pr = [100.0 * (g_pt[("A1", bb, s)][1] - g_pt[("A0", bb, s)][1]) for s in SEEDS]
            for level, d_b, pd in (("slice", d_s, ps), ("recording", d_r, pr)):
                lo, hi = np.percentile(100.0 * d_b, [2.5, 97.5])
                grp_rows.append({"lib": "CTIS", "backbone": bb, "level": level,
                                 "contrast": "A1-A0[%s]" % gname,
                                 "delta_pp": float(np.mean(pd)), "ci_lo": float(lo),
                                 "ci_hi": float(hi),
                                 "seed_deltas": [float(v) for v in pd],
                                 "seed_range": float(max(pd) - min(pd)),
                                 "boot_p": boot_p(d_b), "boot_p_legacy": boot_p_legacy(d_b),
                                 "n_le0": int(np.count_nonzero(d_b <= 0)),
                                 "n_ge0": int(np.count_nonzero(d_b >= 0)),
                                 "n_slices": int(keep_slices.sum()),
                                 "n_recordings": R_g,
                                 "group_definition": group_info["rule"]})
    group_info["rows"] = grp_rows
    _g = {(r["contrast"], r["level"]): r for r in grp_rows}
    print("[分组] exposed %d rec / %d slices | unexposed %d rec / %d slices"
          % (len(exposed), _g[("A1-A0[exposed]", "slice")]["n_slices"],
             len(test_recs - exposed), _g[("A1-A0[unexposed]", "slice")]["n_slices"]))

    # ================= 6. 落盘 =================
    rows = []

    def emit(contrast_name, bb, level, m, n_slices, n_recs):
        rows.append({"lib": "CTIS", "backbone": bb, "level": level, "contrast": contrast_name,
                     "delta_pp": round(m["delta_pp"], 2), "ci_lo": round(m["ci_lo"], 2),
                     "ci_hi": round(m["ci_hi"], 2),
                     "seed_deltas": "|".join("%.2f" % v for v in m["seed_deltas"]),
                     "seed_range": round(m["seed_range"], 2),
                     "n_slices": n_slices, "n_recordings": n_recs,
                     "boot_p": round(m.get("boot_p", float("nan")), 4),
                     "boot_p_legacy": round(m.get("boot_p_legacy", float("nan")), 4)})

    n_slices, n_recs = int(y_ref.size), int(nrec)
    for bb in BBS:
        for level in ("slice", "recording"):
            emit("A1-C1", bb, level, contrasts[("A1-C1", bb)][level], n_slices, n_recs)
    for bb in BBS:
        for level in ("slice", "recording"):
            emit("A1-A0", bb, level, contrasts[("A1-A0", bb)][level], n_slices, n_recs)
            emit("C1-A0", bb, level, contrasts[("C1-A0", bb)][level], n_slices, n_recs)
    for g in grp_rows:
        emit(g["contrast"], g["backbone"], g["level"],
             {"delta_pp": g["delta_pp"], "ci_lo": g["ci_lo"], "ci_hi": g["ci_hi"],
              "seed_deltas": g["seed_deltas"], "seed_range": g["seed_range"],
              "boot_p": g["boot_p"], "boot_p_legacy": g.get("boot_p_legacy", float("nan"))},
             g["n_slices"], g["n_recordings"])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "code/round2_a1_minus_c1.py",
        "inputs": {"sealed": "runs/sealed_eval/CTIS_{A0,A1,A2,C1}_{alexnet,swin_t}/seed{0,1,2}_qtest.npz",
                   "public": ["data/results/e1_delta.csv", "data/results/c1/e1_delta.csv"],
                   "splits": "data/splits/v2/CTIS_{qpool,qtest,a1_train}.csv"},
        "env": {"python": sys.executable, "numpy": np.__version__},
        "pairing_assertions": {
            "passed": True,
            "what": "同骨干同 seed 下 A0/A1/A2/C1 的 y 与 records 逐元素相同；跨 seed 切片集合相同；"
                    "每格宏 F1（切片级/录音级）与 seed{k}.json 的 qtest 记录一致（tol 5e-4）",
            "counts": {"y_compare": n_y, "records_compare": n_r, "f1_recompute": n_f1},
            "detail": checks},
        "method": {
            "metric": "macro-F1，混淆矩阵口径，空类不参与（analyze_e1.macro_f1）",
            "levels": {"slice": "全切片 argmax", "recording": "同录音 probs 均值后 argmax"},
            "bootstrap": {"n": BOOT, "cluster": "recording", "rng": "default_rng(%d)" % BOOT_SEED,
                          "stream": "按 sorted(lib) 逐库抽 size=(2000,n_rec)，与 analyze_e1.py:205-210 同源；"
                                    "CTIS 的索引矩阵与已发表 e1_delta.csv 相同",
                          "delta": "每次重抽样内逐 seed 配对求差后对 3 seed 取均值",
                          "ci": "Δ 分布的 2.5/97.5 百分位",
                          "p": "两个口径都给：boot_p=analyze_e1 现行双侧 +1（2*(k+1)/(B+1)）；"
                               "boot_p_legacy=e1_delta.csv 口径 2*min(k_le0,k_ge0)/B（下限 1/B）"},
            "units": "百分点（pp）",
            "sesoi_pp": SESOI,
            "seeds": list(SEEDS),
        },
        "contrasts": [{"lib": "CTIS", "backbone": bb, "contrast": c,
                       "level": lv, "delta_pp": round(contrasts[(c, bb)][lv]["delta_pp"], 4),
                       "ci_lo": round(contrasts[(c, bb)][lv]["ci_lo"], 4),
                       "ci_hi": round(contrasts[(c, bb)][lv]["ci_hi"], 4),
                       "seed_deltas_pp": [round(v, 4) for v in contrasts[(c, bb)][lv]["seed_deltas"]],
                       "seed_range_pp": round(contrasts[(c, bb)][lv]["seed_range"], 4),
                       "boot_p": round(contrasts[(c, bb)][lv]["boot_p"], 4),
                       "boot_p_legacy": round(contrasts[(c, bb)][lv]["boot_p_legacy"], 4),
                       "n_le0": contrasts[(c, bb)][lv]["n_le0"],
                       "n_ge0": contrasts[(c, bb)][lv]["n_ge0"],
                       "n_slices": n_slices, "n_recordings": n_recs,
                       "ci_excludes_zero": bool(contrasts[(c, bb)][lv]["ci_lo"] > 0
                                                or contrasts[(c, bb)][lv]["ci_hi"] < 0),
                       "meets_sesoi": bool(abs(contrasts[(c, bb)][lv]["delta_pp"]) >= SESOI)}
                      for c in ("A1-C1", "A1-A0", "C1-A0") for bb in BBS
                      for lv in ("slice", "recording")],
        "fidelity_check": fidelity,
        "transitivity_check": {"what": "Δ(A1−C1) = Δ(A1−A0) − Δ(C1−A0)（点估计，pp）",
                               "passed": True, "rows": tri},
        "exploratory_exposure": group_info,
        "notes": [
            "A1−C1 为本轮新增对比，此前无已发表数字；A1−A0 / C1−A0 用于忠实性门禁。",
            "忠实性：切片级与 e1_delta.csv 逐字段相同；录音级点估计以 human_review/"
            "corrected_recording_delta.csv 的 point_unw_pt 为准（稿件正文用此值），"
            "e1_delta.csv 的 delta_rec_pt 为已取代的 legacy slice-weighted 值（correction_pt 即两者之差）。",
            "boot_p 与 e1_delta.csv 不同是因为口径：CSV 用 2*min(k_le0,k_ge0)/B（下限 1/B），"
            "analyze_e1.py 现行为双侧 +1；两者都在 JSON 里列出，A1−C1 的显著性按 CI 判、p 仅参考。",
            "A1/C1 的 n_train 不同（A1=25498, C1=25502），差异是注入池实现，不是评估口径差异。",
            "分组 Δ 的宏 F1 只在组内出现的类别上取平均，两组类别宇宙不同，组间绝对值不可直接比较；"
            "组内 Δ 仍为配对差。",
            "分组自助用独立随机流 default_rng(%d)，与原表 CTIS 的流不同源。" % BOOT_SEED,
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    # ================= 7. 控制台摘要 =================
    print("\n%-8s %-10s %-20s %8s  %-18s %-18s %6s %6s %7s %7s" % (
        "backbone", "level", "contrast", "delta", "95%CI", "per-seed", "rng", "nRec", "p+1", "p_leg"))
    for r in rows:
        print("%-8s %-10s %-20s %+8.2f  [%+6.2f,%+6.2f]  %-18s %6.2f %6d %7.4f %7.4f" % (
            r["backbone"], r["level"], r["contrast"], r["delta_pp"], r["ci_lo"], r["ci_hi"],
            r["seed_deltas"], r["seed_range"], r["n_recordings"], r["boot_p"], r["boot_p_legacy"]))
    print("\n[门禁] %s" % fidelity["verdict"])
    print("写盘: %s | %s" % (OUT_CSV, OUT_JSON))
    return 0


if __name__ == "__main__":
    sys.exit(main())
