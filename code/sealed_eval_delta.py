#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1 vs A0 封存评测：录音簇自助 2000 次 → 切片级/录音级 Δ 的 95%CI、3 seed 极差、SESOI(3pp) 判定、缺失组申报。

分组目录：runs/sealed_eval/<lib>_<arm>_<bb>/   （lib/bb 可含 '_'：bel_canto / swin_t）
  分组身份取自 SEALED.json -> runs.seed0.config 的 lib/arm/bb，并回验 dirname == f"{lib}_{arm}_{bb}"；
  不做 split('_')。seed{k}_qtest.npz: probs(n,C) / y(n,) / records(n,)=录音簇。
指标：slice 级 = 全切片宏F1；recording 级 = 每簇 prob 均值 argmax 后宏F1（与 seed{k}.json 的 qtest 对齐）。
自助：以录音簇为重采样单位，两臂与 3 个 seed 共用同一组簇索引（配对），Δ=F1_A1-F1_A0，单位百分点。
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np


def load_npz(p):
    z = np.load(p, allow_pickle=True)
    return z["probs"].astype(np.float64), z["y"].astype(np.int64), z["records"].astype(str)


def macro_f1(pred, y, C):
    cm = np.zeros((C, C))
    np.add.at(cm, (y, pred), 1.0)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    return float(f1.mean())


def brick(Mr, codes, C, mask):
    """Mr: (R,K) 簇×pair 计数。返回 f(W)->(B,) 的宏F1 向量。"""
    K = Mr.shape[1]
    tc = np.zeros((1, 1))

    def f(W):
        A = W @ Mr
        B = A.shape[0]
        tp = np.zeros((B, C)); fp = np.zeros((B, C)); fn = np.zeros((B, C))
        for j in range(K):
            t, p = divmod(int(codes[j]), C)
            if t == p:
                tp[:, t] += A[:, j]
            else:
                fp[:, p] += A[:, j]; fn[:, t] += A[:, j]
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
        return f1[:, mask].mean(1)
    return f


def rec_labels(probs, y, gidx, R, C):
    M = np.zeros((R, C)); np.add.at(M, gidx, probs)
    return M.argmax(1)


def discover(root):
    groups, stray = {}, []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = d / "SEALED.json"
        if not sp.is_file():
            stray.append((d.name, "非封存分组（无 SEALED.json，如 pending/）"))
            continue
        sg = json.loads(sp.read_text(encoding="utf-8"))
        cfg = sg["runs"]["seed0"]["config"]
        key = (cfg["lib"], cfg["arm"], cfg["bb"])
        if d.name != "_".join(key):
            stray.append((d.name, "目录名与 config %s/%s/%s 不一致" % key))
        groups[key] = (d, sg)
    pd = root / "pending"
    if pd.is_dir():
        for x in sorted(y for y in pd.iterdir() if y.is_dir()):
            ss = sorted(int(f.name[4:].split("_")[0]) for f in x.glob("seed*_qtest.npz"))
            if ss:
                cj = sorted(x.glob("seed*.json"))
                tag = ""
                if cj:
                    c = json.loads(cj[0].read_text(encoding="utf-8"))["config"]
                    tag = "%s/%s/%s" % (c["lib"], c["arm"], c["bb"])
                stray.append(("pending/" + x.name, "未封存 cfg=%s seeds=%s" % (tag, ss)))
    return groups, stray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/sealed_eval")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--sesoi", type=float, default=3.0)
    ap.add_argument("--boot-seed", type=int, default=20261105)
    ap.add_argument("--out", default="report/tables/sealed_eval_delta.csv")
    ap.add_argument("--missing-out", default="report/tables/sealed_eval_missing.csv")
    args = ap.parse_args()
    root = Path(args.root)
    groups, stray = discover(root)
    arms = ["A0", "A1"]
    for k in sorted(groups):
        if k[1] not in arms:
            stray.append(("_".join(k), "臂 %s 不在 A0/A1 比较网格，未纳入（仅登记）" % k[1]))
    libs = sorted({k[0] for k in groups}); bbs = sorted({k[2] for k in groups})
    seeds = sorted({int(s) for d, _ in groups.values()
                    for s in json.loads((d / "SEALED.json").read_text(encoding="utf-8"))["seeds"]})
    miss = []
    for l in libs:
        for b in bbs:
            for m in arms:
                if (l, m, b) not in groups:
                    miss.append((l, m, b, "", "整组缺失")); continue
                d, _ = groups[(l, m, b)]
                for s in seeds:
                    for f in ("seed%d.json" % s, "seed%d_qtest.npz" % s):
                        if not (d / f).is_file():
                            miss.append((l, m, b, s, "缺 " + f))

    rows, notes = [], []
    for l in libs:
        for b in bbs:
            if (l, "A0", b) not in groups or (l, "A1", b) not in groups:
                notes.append("%s/%s: 缺臂" % (l, b)); continue
            have = [s for s in seeds
                    if (groups[(l, "A0", b)][0] / ("seed%d_qtest.npz" % s)).is_file()
                    and (groups[(l, "A1", b)][0] / ("seed%d_qtest.npz" % s)).is_file()]
            if not have:
                notes.append("%s/%s: 0 个完整 seed" % (l, b)); continue
            probs, y, rec = {0: {}, 1: {}}, None, None
            for s in have:
                p0, y0, r0 = load_npz(groups[(l, "A0", b)][0] / ("seed%d_qtest.npz" % s))
                p1, y1, r1 = load_npz(groups[(l, "A1", b)][0] / ("seed%d_qtest.npz" % s))
                if not (np.array_equal(y0, y1) and np.array_equal(r0, r1)):
                    sys.exit("[审计失败] %s/%s/seed%d: 两臂 y/records 未对齐" % (l, b, s))
                if y is None:
                    y, rec = y0, r0
                elif not (np.array_equal(y, y0) and np.array_equal(rec, r0)):
                    sys.exit("[审计失败] %s/%s: 跨 seed 切片集合漂移" % (l, b))
                probs[0][s], probs[1][s] = p0, p1
                C = p0.shape[1]
                for arm, pp in (("A0", p0), ("A1", p1)):
                    st = json.loads((groups[(l, arm, b)][0] / ("seed%d.json" % s))
                                    .read_text(encoding="utf-8"))["qtest"]
                    val = macro_f1(pp.argmax(1), y0, C)
                    if abs(val - st["macro_f1_slice"]) > 5e-4:
                        sys.exit("[审计失败] %s/%s/%s/seed%d: 复算 slice 宏F1 %.4f != %.4f"
                                 % (l, b, arm, s, val, st["macro_f1_slice"]))
            C = probs[0][have[0]].shape[1]
            rid, gidx = np.unique(rec, return_inverse=True)
            R = rid.size
            mask = np.bincount(y, minlength=C) > 0
            ry = np.array([y[gidx == r][0] for r in range(R)])
            rlab = {0: {}, 1: {}}
            for s in have:
                for arm in (0, 1):
                    rlab[arm][s] = rec_labels(probs[arm][s], y, gidx, R, C)

            rng = np.random.default_rng(
                int(hashlib.md5(("%s|%s" % (l, b)).encode()).hexdigest()[:8], 16) ^ args.boot_seed)
            idx = rng.integers(0, R, size=(args.n_boot, R))
            W = np.zeros((args.n_boot, R))
            np.add.at(W, (np.repeat(np.arange(args.n_boot), R), idx.ravel()), 1.0)

            res = {}
            for level in ("slice", "rec"):
                t_all, p_all, boots, pts = [], [], {}, {}
                for arm in (0, 1):
                    for s in have:
                        pred = probs[arm][s].argmax(1) if level == "slice" else rlab[arm][s]
                        t = y if level == "slice" else ry
                        g = gidx if level == "slice" else np.arange(R)
                        pts[(arm, s)] = macro_f1(pred, t, C)
                        t_all.append(t); p_all.append(pred)
                codes = sorted({int(a) * C + int(pp) for t, pred in zip(t_all, p_all)
                                for a, pp in zip(t, pred)})
                c2j = {c: j for j, c in enumerate(codes)}
                for arm in (0, 1):
                    for s in have:
                        pred = probs[arm][s].argmax(1) if level == "slice" else rlab[arm][s]
                        t = y if level == "slice" else ry
                        g = gidx if level == "slice" else np.arange(R)
                        M = np.zeros((R, len(codes)))
                        np.add.at(M, (g, [c2j[int(a) * C + int(pp)] for a, pp in zip(t, pred)]), 1.0)
                        boots[(arm, s)] = brick(M, codes, C, mask)(W)
                d_boot = np.mean([boots[(1, s)] - boots[(0, s)] for s in have], axis=0)
                seed_pt = [pts[(1, s)] - pts[(0, s)] for s in have]
                d_pt = float(np.mean(seed_pt))
                lo, hi = np.percentile(d_boot * 100, [2.5, 97.5])
                res[level] = dict(d=d_pt * 100, lo=float(lo), hi=float(hi),
                                  sr=(max(seed_pt) - min(seed_pt)) * 100)

            def flag(d, lo, hi):
                if lo > args.sesoi or hi < -args.sesoi:
                    return "超SESOI(CI整体越界)"
                if abs(d) >= args.sesoi:
                    return "点估计超SESOI,CI跨界"
                return "未超SESOI"

            rows.append(dict(
                lib=l, bb=b, n_slice=len(y), n_rec=R, n_seed=len(have),
                seeds="|".join(map(str, have)), metric="macro_f1",
                d_slice_pp=round(res["slice"]["d"], 2),
                slice_lo=round(res["slice"]["lo"], 2), slice_hi=round(res["slice"]["hi"], 2),
                d_rec_pp=round(res["rec"]["d"], 2),
                rec_lo=round(res["rec"]["lo"], 2), rec_hi=round(res["rec"]["hi"], 2),
                seed_range_slice_pp=round(res["slice"]["sr"], 2),
                seed_range_rec_pp=round(res["rec"]["sr"], 2),
                flag_slice=flag(res["slice"]["d"], res["slice"]["lo"], res["slice"]["hi"]),
                flag_rec=flag(res["rec"]["d"], res["rec"]["lo"], res["rec"]["hi"]),
                status="OK"))

    cols = list(rows[0].keys()) if rows else ["lib"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    with open(args.missing_out, "w", encoding="utf-8-sig", newline="") as f:
        f.write("lib,arm,bb,seed,issue\n")
        for m in miss:
            f.write(",".join(map(str, m)) + "\n")
        for nm, why in stray:
            f.write(",,,%s,%s\n" % (nm, why))

    hd = "%-12s%-10s%7s%6s  %8s %18s  %8s %18s  %7s" % (
        "lib", "bb", "nSlice", "nRec", "dSlice", "95%CI", "dRec", "95%CI", "seedRng")
    print(hd); print("-" * len(hd))
    for r in rows:
        print("%-12s%-10s%7d%6d  %+8.2f [%+6.2f,%+6.2f]  %+8.2f [%+6.2f,%+6.2f]  %7.2f" % (
            r["lib"], r["bb"], r["n_slice"], r["n_rec"], r["d_slice_pp"],
            r["slice_lo"], r["slice_hi"], r["d_rec_pp"], r["rec_lo"], r["rec_hi"],
            r["seed_range_slice_pp"]))
    print("\nSESOI=%.1fpp  bootstrap=%d  cluster=recording" % (args.sesoi, args.n_boot))
    for r in rows:
        print("  %s/%s: slice=%s | rec=%s" % (r["lib"], r["bb"], r["flag_slice"], r["flag_rec"]))
    if miss:
        print("\n[缺失] %d 格：" % len(miss))
        for m in miss:
            print("  ", "/".join(map(str, m)))
    else:
        print("\n[缺失] 期望网格内无缺失")
    for nm, why in stray:
        print("[注意] %s: %s" % (nm, why))
    print("\n写盘: %s | %s" % (args.out, args.missing_out))


if __name__ == "__main__":
    main()
