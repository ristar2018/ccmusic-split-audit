#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征缓存：把 manifest 里的 mel 图统一成 (N,224,224,3) uint8 memmap。

设计依据（口径不许在代码里改）：
- `worklog/exp-plan/KICKOFF.md` §3.4：五库 mel、uint8 memmap、224×224；尺寸分组
  496×369 / 476×349 / 417×128 / 5360×128 / 7220×128；插值写死并在图注声明；
  `num_workers=6, persistent_workers=True`；验收＝抽样 200 张与原图目视一致 + CACHE.json。
- `report/聚焦版实验设计方案v2.8.md:575`：主样本每 run 用 0.60S ≈ 66,249 张
  （224×224×3 uint8 ≈ 10.0 GB）；全量 110,415 张 ≈ 16.6 GB。
- 同文件 `:587`（风险表）：特征图尺寸不统一（496×369 为主、pianos 128 高长条最宽 28,235），
  「统一 resize 到 224，方法里写明插值方式与长宽比处理；不做尺寸级联对比」。

本脚本的插值选择（写死，改就改这里和图注）：
- `PIL.Image.BILINEAR`。Pillow 的 resize 是 scale-aware 的（按缩放比放大 filter support），
  缩小到 224 时自带抗锯齿；spectrogram 用 LANCZOS 会在高频边缘出过冲环，BILINEAR 不产生振铃。
- 长宽比：**不保**，直接拉伸到 224×224。理由：五库源图长宽比从 1.34（496×369）到 42（5360×128）
  跨两个数量级，任何"保比例 + padding"都会给小图加大量黑边，让 padding 比例本身变成库间混淆变量。
  代价写进方法：长条图（pianos）的横轴被压到 1/24，时间分辨率在缓存里丢失。
- 色彩空间：统一 `convert("RGB")`（源图已全是 RGB，此步是防御性断言）。

用法：
  python code/ccmusic_cache.py --libs CTIS,music_genre --workers 8 --verify 200
  python code/ccmusic_cache.py --libs CTIS --no-verify
  python code/ccmusic_cache.py --libs CTIS --verify 200 --force     # 重写已有缓存

产物（默认 `data/cache/mel224/`）：
  <lib>.u8             uint8 memmap，形状 (N,224,224,3)，行序 = manifest 行序
  <lib>_index.csv      row,sample_id,label,recording_uid,src_w,src_h,resized
  CACHE.json           每库条数/形状/dtype/源尺寸直方图/插值/长宽比策略/耗时/校验结果
  verify_montage.png   --verify 时生成：12 组「原图（等比缩到 112 高）｜缓存回读」
"""

import argparse
import collections
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"D:\VSCode\Datasets"
MANIFEST = os.path.join(PROJ, "data", "manifest")
DEFAULT_OUT = os.path.join(PROJ, "data", "cache", "mel224")

SIZE = 224
CHUNK = 512
RESAMPLE = Image.BILINEAR
RESAMPLE_NAME = "PIL BILINEAR (scale-aware, antialiased downscale)"
ASPECT_POLICY = "stretch to square (aspect NOT preserved)"


def load_manifest(lib):
    path = os.path.join(MANIFEST, lib + ".csv")
    if not os.path.exists(path):
        raise SystemExit("manifest 不存在: " + path)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_one(rel, size=SIZE):
    """单张：open → RGB → resize → uint8 HWC。返回 (arr, (w, h))。"""
    p = os.path.join(DATA, rel.replace("/", os.sep))
    with Image.open(p) as im:
        wh = im.size
        if im.mode != "RGB":
            im = im.convert("RGB")
        if wh != (size, size):
            im = im.resize((size, size), RESAMPLE)
        arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (size, size, 3):
        raise ValueError("unexpected shape %s for %s" % (arr.shape, rel))
    return arr, wh


def build_lib(lib, rows, outdir, workers, force):
    n = len(rows)
    path = os.path.join(outdir, lib + ".u8")
    idx = os.path.join(outdir, lib + "_index.csv")
    expect_bytes = n * SIZE * SIZE * 3
    if os.path.exists(path) and not force:
        got = os.path.getsize(path)
        if got == expect_bytes and os.path.exists(idx):
            print("[skip] %s 已有完整缓存 (%d 行, %.2f GB)" % (lib, n, got / 2 ** 30))
            return None
        raise SystemExit("%s 的缓存不完整（%d/%d 字节），加 --force 重写" % (lib, got, expect_bytes))

    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(n, SIZE, SIZE, 3))
    sizes = collections.Counter()
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(idx, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "sample_id", "label", "recording_uid", "src_w", "src_h", "resized"])
        for start in range(0, n, CHUNK):
            block = rows[start:start + CHUNK]
            res = list(ex.map(lambda r: read_one(r["relative_path"]), block))
            for k, ((arr, wh), r) in enumerate(zip(res, block)):
                i = start + k
                mm[i] = arr
                sizes[wh] += 1
                w.writerow([i, r["sample_id"], r["label"], r["recording_uid"],
                            wh[0], wh[1], int(wh != (SIZE, SIZE))])
            done += len(block)
            if done % (CHUNK * 8) == 0 or done == n:
                el = time.time() - t0
                print("  %s %d/%d  %.1f img/s  %.2f GB  elapsed %.0fs"
                      % (lib, done, n, done / max(el, 1e-9), done * SIZE * SIZE * 3 / 2 ** 30, el))
    mm.flush()
    del mm
    got = os.path.getsize(path)
    if got != expect_bytes:
        raise SystemExit("%s 写入字节数不符：%d != %d" % (lib, got, expect_bytes))
    return {"rows": n, "bytes": got, "gb": round(got / 2 ** 30, 3),
            "src_sizes": {"%dx%d" % k: v for k, v in sizes.most_common()},
            "n_resized": sum(v for k, v in sizes.items() if k != (SIZE, SIZE)),
            "elapsed_s": round(time.time() - t0, 1),
            "img_per_s": round(n / max(time.time() - t0, 1e-9), 1)}


def verify_lib(lib, rows, outdir, k=200, seed=20261105):
    """抽样 k 行：缓存回读 vs 现场重算，逐字节比对；返回差异统计。"""
    path = os.path.join(outdir, lib + ".u8")
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(len(rows), SIZE, SIZE, 3))
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(rows)), min(k, len(rows))))
    diffs = []
    picks = []
    for i in idxs:
        ref, _ = read_one(rows[i]["relative_path"])
        got = np.asarray(mm[i])
        d = int(np.abs(got.astype(np.int16) - ref.astype(np.int16)).max())
        diffs.append(d)
        if len(picks) < 12:
            picks.append((i, ref, got))
    del mm
    bad = [d for d in diffs if d != 0]
    return {"n": len(idxs), "max_abs_diff": max(diffs), "n_nonzero": len(bad),
            "sample_rows": idxs[:5], "picks": picks}


def montage(picks, path):
    """12 组：左＝原图等比缩到 112 高，右＝缓存回读缩到 112 高。"""
    h = 112
    tiles = []
    for i, ref, got in picks:
        a = Image.fromarray(ref).resize((int(round(SIZE * h / SIZE)), h), Image.BILINEAR)
        b = Image.fromarray(np.asarray(got)).resize((a.size[0], h), Image.BILINEAR)
        sep = Image.new("RGB", (3, h), (255, 0, 0))
        row = Image.new("RGB", (a.size[0] * 2 + 3, h), (255, 255, 255))
        row.paste(a, (0, 0))
        row.paste(sep, (a.size[0], 0))
        row.paste(b, (a.size[0] + 3, 0))
        tiles.append(row)
    cols = 3
    rows_n = (len(tiles) + cols - 1) // cols
    tw = max(t.size[0] for t in tiles)
    sheet = Image.new("RGB", (tw * cols + 2 * (cols - 1), h * rows_n + 2 * (rows_n - 1)), (255, 255, 255))
    for n_, t in enumerate(tiles):
        r, c = divmod(n_, cols)
        sheet.paste(t, (c * (tw + 2), r * (h + 2)))
    sheet.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", default="CTIS,music_genre")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    libs = [x.strip() for x in a.libs.split(",") if x.strip()]
    os.makedirs(a.out, exist_ok=True)

    cache = {"generated_by": "code/ccmusic_cache.py",
             "design_ref": ["worklog/exp-plan/KICKOFF.md §3.4",
                            "report/聚焦版实验设计方案v2.8.md:575", "同文件 :587"],
             "size": SIZE, "dtype": "uint8", "channels": 3,
             "resample": RESAMPLE_NAME, "aspect_policy": ASPECT_POLICY,
             "row_order": "manifest 行序（row 列即 manifest 行号）",
             "data_root": DATA, "out": a.out, "libraries": {}}
    # 增量补库时不能丢掉已缓存库的条目：CACHE.json 是整份汇总（total_rows / total_gb 从它累加）
    prev_path = os.path.join(a.out, "CACHE.json")
    if os.path.exists(prev_path):
        try:
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
            cache["libraries"] = dict(prev.get("libraries", {}))
            print("合并既有 CACHE.json：%d 库（%d 行）"
                  % (len(cache["libraries"]), sum(v.get("rows", 0) for v in cache["libraries"].values())))
        except Exception as e:      # noqa: BLE001  旧格式读不出就重建，不阻塞
            print("!! 旧 CACHE.json 读不了（%s），按全新构建" % e)

    for lib in libs:
        rows = load_manifest(lib)
        print("[%s] manifest %d 行" % (lib, len(rows)))
        info = build_lib(lib, rows, a.out, a.workers, a.force)
        if info is None:
            # 跳过重建时也要把体量与源图尺寸补齐，否则汇总 total_gb 只算到新建的那几个库
            p = os.path.join(a.out, lib + ".u8")
            nb = os.path.getsize(p)
            info = {"rows": len(rows), "bytes": nb, "gb": round(nb / 2 ** 30, 3),
                    "cached": True, "skipped_rebuild": True}
            with open(os.path.join(a.out, "CACHE.json"), encoding="utf-8") as f:
                prev = json.load(f)
            old = prev["libraries"].get(lib, {})
            for k in ("src_sizes", "n_resized"):
                if k in old:
                    info[k] = old[k]
            if "src_sizes" not in info:
                # 旧 CACHE.json 没记源图尺寸时，从 index.csv 的 src_w/src_h 重建
                sizes = collections.Counter()
                nres = 0
                with open(os.path.join(a.out, lib + "_index.csv"), encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        sizes["%sx%s" % (r["src_w"], r["src_h"])] += 1
                        if str(r.get("resized")).lower() in ("true", "1"):
                            nres += 1
                info["src_sizes"] = dict(sizes)
                info["n_resized"] = nres
        cache["libraries"][lib] = info
        if not a.no_verify and a.verify > 0:
            v = verify_lib(lib, rows, a.out, k=a.verify)
            mp = montage(v.pop("picks"), os.path.join(a.out, "verify_montage_%s.png" % lib))
            v["montage"] = os.path.basename(mp)
            cache["libraries"][lib]["verify"] = v
            print("[%s] 抽样 %d：max_abs_diff=%d 不符 %d 张 → %s"
                  % (lib, v["n"], v["max_abs_diff"], v["n_nonzero"], v["montage"]))
            if v["n_nonzero"]:
                raise SystemExit("缓存与原图不一致，停止")

    cache["total_rows"] = sum(v.get("rows", 0) for v in cache["libraries"].values())
    cache["total_gb"] = round(sum(v.get("gb", 0) for v in cache["libraries"].values()), 3)
    cache["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(a.out, "CACHE.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print("→ %s" % os.path.join(a.out, "CACHE.json"))
    print("合计 %d 行 / %.3f GB" % (cache["total_rows"], cache["total_gb"]))


if __name__ == "__main__":
    main()
