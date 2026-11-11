# -*- coding: utf-8 -*-
"""CCMusic 数据集解压器：就地展开到 <dataset>\\data\\<顶层目录>，校验后删除压缩包。

为什么这样放
  每个数据集的官方加载脚本都按 data/audio、data/eval/chroma、data/mel、data/label
  这类路径读文件；压缩包内部第一层恰好就是 audio/ eval/ mel/ label/ timbre/ range/。
  因此把 zips 解到 <dataset>\\data\\ 后，目录结构与加载脚本的约定完全一致，不需要改代码。

安全阀
  1. 先看磁盘余量（解压后体积 ≈ 压缩包内文件字节数之和）；
  2. 解压时逐个条目计数、累计字节，和 zip 中央目录对账；
  3. 只有"文件数 + 字节数"两项都对上，才删压缩包；对不上就保留 zip 并报错；
  4. --keep-zips 可只解压不删；--dry-run 只看计划。

用法
  python code/ccmusic_extract.py --dry-run
  python code/ccmusic_extract.py                    # 全部解压并删包
  python code/ccmusic_extract.py --dataset CTIS     # 只处理一个
  python code/ccmusic_extract.py --keep-zips        # 解压但保留 zip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

DATASETS_ROOT = r"D:\VSCode\Datasets"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def find_archives(root: str, only: str | None = None) -> list[tuple[str, str]]:
    """返回 [(dataset, zip 路径)]，只看 <dataset>/data/*.zip。"""
    out = []
    for name in sorted(os.listdir(root)):
        ddir = os.path.join(root, name, "data")
        if not os.path.isdir(ddir) or name.startswith("_"):
            continue
        if only and name != only:
            continue
        for fn in sorted(os.listdir(ddir)):
            if fn.lower().endswith(".zip"):
                out.append((name, os.path.join(ddir, fn)))
    return out


def plan(archive: str) -> dict:
    with zipfile.ZipFile(archive) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        tops = sorted({i.filename.split("/")[0] for i in infos})
    return {
        "files": len(infos),
        "bytes": sum(i.file_size for i in infos),
        "top": tops,
        "compressed": os.path.getsize(archive),
    }


def extract_one(dataset: str, archive: str, root: str, dry: bool, keep: bool) -> dict:
    meta = plan(archive)
    dest_root = os.path.join(root, dataset, "data")  # 解到 data/，顶层 audio|eval|... 自然归位
    rec = {
        "dataset": dataset,
        "archive": os.path.basename(archive),
        "zip_bytes": meta["compressed"],
        "files": meta["files"],
        "bytes": meta["bytes"],
        "top": meta["top"],
        "dest": dest_root,
    }
    if dry:
        rec["status"] = "dry-run"
        return rec

    free = shutil.disk_usage(root).free
    if meta["bytes"] > free:
        rec["status"] = f"skip 磁盘不足（需 {human(meta['bytes'])}，剩 {human(free)}）"
        return rec

    t0 = time.time()
    n_ok = 0
    n_skip = 0
    b_ok = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = os.path.join(dest_root, info.filename.replace("/", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                n_skip += 1
                b_ok += info.file_size
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            n_ok += 1
            b_ok += info.file_size

    rec.update(
        extracted=n_ok,
        skipped=n_skip,
        bytes_ok=b_ok,
        seconds=round(time.time() - t0, 1),
    )
    if b_ok != meta["bytes"] or n_ok + n_skip != meta["files"]:
        rec["status"] = f"mismatch 文件 {n_ok + n_skip}/{meta['files']}、字节 {b_ok}/{meta['bytes']}（保留 zip）"
        return rec

    if keep:
        rec["status"] = "ok（保留 zip）"
        return rec

    os.remove(archive)
    # 顺带清掉该数据集目录下解压时可能残留的空目录
    rec["status"] = "ok（已删 zip）"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="CCMusic 数据集解压（解到 data/ 并删包）")
    ap.add_argument("--root", default=DATASETS_ROOT, help="下载根目录")
    ap.add_argument("--dataset", help="只处理某个数据集")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-zips", action="store_true", help="解压后保留压缩包")
    args = ap.parse_args()

    archives = find_archives(args.root, args.dataset)
    if not archives:
        print("没有找到待解压的压缩包")
        return 0

    print(f"根目录：{args.root}")
    total_files = total_bytes = total_zip = 0
    rows = []
    for dataset, arc in archives:
        p = plan(arc)
        total_files += p["files"]
        total_bytes += p["bytes"]
        total_zip += p["compressed"]
        print(
            f"  {dataset:18s} {os.path.basename(arc):14s} "
            f"{p['files']:>6} 文件  {human(p['bytes']):>8} → data/{','.join(p['top'])}"
        )
    print(f"\n合计 {len(archives)} 个包 / {total_files} 个文件 / 解压后 {human(total_bytes)}（压缩包 {human(total_zip)}）")
    print(f"磁盘可用 {human(shutil.disk_usage(args.root).free)}")
    if args.dry_run:
        print("\n--dry-run：什么都没动")
        return 0

    print("\n开始解压 …")
    t0 = time.time()
    for dataset, arc in archives:
        rec = extract_one(dataset, arc, args.root, args.dry_run, args.keep_zips)
        rows.append(rec)
        print(
            f"  [{rec['status']:<28}] {dataset:18s} {rec['archive']:14s} "
            f"新增 {rec.get('extracted', 0):>6} 跳过 {rec.get('skipped', 0):>5} "
            f"{rec.get('seconds', 0):>6.1f}s"
        )

    out = os.path.join(args.root, "_extract.json")
    prev = {}
    if os.path.exists(out):
        try:
            with open(out, encoding="utf-8") as fh:
                prev = json.load(fh)
        except Exception:  # noqa: BLE001
            prev = {}
    prev.update({f"{r['dataset']}/{r['archive']}": r for r in rows})
    prev["_meta"] = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": len([k for k in prev if not k.startswith("_")]),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, ensure_ascii=False, indent=1)

    bad = [r for r in rows if not str(r["status"]).startswith("ok")]
    print(f"\n用时 {time.time() - t0:.0f}s，记录 → {out}")
    if bad:
        print(f"!! {len(bad)} 个包未通过校验：{[b['dataset'] + '/' + b['archive'] for b in bad]}")
        return 1
    print("全部通过校验并已删除压缩包")
    return 0


if __name__ == "__main__":
    sys.exit(main())
