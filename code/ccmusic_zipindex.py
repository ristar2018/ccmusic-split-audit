"""远程读取 zip 目录（不下载整个包），产出条目清单 / 切片计数 / 源 ID 统计。

用途：P1 需要知道每个数据集 eval.zip 里到底有多少切片、每条源录音被切成几片。
music_genre 的 eval.zip 有 4 GB，CTIS 的 3 GB——为了数条目而全量下载是浪费。
zip 的中央目录（central directory）在文件末尾，用 HTTP Range 只取几 MB 即可。

用法：
    python code/ccmusic_zipindex.py --dataset erhu_playing_tech --file data/eval.zip
    python code/ccmusic_zipindex.py --project p1          # 逐个取 7 个 eval 包 + 2 个对照组
    python code/ccmusic_zipindex.py --dataset music_genre --file data/eval.zip --full   # 落盘完整名单

输出：data/index/<dataset>__<file>.json
      {entries: N, features: {...}, classes: {...}, sources: {...}, names(可选)}
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "index")
API = "https://www.modelscope.cn/api/v1/datasets/ccmusic-database"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ccmusic-zipindex/1.0"
CTX = ssl.create_default_context()
FEATURES = ("mel", "cqt", "chroma")


class HttpFile(io.RawIOBase):
    """只读、可 seek 的远程文件；按块缓存，减少请求数。"""

    def __init__(self, url: str, block: int = 1 << 20):
        super().__init__()
        self.url, self.block, self.pos = url, block, 0
        self._cache: dict[int, bytes] = {}
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()          # 跟随 302 到 CDN

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, min(self.size, base + offset))
        return self.pos

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            start = idx * self.block
            end = min(self.size, start + self.block) - 1
            req = urllib.request.Request(
                self.url, headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                self._cache[idx] = r.read()
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        out, left = [], n
        while left > 0:
            idx, off = divmod(self.pos, self.block)
            chunk = self._block(idx)[off:off + left]
            if not chunk:
                break
            out.append(chunk)
            self.pos += len(chunk)
            left -= len(chunk)
        return b"".join(out)

    def readinto(self, b) -> int:  # zipfile 可能走这里
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def url_for(dataset: str, path: str) -> str:
    return f"{API}/{dataset}/repo?Revision=master&FilePath={urllib.parse.quote(path)}"


def source_id(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if "__" in stem:                       # erhu: detache__detache_01_chroma
        return stem.split("__", 1)[1]
    m = re.match(r"^([0-9a-f]{32})", stem)  # music_genre 等：md5 前缀
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)_", stem)          # chest_falsetto: 0001_m_chest_0
    if m:
        return m.group(1)
    return re.sub(r"_(\d+)$", "", stem)     # GZ_IsoTech: record4_xiahua11


def feature_of(name: str) -> str:
    """特征名可能出现在文件名后缀，也可能在目录名里（pianos/CTIS 用目录）。"""
    stem = os.path.splitext(os.path.basename(name))[0]
    for f in FEATURES:
        if stem.endswith(f):
            return f
    for part in name.replace("\\", "/").split("/")[:-1]:
        if part.lower() in FEATURES:
            return part.lower()
    return "other"


def class_of(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    d = parts[-2] if len(parts) >= 2 else ""
    if d and d.lower() not in (*FEATURES, "eval", "audio", "train", "test", "validation", "val"):
        return d
    if "__" in stem:
        return stem.split("__", 1)[0]
    return "unknown"


def index(dataset: str, path: str, full: bool = False) -> dict:
    zf = zipfile.ZipFile(HttpFile(url_for(dataset, path)))
    names = [n for n in zf.namelist() if not n.endswith("/")]
    zf.close()
    feats = Counter(feature_of(n) for n in names)
    if feats.get("mel"):
        anchor = "mel"
    elif any(feats.get(f) for f in FEATURES):
        anchor = max(FEATURES, key=lambda f: feats.get(f, 0))
    else:
        anchor = "other"                      # 文件名里没有特征标记（如 pianos）
    anchor_names = [n for n in names if feature_of(n) == anchor]
    # 源键 = 类别目录 + 源 ID：不同类别下可能出现相同数字编号（如 pianos 的 009）
    keys = [f"{class_of(n)}::{source_id(n)}" for n in anchor_names]
    src = Counter(keys)
    cls = Counter(class_of(n) for n in anchor_names)
    res = {
        "dataset": dataset,
        "file": path,
        "entries_total": len(names),
        "features": dict(feats),
        "anchor_feature": anchor,
        "slices": len(anchor_names),
        "sources": len(src),
        "slices_per_source": round(len(anchor_names) / max(len(src), 1), 2),
        "source_slice_hist": dict(Counter(src.values())),
        "classes": dict(cls.most_common(30)),
        "class_count": len(cls),
    }
    if full:
        res["names"] = anchor_names
    return res


PROJECT_P1 = {
    "pianos": "data/eval.zip",
    "music_genre": "data/eval.zip",
    "chest_falsetto": "data/eval.zip",
    "CNPM": "data/eval.zip",
    "bel_canto": "data/eval.zip",
    "erhu_playing_tech": "data/eval.zip",
    "CTIS": "data/eval.zip",
    "GZ_IsoTech": "data/eval.zip",
    "Guzheng_Tech99": "data/mel.zip",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="远程读取 CCMusic zip 目录")
    ap.add_argument("--dataset")
    ap.add_argument("--file")
    ap.add_argument("--project", choices=["p1"])
    ap.add_argument("--full", action="store_true", help="保存完整名单（P1 分组需要）")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = ([(k, v) for k, v in PROJECT_P1.items()] if args.project
            else [(args.dataset, args.file)])
    for dataset, path in jobs:
        try:
            res = index(dataset, path, full=args.full)
        except Exception as exc:  # noqa: BLE001
            print(f"[失败] {dataset}/{path}: {str(exc)[:120]}")
            continue
        out = os.path.join(OUT_DIR, f"{dataset}__{path.replace('/', '_')}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print(f"{dataset:18s} {path:14s} 条目 {res['entries_total']:6d}  锚定({res['anchor_feature']}) "
              f"切片 {res['slices']:6d}  源 {res['sources']:5d}  K={res['slices_per_source']:5.2f}  "
              f"类 {res['class_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
