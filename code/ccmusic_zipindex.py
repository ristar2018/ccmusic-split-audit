"""Read a remote zip directory (without downloading the whole package) and emit the entry list / slice counts / source-ID statistics.

Purpose: P1 needs to know how many slices each dataset's eval.zip really holds and how many slices each source recording is cut into.
The music_genre eval.zip is 4 GB and the CTIS one is 3 GB -- downloading them in full just to count entries would be wasteful.
The zip central directory sits at the end of the file, so a few MB of HTTP Range requests are enough.

Usage:
    python code/ccmusic_zipindex.py --dataset erhu_playing_tech --file data/eval.zip
    python code/ccmusic_zipindex.py --project p1          # fetch the 7 eval packages + 2 control groups one by one
    python code/ccmusic_zipindex.py --dataset music_genre --file data/eval.zip --full   # persist the full name list

Output: data/index/<dataset>__<file>.json
      {entries: N, features: {...}, classes: {...}, sources: {...}, names (optional)}
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
    """Read-only, seekable remote file; caches by block to cut the number of requests."""

    def __init__(self, url: str, block: int = 1 << 20):
        super().__init__()
        self.url, self.block, self.pos = url, block, 0
        self._cache: dict[int, bytes] = {}
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()          # follow the 302 to the CDN

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

    def readinto(self, b) -> int:  # zipfile may come through here
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def url_for(dataset: str, path: str) -> str:
    return f"{API}/{dataset}/repo?Revision=master&FilePath={urllib.parse.quote(path)}"


def source_id(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    if "__" in stem:                       # erhu: detache__detache_01_chroma
        return stem.split("__", 1)[1]
    m = re.match(r"^([0-9a-f]{32})", stem)  # music_genre and friends: md5 prefix
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)_", stem)          # chest_falsetto: 0001_m_chest_0
    if m:
        return m.group(1)
    return re.sub(r"_(\d+)$", "", stem)     # GZ_IsoTech: record4_xiahua11


def feature_of(name: str) -> str:
    """The feature name may appear as a file-name suffix or in the directory name (pianos/CTIS use directories)."""
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
        anchor = "other"                      # no feature marker in the file name (e.g. pianos)
    anchor_names = [n for n in names if feature_of(n) == anchor]
    # source key = class directory + source ID: different classes may reuse the same numeric id (e.g. 009 in pianos)
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
    ap = argparse.ArgumentParser(description="read a CCMusic zip directory remotely")
    ap.add_argument("--dataset")
    ap.add_argument("--file")
    ap.add_argument("--project", choices=["p1"])
    ap.add_argument("--full", action="store_true", help="save the full name list (needed by P1 grouping)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = ([(k, v) for k, v in PROJECT_P1.items()] if args.project
            else [(args.dataset, args.file)])
    for dataset, path in jobs:
        try:
            res = index(dataset, path, full=args.full)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {dataset}/{path}: {str(exc)[:120]}")
            continue
        out = os.path.join(OUT_DIR, f"{dataset}__{path.replace('/', '_')}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print(f"{dataset:18s} {path:14s} entries {res['entries_total']:6d}  anchor({res['anchor_feature']}) "
              f"slices {res['slices']:6d}  sources {res['sources']:5d}  K={res['slices_per_source']:5.2f}  "
              f"classes {res['class_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
