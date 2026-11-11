"""List the contents of the remote 2 GB eval.zip without downloading it.

Reads the ZIP central directory through HTTP range requests, so we learn the
exact number of mel/cqt images, their naming scheme and per-class counts.
"""
import collections
import json
import os
import re
import ssl
import struct
import sys
import urllib.request

URL = "https://www.modelscope.cn/api/v1/datasets/Genius-Society/emo163/repo?Revision=master&FilePath=data%2Feval.zip"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get(url, start=None, end=None, method="GET", follow=True):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "Mozilla/5.0"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    resp = urllib.request.urlopen(req, context=CTX, timeout=180)
    return resp


def main():
    resp = get(URL, method="HEAD")
    size = int(resp.headers["Content-Length"])
    final = resp.geturl()
    print("content-length:", size, "resolved:", final[:120], file=sys.stderr)

    # central directory location from EOCD in the tail
    tail_len = min(size, 1 << 20)
    tail = get(URL, size - tail_len, size - 1).read()
    idx = tail.rfind(b"PK\x05\x06")
    if idx < 0:
        print("no EOCD found in tail", file=sys.stderr)
        return
    (_, _, _, _, n_entries, cd_size, cd_offset, _) = struct.unpack("<4s4H2LH", tail[idx:idx + 22])
    print(f"entries={n_entries} cd_size={cd_size} cd_offset={cd_offset}", file=sys.stderr)

    if cd_offset + cd_size > size:  # zip64 fallback: scan whole tail as CD
        cd = tail[tail.find(b"PK\x01\x02"):]
    else:
        cd = get(URL, cd_offset, cd_offset + cd_size - 1).read()

    entries = []
    pos = 0
    while True:
        pos = cd.find(b"PK\x01\x02", pos)
        if pos < 0:
            break
        (_, _, _, _, _, _, _, _crc, csize, usize, fnlen, extralen, cmtlen,
         _, _, _, _off) = struct.unpack("<4s6H3L5H2L", cd[pos:pos + 46])
        name = cd[pos + 46:pos + 46 + fnlen].decode("utf-8", "replace")
        entries.append({"name": name, "csize": csize, "usize": usize})
        pos += 46 + fnlen + extralen + cmtlen

    print("parsed entries:", len(entries), file=sys.stderr)
    dirs = collections.Counter("/".join(e["name"].split("/")[:2]) for e in entries)
    exts = collections.Counter(os.path.splitext(e["name"])[1].lower() for e in entries)
    labels = collections.Counter()
    songs = collections.defaultdict(set)
    for e in entries:
        base = os.path.basename(e["name"])
        m = re.match(r"^([A-Za-z]+)_(\d+)_", base)
        if m:
            labels[m.group(1)] += 1
            songs[m.group(1)].add(m.group(2))

    per_dir_label = collections.defaultdict(collections.Counter)
    for e in entries:
        parts = e["name"].split("/")
        base = parts[-1]
        m = re.match(r"^([A-Za-z]+)_(\d+)_", base)
        if m and len(parts) >= 2:
            per_dir_label[parts[-2]][m.group(1)] += 1

    rep = {
        "zip_size_bytes": size,
        "num_entries": len(entries),
        "top_dirs": dict(dirs.most_common(10)),
        "extensions": dict(exts.most_common(10)),
        "files_per_label": dict(labels.most_common()),
        "unique_songs_per_label": {k: len(v) for k, v in songs.items()},
        "unique_songs_total": len(set().union(*songs.values())) if songs else 0,
        "per_spectrogram_label_counts": {k: dict(v) for k, v in per_dir_label.items()},
        "sample_names": [e["name"] for e in entries[:8]],
        "mean_jpg_bytes": round(
            sum(e["usize"] for e in entries if e["name"].lower().endswith(".jpg")) / max(1, len(entries)), 1
        ),
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "eval_zip.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
