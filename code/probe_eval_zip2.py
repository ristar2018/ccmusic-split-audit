"""Second-pass probe of the remote eval.zip: per-song segment structure + sample images.

Parses the ZIP central directory (range requests only), reconstructs the
per-song segment layout from file names, then downloads a few JPEGs to check
their pixel dimensions.
"""
import collections
import json
import os
import re
import ssl
import struct
import urllib.request
import zlib

URL = "https://www.modelscope.cn/api/v1/datasets/Genius-Society/emo163/repo?Revision=master&FilePath=data%2Feval.zip"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME_RE = re.compile(r"^([A-Za-z]+)_(\d+)_([\d.]+)_(\d+)\.jpg$")


def get(url, start=None, end=None, method="GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "Mozilla/5.0"})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    return urllib.request.urlopen(req, context=CTX, timeout=300)


def parse_cd():
    size = int(get(URL, method="HEAD").headers["Content-Length"])
    tail_len = min(size, 1 << 20)
    tail = get(URL, size - tail_len, size - 1).read()
    idx = tail.rfind(b"PK\x05\x06")
    n_entries, cd_size, cd_offset = struct.unpack("<H2L", tail[idx + 10:idx + 20])
    cd = get(URL, cd_offset, cd_offset + cd_size - 1).read()
    entries, pos = [], 0
    while True:
        pos = cd.find(b"PK\x01\x02", pos)
        if pos < 0:
            break
        (_, _, _, _, method, _, _, _, csize, usize, fnlen, extralen, cmtlen,
         _, _, _, off) = struct.unpack("<4s6H3L5H2L", cd[pos:pos + 46])
        name = cd[pos + 46:pos + 46 + fnlen].decode("utf-8", "replace")
        entries.append({"name": name, "method": method, "csize": csize,
                        "usize": usize, "off": off})
        pos += 46 + fnlen + extralen + cmtlen
    return size, n_entries, entries


def fetch_entry(e):
    """Download one entry's compressed bytes and inflate them."""
    hdr = get(URL, e["off"], e["off"] + 30 + 1024).read()
    fnlen, extralen = struct.unpack("<2H", hdr[26:30])
    start = e["off"] + 30 + fnlen + extralen
    raw = get(URL, start, start + e["csize"] - 1).read()
    return zlib.decompress(raw, -15) if e["method"] == 8 else raw


def main():
    size, n_entries, entries = parse_cd()
    print(f"zip={size} entries={n_entries} parsed={len(entries)}")

    by_spect = collections.defaultdict(list)
    bad = []
    for e in entries:
        parts = e["name"].split("/")
        m = NAME_RE.match(parts[-1])
        if not m or len(parts) < 3:
            bad.append(e["name"])
            continue
        label, idx, dur, frame = m.group(1), int(m.group(2)), float(m.group(3)), int(m.group(4))
        by_spect[parts[-2]].append((label, idx, dur, frame, e))

    mel = [(l, i, d, f, e) for (l, i, d, f, e) in by_spect["mel"]]
    cqt = by_spect["cqt"]
    names_mel = {e["name"].replace("/mel/", "/XX/") for (_, _, _, _, e) in mel}
    names_cqt = {e["name"].replace("/cqt/", "/XX/") for (_, _, _, _, e) in cqt}
    print("mel/cqt name sets identical:", names_mel == names_cqt,
          "mel_only:", len(names_mel - names_cqt), "cqt_only:", len(names_cqt - names_mel))
    for s in list(names_mel - names_cqt)[:3] + list(names_cqt - names_mel)[:3]:
        print("  diff:", s)

    # reconstruct songs: (label, duration) pairs, index is per-worker so unreliable
    songs = collections.defaultdict(lambda: {"segs": set(), "dur": None})
    for label, idx, dur, frame, _ in mel:
        key = (label, dur)
        songs[key]["dur"] = dur
        songs[key]["segs"].add(frame)
    per_label = collections.Counter(k[0] for k in songs)
    dur_by_label = collections.defaultdict(float)
    segs_by_label = collections.Counter()
    for (label, dur), info in songs.items():
        dur_by_label[label] += dur
        segs_by_label[label] += len(info["segs"])

    seg_counts = sorted(len(v["segs"]) for v in songs.values())
    rep = {
        "zip_size_bytes": size,
        "num_entries": n_entries,
        "parsed_entries": len(entries),
        "unparsed_names": bad[:5],
        "compression_methods": dict(collections.Counter(e["method"] for e in entries)),
        "mel_images": len(mel),
        "cqt_images": len(cqt),
        "estimated_songs": len(songs),
        "songs_per_label": dict(per_label.most_common()),
        "segments_per_label": dict(segs_by_label.most_common()),
        "audio_minutes_per_label": {k: round(v / 60, 1) for k, v in dur_by_label.items()},
        "total_audio_minutes": round(sum(dur_by_label.values()) / 60, 1),
        "segment_count_stats": {
            "min": seg_counts[0], "median": seg_counts[len(seg_counts) // 2],
            "max": seg_counts[-1], "mean": round(sum(seg_counts) / len(seg_counts), 2),
        },
        "duration_stats_sec": {
            "min": round(min(k[1] for k in songs), 2),
            "median": round(sorted(k[1] for k in songs)[len(songs) // 2], 2),
            "max": round(max(k[1] for k in songs), 2),
        },
        "avg_jpeg_bytes": round(sum(e["usize"] for e in entries) / len(entries), 1),
    }

    # sample images -> pixel size
    from PIL import Image
    import io
    samp = os.path.join(ROOT, "results", "samples")
    os.makedirs(samp, exist_ok=True)
    picks = [mel[0][-1], mel[len(mel) // 2][-1], mel[-1][-1],
             cqt[0][-1], cqt[len(cqt) // 2][-1], cqt[-1][-1]]
    rep["sample_images"] = []
    for e in picks:
        data = fetch_entry(e)
        img = Image.open(io.BytesIO(data))
        fn = os.path.basename(e["name"])
        with open(os.path.join(samp, fn), "wb") as fh:
            fh.write(data)
        rep["sample_images"].append({"name": e["name"], "bytes": len(data),
                                     "size_px": list(img.size), "mode": img.mode})
        print(rep["sample_images"][-1])

    with open(os.path.join(ROOT, "results", "eval_zip2.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "sample_images"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
