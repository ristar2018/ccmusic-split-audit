# -*- coding: utf-8 -*-
"""Recheck audio sample rate and duration across the 13 libraries: pure Python header reads, no decoding, no subprocesses.

- wav/riff: parse the fmt + data chunks by hand (PCM integer, IEEE float 0x0003, WAVE_FORMAT_EXTENSIBLE, mp3-in-wav)
- flac: parse STREAMINFO (20-bit sample rate + 36-bit total sample count)
- mp3: skip ID3v2 and read the sample rate from the first frame header; use the frame count when Xing/Info is present, otherwise estimate from the bitrate
Writes data/sample_rates.json and reconciles it library by library against data/datasets_inventory.json.
"""
import io, json, os, struct
from collections import Counter, defaultdict

ROOT = r"D:\VSCode\Datasets"
OUT = r"D:\VSCode\科研任务\音乐情感研究\data\sample_rates.json"
INV = r"D:\VSCode\科研任务\音乐情感研究\data\datasets_inventory.json"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

MP3_SR = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}
MP3_BR_V1_L3 = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
MP3_BR_V2_L3 = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]


def riff_info(path):
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            raise ValueError("not RIFF/WAVE")
        fmt = data_size = None
        while True:
            ch = f.read(8)
            if len(ch) < 8:
                break
            cid, csz = ch[:4], struct.unpack("<I", ch[4:])[0]
            if cid == b"fmt ":
                body = f.read(csz)
                tag, nch, sr, byterate, align, bits = struct.unpack("<HHIIHH", body[:16])
                fmt = dict(tag=tag, ch=nch, sr=sr, byterate=byterate, align=align, bits=bits)
                if csz % 2:
                    f.read(1)
            elif cid == b"data":
                data_size = csz
                if fmt is not None and fmt["sr"]:
                    break
                f.seek(csz + (csz % 2), 1)
            else:
                f.seek(csz + (csz % 2), 1)
        if not fmt or not fmt["sr"]:
            raise ValueError("no fmt chunk")
        if data_size in (None, 0xFFFFFFFF):
            data_size = max(os.path.getsize(path) - 44, 0)
        if fmt["byterate"]:
            dur = data_size / float(fmt["byterate"])
        else:
            dur = data_size / float(fmt["sr"] * max(fmt["align"], 1))
        return fmt["sr"], dur, fmt["ch"], "riff:tag0x%04x" % fmt["tag"]


def flac_info(path):
    with open(path, "rb") as f:
        if f.read(4) != b"fLaC":
            raise ValueError("not flac")
        while True:
            bh = f.read(4)
            if len(bh) < 4:
                raise ValueError("no streaminfo")
            last = bh[0] & 0x80
            btype = bh[0] & 0x7F
            blen = int.from_bytes(bh[1:4], "big")
            body = f.read(blen)
            if btype == 0:
                b = body
                sr = (b[10] << 12) | (b[11] << 4) | (b[12] >> 4)
                total = ((b[13] & 0x0F) << 32) | int.from_bytes(b[14:18], "big")
                ch = ((b[12] >> 1) & 0x07) + 1  # STREAMINFO stores channels-1
                return sr, total / float(sr), ch, "flac"
            if last:
                raise ValueError("no streaminfo")


def mp3_info(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h = f.read(10)
        start = 0
        if h[:3] == b"ID3" and len(h) >= 10:
            sz = ((h[6] & 0x7F) << 21) | ((h[7] & 0x7F) << 14) | ((h[8] & 0x7F) << 7) | (h[9] & 0x7F)
            start = 10 + sz + (10 if (h[5] & 0x10) else 0)
        f.seek(start)
        buf = f.read(4096)
    i = 0
    while i < len(buf) - 4:
        b1, b2 = buf[i], buf[i + 1]
        if b1 == 0xFF and (b2 & 0xE0) == 0xE0:
            ver = (b2 >> 3) & 0x03
            layer = (b2 >> 1) & 0x03
            bri = (buf[i + 2] >> 4) & 0x0F
            sri = (buf[i + 2] >> 2) & 0x03
            if ver != 1 and layer == 1 and bri not in (0, 15) and sri != 3:
                sr = MP3_SR[ver][sri]
                br = (MP3_BR_V1_L3 if ver == 3 else MP3_BR_V2_L3)[bri] * 1000
                spf = 1152 if ver == 3 else 576
                frame = buf[i:i + 200]
                dur = None
                for tag in (b"Xing", b"Info"):
                    j = frame.find(tag)
                    if j >= 0:
                        flags = struct.unpack(">I", frame[j + 4:j + 8])[0]
                        if flags & 1:
                            frames = struct.unpack(">I", frame[j + 8:j + 12])[0]
                            dur = frames * spf / float(sr)
                        break
                if dur is None:
                    dur = max(size - start - i, 0) * 8.0 / br
                ch = 1 if ((buf[i + 3] >> 6) & 0x03) == 3 else 2
                return sr, dur, ch, "mp3:v%d" % ver
        i += 1
    raise ValueError("no mp3 frame sync")


def probe(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return riff_info(path)
    if ext == ".flac":
        return flac_info(path)
    if ext == ".mp3":
        return mp3_info(path)
    raise ValueError("unsupported ext " + ext)


def main():
    inv = json.load(io.open(INV, encoding="utf-8"))
    inv_rows = {r["dataset"]: r for r in inv["datasets"]}
    result, fails_all = {}, []
    for ds in sorted(os.listdir(ROOT)):
        ddir = os.path.join(ROOT, ds)
        if not os.path.isdir(ddir) or ds.startswith(("_", ".")):
            continue
        files = []
        for dp, _, fns in os.walk(ddir):
            for fn in fns:
                if os.path.splitext(fn)[1].lower() in AUDIO_EXT:
                    files.append(os.path.join(dp, fn))
        if not files:
            continue
        srs, chs, tools, fails = Counter(), Counter(), Counter(), []
        dur_sum = 0.0
        by_sub = defaultdict(lambda: [0, 0.0])
        for p in files:
            try:
                sr, dur, ch, tool = probe(p)
            except Exception as e:
                fails.append([os.path.relpath(p, ROOT), "%s: %s" % (type(e).__name__, str(e)[:80])])
                continue
            srs[sr] += 1
            chs[ch] += 1
            tools[tool] += 1
            dur_sum += dur
            rel = os.path.relpath(p, ddir).replace("\\", "/").split("/")
            key = "/".join(rel[:2]) if len(rel) > 2 else rel[0]
            by_sub[key][0] += 1
            by_sub[key][1] += dur
        result[ds] = {
            "files": len(files),
            "probed": sum(srs.values()),
            "fails": len(fails),
            "fail_examples": fails[:5],
            "sr_counts": {str(k): v for k, v in sorted(srs.items())},
            "channel_counts": {str(k): v for k, v in sorted(chs.items())},
            "sr_min": min(srs) if srs else None,
            "sr_max": max(srs) if srs else None,
            "duration_h": round(dur_sum / 3600.0, 2),
            "tools": dict(tools),
            "by_subdir": {k: [v[0], round(v[1] / 3600.0, 2)] for k, v in sorted(by_sub.items())},
        }
        fails_all += fails
        print("%-19s files=%5d probed=%5d fails=%3d sr=%-42s dur=%6.2fh"
              % (ds, len(files), result[ds]["probed"], len(fails),
                 ",".join("%d:%d" % kv for kv in sorted(srs.items())), result[ds]["duration_h"]))

    total_files = sum(r["files"] for r in result.values())
    total_h = round(sum(r["duration_h"] for r in result.values()), 2)
    inv_h = round(sum(r.get("duration_h", 0) for r in inv["datasets"]), 2)
    print("\ntotal audio %d files (inventory %d) / duration %.2f h (inventory %.2f h)"
          % (total_files, inv["_meta"]["total_audio"], total_h, inv_h))
    print("\nper-library reconciliation (scan vs inventory):")
    for ds, r in sorted(result.items()):
        ia = inv_rows.get(ds, {}).get("audio")
        ih = inv_rows.get(ds, {}).get("duration_h")
        print("  %-19s audio %5s / %5s  dur %6.2f / %6.2f  %s"
              % (ds, r["probed"], ia, r["duration_h"], ih or 0,
                 "OK" if ia == r["probed"] and abs((r["duration_h"] or 0) - (ih or 0)) <= 0.05 else "DIFF"))
    if fails_all:
        print("\n%d unparsed, examples:" % len(fails_all))
        for f in fails_all[:8]:
            print("   ", f)

    payload = {
        "_meta": {
            "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": "pure header parsing: wav via RIFF fmt+data, flac/STREAMINFO, mp3/first frame header + Xing; no decoding, no ffprobe",
            "root": ROOT,
            "total_files": total_files,
            "total_duration_h": total_h,
            "inventory_total_duration_h": inv_h,
            "unparsed": len(fails_all),
        },
        "datasets": result,
    }
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("\nwritten:", OUT)


if __name__ == "__main__":
    main()
