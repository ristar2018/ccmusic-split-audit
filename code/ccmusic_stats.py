# -*- coding: utf-8 -*-
"""Inventory the extracted data under D:\\VSCode\\Datasets: audio duration, sample rate, feature-map size, label rows.

Writes data/actual_stats.json so the README and the research plan can cite measured numbers instead of model-card claims.
Usage: python code/ccmusic_stats.py [--root D:\\VSCode\\Datasets]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT_DEFAULT = r"D:\VSCode\Datasets"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def walk_files(root: str):
    for dirpath, _, files in os.walk(root):
        for fn in files:
            yield os.path.join(dirpath, fn)


def audio_stat(path: str) -> tuple[float, int, int] | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.frames and info.samplerate:
            return info.frames / info.samplerate, info.samplerate, info.channels
    except Exception:  # noqa: BLE001
        pass
    try:  # mp3 fallback: when neither audioread nor mutagen is available, return None rather than crash
        import subprocess
        import re

        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip()), 0, 0
    except Exception:  # noqa: BLE001
        pass
    return None


def image_size(path: str) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()
    root = args.root

    out = {}
    for ds in sorted(os.listdir(root)):
        ddir = os.path.join(root, ds)
        if not os.path.isdir(ddir) or ds.startswith("_"):
            continue
        rec: dict = {"path": ddir, "ext": Counter(), "bytes": 0}
        n_audio = 0
        dur = 0.0
        sr = Counter()
        ch = Counter()
        no_dur = 0
        imgs = []
        csvs = []
        for p in walk_files(ddir):
            ext = os.path.splitext(p)[1].lower()
            rec["ext"][ext] += 1
            rec["bytes"] += os.path.getsize(p)
            if ext in AUDIO_EXT:
                n_audio += 1
                s = audio_stat(p)
                if s:
                    dur += s[0]
                    sr[s[1]] += 1
                    ch[s[2]] += 1
                else:
                    no_dur += 1
            elif ext in IMG_EXT and len(imgs) < 3:
                imgs.append(p)
            elif ext == ".csv" and len(csvs) < 3:
                csvs.append(p)
        rec["audio_files"] = n_audio
        rec["duration_h"] = round(dur / 3600, 2)
        rec["duration_s"] = round(dur, 1)
        rec["sr"] = dict(sr)
        rec["channels"] = dict(ch)
        rec["audio_no_duration"] = no_dur
        rec["ext"] = dict(rec["ext"].most_common())
        rec["bytes_h"] = human(rec["bytes"])
        rec["img_sizes"] = [image_size(p) for p in imgs]
        rec["img_sample"] = [os.path.relpath(p, ddir) for p in imgs]
        rec["csv_sample"] = []
        for p in csvs:
            try:
                with open(p, encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
                rec["csv_sample"].append(
                    {"file": os.path.relpath(p, ddir), "lines": len(lines), "head": lines[:3]}
                )
            except Exception:  # noqa: BLE001
                pass
        out[ds] = rec
        print(
            f"{ds:18s} audio={n_audio:>5} ({rec['duration_h']:>7.2f} h) "
            f"files={sum(rec['ext'].values()):>7} {rec['bytes_h']:>8} "
            f"sr={list(rec['sr'])[:3]} img={rec['img_sizes'][:1]}"
        )
        if no_dur:
            print(f"                   !! {no_dur} audio files with unreadable duration")

    dst_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "actual_stats.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"\nwrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
