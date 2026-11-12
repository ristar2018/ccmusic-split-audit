# -*- coding: utf-8 -*-
"""manifest_neg.py -- sample_manifest for the three K̄≈1 negative-control libraries (E1f prerequisite)

Fields are identical to code/manifest.py (its COLUMNS are reused directly); the manifests of the five main-sample libraries are untouched.

Coverage (design note §3.7 :296-310):
  chest_falsetto     1,280 片 / 1,280 录音 / K̄=1.00  mel: data/eval/mel/<序号>_<m|f>_<chest|falsetto>_<片位>.jpg
  erhu_playing_tech  1,253 片 / 1,253 录音 / K̄=1.00  mel: data/eval/<技法>__<录音>_mel.jpg（扁目录）
  GZ_IsoTech         2,899 片 / 2,824 录音 / K̄=1.03  mel: data/eval/mel/<split>/<技法>/<录音>_<段>.jpg

Guzheng_Tech99 **is excluded from this manifest** (the data side has no per-recording class layer):
  the library holds 99 recordings with 1 slice each, and per-note IPT labels exist (10 classes / 15,838 notes), but the
  dominant technique of every recording is boxian (99/99) and the disk has no "one class per recording" label; a library with
  K̄=1 and no classes cannot form a classification task. The evidence is in the excluded section of data/manifest/summary_neg.json.
  -> E1f drops from the 16 runs of the design note to 12 runs (3 libraries x 2 backbones x {A0,A2} x 1 seed).

Usage
  python code/manifest_neg.py            # all three libraries
  python code/manifest_neg.py --lib chest_falsetto --limit 400
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest import COLUMNS, sha1_file  # noqa: E402  fields and hash caliber identical to the main sample

ROOT = r"D:/VSCode/Datasets"
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "manifest")
LABEL_VERSION = {
    "chest_falsetto": "chest.sexmode4.v1",
    "erhu_playing_tech": "erhu.technique11.v1",
    "GZ_IsoTech": "guzheng.technique8.v1",
}
EXPECT = {
    "chest_falsetto": {"classes": 4, "recordings": 1280, "slices": 1280},
    "erhu_playing_tech": {"classes": 11, "recordings": 1253, "slices": 1253},
    "GZ_IsoTech": {"classes": 8, "recordings": 2824, "slices": 2899},
}
CHEST_RE = re.compile(r"^(?P<idx>\d+)_(?P<sex>[mf])_(?P<mode>chest|falsetto)_(?P<pos>\d+)\.jpg$")


def walk_jpg(base):
    for dp, _, fs in os.walk(base):
        for f in sorted(fs):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield dp, f


def rel(p):
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def row(lib, path, label, uid, slice_index, label_name="", top="", mid="", domain="real"):
    h = sha1_file(path)
    return {
        "sample_id": "%s:%s" % (lib, rel(path)),
        "dataset": lib,
        "feature": "mel",
        "relative_path": rel(path),
        "label": label,
        "recording_uid": uid,
        "candidate_group": "g" + h[:12],
        "label_version": LABEL_VERSION[lib],
        "content_hash": h,
        "domain": domain,
        "song_id": uid,
        "identity_level": "performance",
        "label_name": label_name or label,
        "slice_index": slice_index,
        "recording_duration_s": "",
        "instrument": lib.split("_")[0],
        "technique": "",
        "take": "",
        "top_category": top,
        "mid_category": mid,
        "audio_hash": "",
        "recording_canonical": uid,
    }


def build_chest(limit):
    base = os.path.join(ROOT, "chest_falsetto", "data", "eval", "mel")
    rows, bad = [], []
    for dp, f in walk_jpg(base):
        m = CHEST_RE.match(f)
        if not m:
            bad.append(f)
            continue
        label = "%s_%s" % (m.group("sex"), m.group("mode"))
        rows.append(row("chest_falsetto", os.path.join(dp, f), label, m.group("idx"),
                        int(m.group("pos")), label_name=label,
                        top="falsetto" if "falsetto" in label else "chest"))
        if limit and len(rows) >= limit:
            break
    return rows, bad


def build_erhu(limit):
    base = os.path.join(ROOT, "erhu_playing_tech", "data", "eval")
    rows, bad = [], []
    for dp, f in walk_jpg(base):
        if not f.lower().endswith("_mel.jpg"):   # mel channel only; chroma/cqt stay out of the main line
            continue
        if "__" not in f:
            bad.append(f)
            continue
        stem = f[:-len("_mel.jpg")]              # detache__detache_01
        tech = stem.split("__")[0]
        rows.append(row("erhu_playing_tech", os.path.join(dp, f), tech, stem, 0,
                        label_name=tech, top="technique"))
        if limit and len(rows) >= limit:
            break
    return rows, bad


def build_gziso(limit):
    base = os.path.join(ROOT, "GZ_IsoTech", "data", "eval", "mel")
    rows, bad = [], []
    for dp, f in walk_jpg(base):
        relp = os.path.relpath(dp, base)
        split = "eval" if relp == "." else relp.split(os.sep)[0]
        tech = os.path.basename(dp)
        stem = os.path.splitext(f)[0]
        m = re.match(r"^(?P<base>.+)_(?P<seg>\d+)$", stem)
        if not m:
            bad.append(f)
            continue
        uid = "%s/%s" % (split, m.group("base"))   # segments of one long recording share a uid (this is exactly where K>1 comes from)
        rows.append(row("GZ_IsoTech", os.path.join(dp, f), tech, uid, int(m.group("seg")),
                        label_name=tech, top=split, mid=tech))
        if limit and len(rows) >= limit:
            break
    return rows, bad


BUILDERS = {"chest_falsetto": build_chest, "erhu_playing_tech": build_erhu,
            "GZ_IsoTech": build_gziso}


def summarise(lib, rows, bad):
    per = {}
    uids = {}
    for r in rows:
        per.setdefault(r["label"], set()).add(r["recording_uid"])
        uids.setdefault(r["recording_uid"], 0)
        uids[r["recording_uid"]] += 1
    k = sorted(uids.values())
    n_rec, n_sl = len(uids), len(rows)
    return {
        "library": lib,
        "classes": len(per),
        "recordings": n_rec,
        "slices": n_sl,
        "kbar": round(n_sl / n_rec, 4) if n_rec else None,
        "k_dist": {str(x): k.count(x) for x in sorted(set(k))},
        "per_class_recordings": {c: len(v) for c, v in sorted(per.items())},
        "min_class_recordings": min((len(v) for v in per.values()), default=0),
        "bad_filename_count": len(bad),
        "bad_filename_examples": bad[:10],
        "uid_parse_rate": 1.0 if rows else 0.0,
        "label_version": LABEL_VERSION[lib],
        "duplicate_content_hash_groups": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="chest_falsetto,erhu_playing_tech,GZ_IsoTech")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--out", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    libs = [x.strip() for x in a.lib.split(",") if x.strip()]
    summary = {"generated_by": "code/manifest_neg.py",
               "scope": "3 negative-control libraries (E1f: A0 vs A2, 1 seed, 2 backbones)",
               "design_ref": "design note v2.8 §3.7 :296-310",
               "libraries": {}, "excluded": {}, "checks_vs_design": {}}
    bad_any = False
    for lib in libs:
        rows, bad = BUILDERS[lib](a.limit)
        if a.no_hash:
            for r in rows:
                r["content_hash"], r["candidate_group"] = "", "g"
        path = os.path.join(a.out, "%s.csv" % lib)
        with io.open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        st = summarise(lib, rows, bad)
        summary["libraries"][lib] = st
        if not a.limit:
            e = EXPECT[lib]
            ok = (st["classes"] == e["classes"] and st["recordings"] == e["recordings"]
                  and st["slices"] == e["slices"])
            summary["checks_vs_design"][lib] = {"expect": e, "got": {
                k: st[k] for k in ("classes", "recordings", "slices")}, "ok": ok}
            bad_any = bad_any or not ok
        print("[%s] %d slices / %d recordings / %d classes / K̄ %.4f / bad names %d -> %s"
              % (lib, st["slices"], st["recordings"], st["classes"], st["kbar"] or 0,
                 st["bad_filename_count"], path), flush=True)
    summary["excluded"]["Guzheng_Tech99"] = {
        "reason": "no per-recording class layer: 99 recordings with 1 slice each, per-note IPT has 10 classes (boxian 11,860 / "
                  "chanyin 1,994 / shanghua 756 / huazhi 663 / xianyin classes 208+209 / ..., 15,838 notes in total), "
                  "but the dominant technique of every recording is boxian (99/99), so no single-label classification task is possible",
        "evidence": "data/splits/v2/negcontrol/negcontrol.json + per-CSV statistics (2026-11-05)",
        "consequence": "E1f drops from 16 runs to 12 runs (3 libraries x 2 backbones x {A0,A2} x 1 seed)",
    }
    sp = os.path.join(a.out, "summary_neg.json")
    with io.open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("→", sp, flush=True)
    return 2 if bad_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
