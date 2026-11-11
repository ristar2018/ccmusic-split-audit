# -*- coding: utf-8 -*-
"""manifest_neg.py — 三个 K̄≈1 负对照库的 sample_manifest（E1f 前置）

字段与 code/manifest.py 完全一致（直接复用其 COLUMNS），主样本五库的 manifest 不受影响。

覆盖（设计稿 §3.7 :296-310）：
  chest_falsetto     1,280 片 / 1,280 录音 / K̄=1.00  mel: data/eval/mel/<序号>_<m|f>_<chest|falsetto>_<片位>.jpg
  erhu_playing_tech  1,253 片 / 1,253 录音 / K̄=1.00  mel: data/eval/<技法>__<录音>_mel.jpg（扁目录）
  GZ_IsoTech         2,899 片 / 2,824 录音 / K̄=1.03  mel: data/eval/mel/<split>/<技法>/<录音>_<段>.jpg

Guzheng_Tech99 **不进本清单**（数据侧无逐录音类别层）：
  该库 99 条录音各 1 片，逐音符 IPT 标注可得（10 类 / 15,838 音符），但每条录音的
  主技法恒为 boxian（99/99），盘面没有「每条录音一个类别」的标签；K̄=1 且无类别的库
  构不成分类任务。证据在 data/manifest/summary_neg.json 的 excluded 段。
  → E1f 由设计稿的 16 runs 降为 12 runs（3 库 × 2 骨干 × {A0,A2} × 1 seed）。

用法
  python code/manifest_neg.py            # 三库全量
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
from manifest import COLUMNS, sha1_file  # noqa: E402  字段与哈希口径与主样本完全一致

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
        if not f.lower().endswith("_mel.jpg"):   # 只取 mel 通道，chroma/cqt 不进主线
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
        uid = "%s/%s" % (split, m.group("base"))   # 同一条长录音的多段共享 uid（这正是 K>1 的来源）
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
               "scope": "负对照 3 库（E1f: A0 vs A2，1 seed，2 骨干）",
               "design_ref": "report/聚焦版实验设计方案v2.8.md §3.7 :296-310",
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
        print("[%s] %d 片 / %d 录音 / %d 类 / K̄ %.4f / 坏名 %d → %s"
              % (lib, st["slices"], st["recordings"], st["classes"], st["kbar"] or 0,
                 st["bad_filename_count"], path), flush=True)
    summary["excluded"]["Guzheng_Tech99"] = {
        "reason": "无逐录音类别层：99 条录音各 1 片，逐音符 IPT 有 10 类（boxian 11,860 / "
                  "chanyin 1,994 / shanghua 756 / huazhi 663 / xianyin 类 208+209 / ...，共 15,838 音符），"
                  "但每条录音的主技法恒为 boxian（99/99），无法构成单标签分类任务",
        "evidence": "data/splits/v2/negcontrol/negcontrol.json + 逐 CSV 统计（2026-11-05）",
        "consequence": "E1f 由 16 runs 降为 12 runs（3 库 × 2 骨干 × {A0,A2} × 1 seed）",
    }
    sp = os.path.join(a.out, "summary_neg.json")
    with io.open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("→", sp, flush=True)
    return 2 if bad_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
