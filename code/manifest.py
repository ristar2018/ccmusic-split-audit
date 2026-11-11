# -*- coding: utf-8 -*-
"""样本清单 sample_manifest 构建（依据 report/聚焦版实验设计方案v2.8.md §4.1）

覆盖两库（其余三库待补）：
  CTIS         mel 219 类 / 4,956 录音 / 43,054 切片 / 32.63 h 音频
  music_genre  mel  16 类 / 1,713 录音 / 36,375 切片 / 无音频（仅特征）

只读 D:\\VSCode\\Datasets；写 data/manifest/<lib>.csv + data/manifest/summary.json。

字段（§4.1 全集 + 本库需要的扩展列）：
  sample_id          dataset + ":" + relative_path，全库唯一
  dataset            库名
  feature            mel
  relative_path      相对 D:\\VSCode\\Datasets 的路径（/ 分隔）
  label              分类标签（CTIS=类码 C0090；music_genre=叶子类目录名）
  recording_uid      录音身份（见下"身份规则"）
  candidate_group    内容完全相同的切片同组（"g"+sha1[:12]）；录音级互斥由 splits_v2 的 A0 保证
  label_version      标签体系版本
  content_hash       图片字节 sha1（全 40 位）
  domain             synth/real（本两库均 real，依据见 summary.domain_basis）
  song_id            CTIS=""（单音源库，无歌曲层，§4.1 允许留空）；music_genre=recording_uid（一轨一作品）
  identity_level     CTIS=performance（每条 wav 是一次演奏）；music_genre=work（一轨一作品）
扩展列：label_name, slice_index, recording_duration_s, instrument, technique, take,
        top_category, mid_category（music_genre 三级目录用）

身份规则（两库各不相同，不能共用"去尾部数字"的 src_of）
  CTIS:        mel 文件名 = <全局录音号>_<切片位置>.jpg，全局录音号按
               「类名升序 → 类内 wav 文件名升序」跨类累加（C0090=0..27, C0091=28..57, …）
               → recording_uid = 该 wav 的文件名主干
  music_genre: mel 文件名 = <轨道序号>_<时长秒>_<切片列位置>.jpg
               → recording_uid = "<叶子类>/<轨道序号>_<时长秒>"

用法
  python code/manifest.py                       # 两库全量
  python code/manifest.py --lib CTIS --limit 2000   # 冒烟（只取前 2000 条切片）
  python code/manifest.py --lib CTIS,music_genre --no-hash   # 跳过 sha1（调试用）
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict

ROOT = r"D:\VSCode\Datasets"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "data", "manifest")

# 与 code/ctis_capacity.py:25 同口径
CTIS_RE = re.compile(r"^【(?P<inst>[^】]+)】\s*(?P<rest>.+?)\s*\.wav$")

COMMON = ["sample_id", "dataset", "feature", "relative_path", "label", "recording_uid",
          "candidate_group", "label_version", "content_hash", "domain", "song_id",
          "identity_level"]
EXTRA = ["label_name", "slice_index", "recording_duration_s", "instrument", "technique",
         "take", "top_category", "mid_category"]
# 去重列：audio_hash = 录音文件字节 sha1（CTIS 才有音频）；recording_canonical = 同一音频的
# 多个发布名折叠后的规范 id（= 组内字典序最小 uid）。splits_v2 的 A0/A2 用 canonical 判互斥。
DEDUP = ["audio_hash", "recording_canonical"]
COLUMNS = COMMON + EXTRA + DEDUP

EXPECT = {
    "CTIS": {"classes": 219, "recordings": 4956, "slices": 43054, "hours": 32.63},
    "music_genre": {"classes": 16, "recordings": 1713, "slices": 36375},
    "pianos": {"classes": 8, "recordings": 668, "slices": 18352},
    "CNPM": {"classes": 5, "recordings": 287, "slices": 2724},
    "bel_canto": {"classes": 4, "recordings": 203, "slices": 9910},
}


# ---------------------------------------------------------------- 工具
def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(4 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def wav_duration(path):
    """纯文件头解析（wave 模块打不开 IEEE float WAV；soundfile 在本环境取不到）"""
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            return None
        f.read(8)
        while True:
            ck = f.read(8)
            if len(ck) < 8:
                return None
            cid, size = ck[:4], struct.unpack("<I", ck[4:])[0]
            if cid == b"fmt ":
                body = f.read(size)
                if len(body) < 16:
                    return None
                _fmt, _ch, _sr, br, _ba, _bits = struct.unpack("<HHIIHH", body[:16])
                while True:
                    ck2 = f.read(8)
                    if len(ck2) < 8:
                        return None
                    cid2, size2 = ck2[:4], struct.unpack("<I", ck2[4:])[0]
                    if cid2 == b"data":
                        return (size2 / float(br)) if br else None
                    f.seek(size2 + (size2 & 1), 1)
            f.seek(size + (size & 1), 1)


def parse_ctis_name(stem):
    """【乐器】演奏技法 华彩 第一遍 → (大笒, 华彩, 第一遍)；解析失败返回 (None,None,None)"""
    m = CTIS_RE.match(stem + ".wav")
    if not m:
        return None, None, None
    parts = m.group("rest").split()
    if not parts:
        return m.group("inst"), None, None
    if parts[0].startswith("演奏"):
        tech = parts[1] if len(parts) > 1 else None
    else:
        tech = parts[0]
    take = None
    for p in parts:
        if p.startswith("第") and p.endswith("遍"):
            take = p
    return m.group("inst"), tech, take


def write_csv(rows, path):
    import csv
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


SPOT_COLS = ["spot_id", "library", "label", "recording_uid", "recording_canonical",
             "n_slices", "duration_s", "slice_positions", "sample_image", "sample_audio",
             "check_image_matches_recording", "check_no_other_recording_same", "note"]


def write_spotcheck(rows, lib, outdir, n=100, seed=20261105):
    """G-A 身份人工抽查表：每库抽 n 条录音，留三列空白给人判（100% 才算过门）"""
    import csv
    import random
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["recording_canonical"]].append(r)
    keys = sorted(by_rec)
    pick = random.Random(seed).sample(keys, min(n, len(keys)))
    out = []
    for i, uid in enumerate(sorted(pick), 1):
        rs = sorted(by_rec[uid], key=lambda r: int(r["slice_index"]))
        poss = [int(r["slice_index"]) for r in rs]
        mid = rs[len(rs) // 2]
        audio = ""
        if lib == "CTIS":
            audio = "CTIS/data/audio/%s/%s.wav" % (mid["label"], uid.split("/", 1)[-1])
        out.append({"spot_id": "%s-%03d" % (lib, i), "library": lib,
                    "label": mid["label"], "recording_uid": mid["recording_uid"],
                    "recording_canonical": uid, "n_slices": len(rs),
                    "duration_s": mid["recording_duration_s"],
                    "slice_positions": " ".join(str(p) for p in poss[:6]) +
                                       (" ... %d" % poss[-1] if len(poss) > 6 else ""),
                    "sample_image": mid["relative_path"], "sample_audio": audio,
                    "check_image_matches_recording": "", "check_no_other_recording_same": "",
                    "note": ""})
    path = os.path.join(outdir, lib + "_spotcheck.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SPOT_COLS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    return path, len(out)


# ---------------------------------------------------------------- CTIS
def build_ctis(limit=None, do_hash=True, verbose=True, do_audio_hash=True):
    root = os.path.join(ROOT, "CTIS", "data")
    adir = os.path.join(root, "audio")
    mdir = os.path.join(root, "eval", "mel")
    classes = sorted(d for d in os.listdir(adir) if os.path.isdir(os.path.join(adir, d)))

    # 1) 全局录音号 → (类码, wav 名)；全局号 = 类名升序 + 类内文件名升序的累加序号
    recs = {}
    g = 0
    for cls in classes:
        for fn in sorted(os.listdir(os.path.join(adir, cls))):
            if fn.lower().endswith(".wav"):
                recs[g] = (cls, fn)
                g += 1
    n_rec = len(recs)

    # 2) 扫 mel，按全局号归组
    slices = defaultdict(list)          # gidx -> [(pos, filename)]
    other_ext = Counter()
    for cls in classes:
        cd = os.path.join(mdir, cls)
        if not os.path.isdir(cd):
            continue
        for fn in os.listdir(cd):
            if not fn.lower().endswith(".jpg"):
                other_ext[os.path.splitext(fn)[1].lower()] += 1
                continue
            gidx, pos = fn.rsplit(".", 1)[0].split("_")
            slices[int(gidx)].append((int(pos), fn))

    orphan = sorted(set(slices) - set(recs))
    missing = sorted(set(recs) - set(slices))

    # 3) 乐器名统计（类码 → 主导乐器）与取文件名唯一性判定
    inst_of = {}
    stems = Counter()
    parse_fail = []
    for gidx, (cls, fn) in recs.items():
        stem = fn.rsplit(".", 1)[0]
        stems[stem] += 1
        inst, tech, take = parse_ctis_name(stem)
        inst_of.setdefault(cls, Counter())[inst if inst else "<unparsed>"] += 1
        if inst is None:
            parse_fail.append({"class": cls, "file": fn})

    uid_prefix = any(c > 1 for c in stems.values())
    per_class_inst = {c: cnt.most_common() for c, cnt in inst_of.items()}
    mixed = {c: v for c, v in per_class_inst.items() if len(v) > 1}

    # 3b) 音频去重：同一段音频被挂两个名字（如 L0148 的 "...第一遍(1)"）必须折成同一录音，
    #     否则 A0 的录音级互斥会被同一音频的两份拷贝绕过。
    uid_of = {}
    for gidx, (cls, fn) in recs.items():
        stem = fn.rsplit(".", 1)[0]
        uid_of[gidx] = (cls + "/" + stem) if uid_prefix else stem
    audio_hash = {}
    if do_audio_hash:
        print("  音频去重：对 %d 个 wav 取字节 sha1 ..." % n_rec, flush=True)
        for i, gidx in enumerate(sorted(recs), 1):
            cls, fn = recs[gidx]
            audio_hash[uid_of[gidx]] = sha1_file(os.path.join(adir, cls, fn))
            if verbose and i % 1000 == 0:
                print("    audio %d/%d ..." % (i, n_rec), flush=True)
    canon = {}
    if audio_hash:
        by_hash = defaultdict(list)
        for uid, h in audio_hash.items():
            by_hash[h].append(uid)
        dup_audio = {h: sorted(v) for h, v in by_hash.items() if len(v) > 1}
        for h, uids in by_hash.items():
            for uid in uids:
                canon[uid] = uids[0]
    else:
        dup_audio = {}
        canon = {uid: uid for uid in uid_of.values()}

    # 4) 生成行
    rows, n_parsed, n_uid, dur_total = [], 0, 0, 0.0
    lim = 0
    by_class = defaultdict(list)
    for gidx, (c, _fn) in recs.items():
        by_class[c].append(gidx)
    for cls in classes:
        for gidx in sorted(by_class[cls]):
            fn = recs[gidx][1]
            stem = fn.rsplit(".", 1)[0]
            inst, tech, take = parse_ctis_name(stem)
            dur = wav_duration(os.path.join(adir, cls, fn))
            dur_total += dur or 0.0
            uid = (cls + "/" + stem) if uid_prefix else stem
            dom_inst = per_class_inst[cls][0][0] if per_class_inst[cls] else ""
            for pos, jpg in sorted(slices.get(gidx, [])):
                rel = "CTIS/data/eval/mel/%s/%s" % (cls, jpg)
                abs_p = os.path.join(mdir, cls, jpg)
                h = sha1_file(abs_p) if do_hash else ""
                rows.append({
                    "sample_id": "CTIS:" + rel,
                    "dataset": "CTIS", "feature": "mel", "relative_path": rel,
                    "label": cls, "recording_uid": uid,
                    "candidate_group": ("g" + h[:12]) if h else "",
                    "label_version": "ctis.class219.v1",
                    "content_hash": h, "domain": "real", "song_id": "",
                    "identity_level": "performance",
                    "label_name": dom_inst, "slice_index": pos,
                    "recording_duration_s": ("%.2f" % dur) if dur else "",
                    "instrument": inst or "", "technique": tech or "", "take": take or "",
                    "top_category": "", "mid_category": "",
                    "audio_hash": audio_hash.get(uid, ""),
                    "recording_canonical": canon.get(uid, uid),
                })
                n_parsed += 1 if inst else 0
                n_uid += 1
                lim += 1
                if verbose and lim % 5000 == 0:
                    print("  CTIS %d slices ..." % lim, flush=True)
                if limit and lim >= limit:
                    return rows, _ctis_stats(rows, recs, slices, classes, per_class_inst, mixed,
                                             parse_fail, orphan, missing, other_ext,
                                             n_rec, dur_total, limit, uid_prefix, dup_audio)

    return rows, _ctis_stats(rows, recs, slices, classes, per_class_inst, mixed,
                             parse_fail, orphan, missing, other_ext, n_rec, dur_total, None,
                             uid_prefix, dup_audio)


def _ctis_stats(rows, recs, slices, classes, per_class_inst, mixed, parse_fail,
                orphan, missing, other_ext, n_rec, dur_total, limit, uid_prefix, dup_audio):
    n = len(rows)
    recs_seen = len({r["recording_uid"] for r in rows})
    per_class = {}
    for r in rows:
        d = per_class.setdefault(r["label"], {"recordings": set(), "slices": 0})
        d["recordings"].add(r["recording_uid"])
        d["slices"] += 1
    per_class_out = {c: {"recordings": len(v["recordings"]), "slices": v["slices"],
                         "K": round(v["slices"] / len(v["recordings"]), 2) if v["recordings"] else 0}
                     for c, v in sorted(per_class.items())}
    uid_rate = (sum(1 for r in rows if r["instrument"]) / n) if n else 0.0
    st = {
        "library": "CTIS",
        "audio_root": "CTIS/data/audio",
        "mel_root": "CTIS/data/eval/mel",
        "classes": len(classes) if not limit else len(per_class),
        "recordings_on_disk": n_rec,
        "recordings_with_slices": sum(1 for gd in recs if slices.get(gd)),
        "recordings_in_manifest": recs_seen,
        "slices": n,
        "K": round(n / recs_seen, 2) if recs_seen else 0,
        "audio_hours": round(dur_total / 3600.0, 2),
        "recordings_without_mel": missing,
        "mel_gidx_without_audio": orphan,
        "non_jpg_in_mel": dict(other_ext),
        "instrument_parse_ok": n - sum(1 for r in rows if not r["instrument"]),
        "uid_parse_rate": round(uid_rate, 4),
        "unparsed_files": parse_fail,
        "per_class": per_class_out,
        "classes_with_multi_instrument": {c: v for c, v in mixed.items()},
        "domain": "real",
        "domain_basis": "盘内有 wav 音频（4,956 条真实乐器演奏录音，32.63 h），非合成",
        "song_id_rule": "留空：CTIS 为单音源库（一条 wav = 一次演奏），无歌曲/作品层，§4.1 允许留空",
        "identity_level": "performance",
        "uid_rule": "mel 首段全局录音号按「类名升序→类内 wav 名升序」跨类累加；uid = wav 文件名主干（%s）"
                    % ("文件名跨类重名 → uid 加类码前缀" if uid_prefix else "无重名"),
        "audio_hash_done": bool(dup_audio),
        "duplicate_audio_groups": dup_audio,
        "duplicate_audio_recordings": sum(len(v) for v in dup_audio.values()),
        "dedup_rule": "audio_hash 相同的多个发布名折成 recording_canonical（组内字典序最小 uid）；"
                      "splits_v2 的 A0/A2 按 canonical 判互斥",
        "limit": limit,
    }
    return st


# ---------------------------------------------------------------- music_genre
def build_music_genre(limit=None, do_hash=True, verbose=True):
    root = os.path.join(ROOT, "music_genre", "data", "eval", "mel")
    rows, lim = [], 0
    per_class = defaultdict(lambda: {"slices": 0, "recs": set(), "dur": 0.0})
    bad_name = []
    meta = {}   # rel dir -> (top, mid)
    for dirpath, _dirs, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".jpg"):
                continue
            rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
            segs = rel_dir.split("/")
            leaf = segs[-1]
            top = segs[0] if len(segs) >= 2 else ""
            mid = "/".join(segs[1:-1]) if len(segs) >= 3 else ""
            stem = fn.rsplit(".", 1)[0]
            parts = stem.split("_")
            if len(parts) != 3:
                bad_name.append({"dir": rel_dir, "file": fn})
                continue
            track, dur_s, pos = parts
            try:
                dur = float(dur_s)
                pos_i = int(pos)
            except ValueError:
                bad_name.append({"dir": rel_dir, "file": fn})
                continue
            uid = "%s/%s_%s" % (leaf, track, dur_s)
            rel = "music_genre/data/eval/mel/%s/%s" % (rel_dir, fn)
            h = sha1_file(os.path.join(dirpath, fn)) if do_hash else ""
            d = per_class[leaf]
            if uid not in d["recs"]:
                d["recs"].add(uid)
                d["dur"] += dur
            d["slices"] += 1
            rows.append({
                "sample_id": "music_genre:" + rel,
                "dataset": "music_genre", "feature": "mel", "relative_path": rel,
                "label": leaf, "recording_uid": uid,
                "candidate_group": ("g" + h[:12]) if h else "",
                "label_version": "mg.leaf16.v1",
                "content_hash": h, "domain": "real", "song_id": uid,
                "identity_level": "work",
                "label_name": leaf, "slice_index": pos_i, "recording_duration_s": dur_s,
                "instrument": "", "technique": "", "take": "",
                "top_category": top, "mid_category": mid,
                "audio_hash": "", "recording_canonical": uid,
            })
            lim += 1
            if verbose and lim % 5000 == 0:
                print("  music_genre %d slices ..." % lim, flush=True)
            if limit and lim >= limit:
                return rows, _mg_stats(rows, per_class, bad_name, limit)

    return rows, _mg_stats(rows, per_class, bad_name, None)


def _mg_stats(rows, per_class, bad_name, limit):
    n = len(rows)
    recs = len({r["recording_uid"] for r in rows})
    pc = {c: {"recordings": len(v["recs"]), "slices": v["slices"],
              "K": round(v["slices"] / len(v["recs"]), 2) if v["recs"] else 0,
              "audio_hours": round(v["dur"] / 3600, 2)}
          for c, v in sorted(per_class.items())}
    return {
        "library": "music_genre",
        "mel_root": "music_genre/data/eval/mel",
        "classes": len({r["label"] for r in rows}),
        "recordings_in_manifest": recs,
        "slices": n,
        "K": round(n / recs, 2) if recs else 0,
        "audio_hours": 0.0,
        "audio_note": "盘面无音频，仅 eval 特征；录音时长取自文件名中段（秒）",
        "duration_from_filename_h": round(sum(v["dur"] for v in per_class.values()) / 3600.0, 2),
        "bad_filenames": bad_name[:20],
        "bad_filename_count": len(bad_name),
        "uid_parse_rate": round((n - len(bad_name)) / n, 4) if n else 0.0,
        "per_class": pc,
        "domain": "real",
        "domain_basis": "MTG-Jamendo 真实曲目录音（上游官方描述；本地仅有特征、无音频与元数据可复核）",
        "song_id_rule": "song_id = recording_uid：一轨即一作品，作品层与录音层同一标识",
        "identity_level": "work",
        "uid_rule": "mel 文件名 <轨道序号>_<时长秒>_<切片列位置> → uid = <叶子类>/<轨道序号>_<时长秒>",
        "audio_hash_done": False,
        "dedup_rule": "盘面无音频，无法做音频级去重；recording_canonical = recording_uid",
        "limit": limit,
    }


# ---------------------------------------------------------------- 另外三库（P1 主线补齐）
def _generic_stats(lib, rows, per_class, bad, limit, uid_rule, identity):
    n = len(rows)
    recs = {r["recording_uid"] for r in rows}
    pc = {c: {"recordings": len(v["recs"]), "slices": v["slices"],
              "K": round(v["slices"] / len(v["recs"]), 2) if v["recs"] else 0,
              "audio_hours": round(v["dur"] / 3600, 2)}
          for c, v in sorted(per_class.items())}
    return {
        "library": lib,
        "classes": len({r["label"] for r in rows}),
        "recordings_in_manifest": len(recs),
        "slices": n,
        "K": round(n / len(recs), 2) if recs else 0,
        "audio_hours": round(sum(v["dur"] for v in per_class.values()) / 3600.0, 2),
        "per_class": pc,
        "bad_filenames": bad[:20],
        "bad_filename_count": len(bad),
        "uid_parse_rate": round(1.0 - len(bad) / n, 4) if n else 0.0,
        "uid_rule": uid_rule,
        "identity_level": identity,
        "domain": "real",
        "audio_hash_done": False,
        "dedup_rule": "盘面音频无可靠命名映射，未做音频级去重；recording_canonical = recording_uid",
        "limit": limit,
    }


def build_pianos(limit=None, do_hash=True, verbose=True):
    """pianos：mel 在 `data/eval/<品牌>/<录音号>_<时长秒>_<片位>.jpg`（8 品牌 = 8 类）。

    `data/mel/` 是每录音一张图的旧目录，**不是**主线切片目录；主线用 `data/eval/`（18,352 片）。
    音频在 `data/audio/<品牌>/<录音号>.wav`，与 eval 的录音号不同名（6009 vs 009），无可靠映射，
    故不填 audio_hash；时长取自文件名中段。
    """
    root = os.path.join(ROOT, "pianos", "data", "eval")
    pat = re.compile(r"^(?P<rid>\d+)_(?P<dur>[0-9.]+)_(?P<pos>\d+)\.jpg$")
    rows, lim, bad = [], 0, []
    per_class = defaultdict(lambda: {"slices": 0, "recs": set(), "dur": 0.0})
    for brand in sorted(os.listdir(root)):
        bdir = os.path.join(root, brand)
        if not os.path.isdir(bdir):
            continue
        for fn in sorted(os.listdir(bdir)):
            m = pat.match(fn)
            if not m:
                bad.append({"dir": brand, "file": fn})
                continue
            dur_s, pos_i = m.group("dur"), int(m.group("pos"))
            uid = "%s/%s_%s" % (brand, m.group("rid"), dur_s)
            rel = "pianos/data/eval/%s/%s" % (brand, fn)
            h = sha1_file(os.path.join(bdir, fn)) if do_hash else ""
            d = per_class[brand]
            if uid not in d["recs"]:
                d["recs"].add(uid)
                d["dur"] += float(dur_s)
            d["slices"] += 1
            rows.append({
                "sample_id": "pianos:" + rel, "dataset": "pianos", "feature": "mel",
                "relative_path": rel, "label": brand, "recording_uid": uid,
                "candidate_group": ("g" + h[:12]) if h else "",
                "label_version": "pianos.brand8.v1", "content_hash": h,
                "domain": "real", "song_id": uid, "identity_level": "performance",
                "label_name": brand, "slice_index": pos_i, "recording_duration_s": dur_s,
                "instrument": "piano", "technique": "", "take": "",
                "top_category": brand, "mid_category": "",
                "audio_hash": "", "recording_canonical": uid,
            })
            lim += 1
            if verbose and lim % 5000 == 0:
                print("  pianos %d slices ..." % lim, flush=True)
            if limit and lim >= limit:
                break
        if limit and lim >= limit:
            break
    return rows, _generic_stats("pianos", rows, per_class, bad,
                                limit if limit and lim >= limit else None,
                                uid_rule="<品牌>/<录音号>_<时长秒>", identity="performance")


def build_CNPM(limit=None, do_hash=True, verbose=True):
    """CNPM：mel 在 `data/eval/mel/<曲名> - <演奏者>.mp3_<片位>.jpg`（扁目录），标签取 GBK `label.csv`
    的「样式/Pattern」（5 类，实测 0–4）。录音 id = 去最后一个 `_<片位>`（含 .mp3/.wav 扩展名），
    与 csv 的「文件名/File Name」100% 对上（287/287，双向 0 差）。
    """
    root = os.path.join(ROOT, "CNPM", "data")
    mel = os.path.join(root, "eval", "mel")
    with open(os.path.join(root, "label.csv"), "rb") as f:
        txt = f.read().decode("gbk")
    meta = {}
    for r in csv.DictReader(io.StringIO(txt)):
        meta[r["文件名/File Name"]] = r
    pat = re.compile(r"^(?P<uid>.+)_(?P<pos>\d+)\.jpg$")
    rows, lim, bad, nomatch = [], 0, [], set()
    per_class = defaultdict(lambda: {"slices": 0, "recs": set(), "dur": 0.0})
    for fn in sorted(os.listdir(mel)):
        m = pat.match(fn) if fn.lower().endswith(".jpg") else None
        if not m:
            bad.append({"dir": "", "file": fn})
            continue
        uid, pos_i = m.group("uid"), int(m.group("pos"))
        r = meta.get(uid)
        dur_s = ""
        if r is None:
            nomatch.add(uid)
            lbl = "?"
        else:
            lbl = "pattern_" + str(r["样式/Pattern"]).strip()
            try:
                mmss = r["时长/Length"].strip()
                if mmss:
                    a, b2 = mmss.split(":")
                    dur_s = "%.2f" % (int(a) * 60 + int(b2))
            except Exception:
                dur_s = ""
        rel = "CNPM/data/eval/mel/%s" % fn
        h = sha1_file(os.path.join(mel, fn)) if do_hash else ""
        d = per_class[lbl]
        if uid not in d["recs"]:
            d["recs"].add(uid)
            if dur_s:
                d["dur"] += float(dur_s)
        d["slices"] += 1
        rows.append({
            "sample_id": "CNPM:" + rel, "dataset": "CNPM", "feature": "mel",
            "relative_path": rel, "label": lbl, "recording_uid": uid,
            "candidate_group": ("g" + h[:12]) if h else "",
            "label_version": "cnpm.pattern5.v1", "content_hash": h,
            "domain": "real", "song_id": uid, "identity_level": "performance",
            "label_name": lbl, "slice_index": pos_i, "recording_duration_s": dur_s,
            "instrument": "", "technique": "", "take": "",
            "top_category": "CNPM", "mid_category": "",
            "audio_hash": "", "recording_canonical": uid,
        })
        lim += 1
        if verbose and lim % 500 == 0:
            print("  CNPM %d slices ..." % lim, flush=True)
        if limit and lim >= limit:
            break
    st = _generic_stats("CNPM", rows, per_class, bad,
                        limit if limit and lim >= limit else None,
                        uid_rule="mel 文件名去 `_<片位>`（含扩展名）", identity="performance")
    st["label_source"] = "CNPM/data/label.csv（GBK，列「样式/Pattern」5 类）"
    st["csv_join"] = {"csv_rows": len(meta),
                      "mel_recs": len({r["recording_uid"] for r in rows}),
                      "mel_not_in_csv": len(nomatch)}
    return rows, st


def build_bel_canto(limit=None, do_hash=True, verbose=True):
    """bel_canto：mel 在 `data/eval/mel/<f_bel|f_folk|m_bel|m_folk>/<录音号>_<片位>.jpg`（4 类）。"""
    root = os.path.join(ROOT, "bel_canto", "data", "eval", "mel")
    pat = re.compile(r"^(?P<rid>\d+)_(?P<pos>\d+)\.jpg$")
    rows, lim, bad = [], 0, []
    per_class = defaultdict(lambda: {"slices": 0, "recs": set(), "dur": 0.0})
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            bad.append({"dir": "", "file": cls})
            continue
        for fn in sorted(os.listdir(cdir)):
            m = pat.match(fn)
            if not m:
                bad.append({"dir": cls, "file": fn})
                continue
            uid = "%s/%s" % (cls, m.group("rid"))
            rel = "bel_canto/data/eval/mel/%s/%s" % (cls, fn)
            h = sha1_file(os.path.join(cdir, fn)) if do_hash else ""
            per_class[cls]["recs"].add(uid)
            per_class[cls]["slices"] += 1
            rows.append({
                "sample_id": "bel_canto:" + rel, "dataset": "bel_canto", "feature": "mel",
                "relative_path": rel, "label": cls, "recording_uid": uid,
                "candidate_group": ("g" + h[:12]) if h else "",
                "label_version": "belcanto.sexstyle4.v1", "content_hash": h,
                "domain": "real", "song_id": uid, "identity_level": "performance",
                "label_name": cls, "slice_index": int(m.group("pos")),
                "recording_duration_s": "",
                "instrument": "", "technique": "", "take": "",
                "top_category": cls.split("_")[0], "mid_category": cls.split("_")[-1],
                "audio_hash": "", "recording_canonical": uid,
            })
            lim += 1
            if verbose and lim % 2000 == 0:
                print("  bel_canto %d slices ..." % lim, flush=True)
            if limit and lim >= limit:
                break
        if limit and lim >= limit:
            break
    return rows, _generic_stats("bel_canto", rows, per_class, bad,
                                limit if limit and lim >= limit else None,
                                uid_rule="<类别>/<录音号>", identity="performance")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="CTIS,music_genre")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-hash", action="store_true", help="跳过图片 sha1（调试）")
    ap.add_argument("--no-audio-hash", action="store_true", help="跳过 CTIS wav 字节 sha1（约 13.3 GB）")
    ap.add_argument("--out", default=OUTDIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = {"built_by": "code/manifest.py", "root": ROOT,
               "feature": "mel", "columns": COLUMNS, "libraries": {}}
    for lib in [x.strip() for x in a.lib.split(",") if x.strip()]:
        print("[%s] 扫描 ..." % lib, flush=True)
        if lib == "CTIS":
            rows, st = build_ctis(a.limit or None, not a.no_hash,
                                  do_audio_hash=not a.no_audio_hash and not a.limit)
        elif lib == "music_genre":
            rows, st = build_music_genre(a.limit or None, not a.no_hash)
        elif lib == "pianos":
            rows, st = build_pianos(a.limit or None, not a.no_hash)
        elif lib == "CNPM":
            rows, st = build_CNPM(a.limit or None, not a.no_hash)
        elif lib == "bel_canto":
            rows, st = build_bel_canto(a.limit or None, not a.no_hash)
        else:
            print("  跳过（未实现）：%s" % lib)
            continue
        rows.sort(key=lambda r: (r["label"], r["recording_uid"], r["slice_index"]))
        p = os.path.join(a.out, lib + ".csv")
        write_csv(rows, p)
        exp = EXPECT.get(lib, {})
        checks = {}
        for k in ("classes", "recordings", "slices"):
            got = st.get("classes" if k == "classes" else
                         "recordings_in_manifest" if k == "recordings" else "slices")
            if k in exp:
                checks[k] = {"expected": exp[k], "got": got, "ok": got == exp[k]}
        if "hours" in exp and not a.limit:
            checks["audio_hours"] = {"expected": exp["hours"], "got": st.get("audio_hours"),
                                     "ok": abs((st.get("audio_hours") or 0) - exp["hours"]) < 0.02}
        st["checks_vs_design"] = checks
        st["csv"] = p
        st["csv_rows"] = len(rows)
        if not a.limit:
            sp2, nsp = write_spotcheck(rows, lib, a.out)
            st["spotcheck"] = {"path": sp2, "n": nsp}
        summary["libraries"][lib] = st
        ok = all(v["ok"] for v in checks.values()) if checks else None
        print("[%s] %d 行 → %s  校验=%s" % (lib, len(rows), p, ok), flush=True)
        for k, v in checks.items():
            if not v["ok"]:
                print("   !! %s 期望 %s 实得 %s" % (k, v["expected"], v["got"]))
    sp = os.path.join(a.out, "summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("summary → %s" % sp)
    allok = all(v["ok"] for lib in summary["libraries"].values()
                for v in lib["checks_vs_design"].values())
    if not allok and summary["libraries"]:
        print("!! 有校验未通过")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
