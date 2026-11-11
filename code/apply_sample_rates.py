# -*- coding: utf-8 -*-
"""把 data/sample_rates.json 的复核结果写回 data/actual_stats.json 的 sr / channels。

只改这两个字段（原值全是 {"0": n}，soundfile 取采样率失败的残留），其余字段不动；
写前备份到 _archive/actual_stats-2026-11-04-before-sr.json。
"""
import io, json, shutil, os
from collections import Counter

STATS = r"D:\VSCode\科研任务\音乐情感研究\data\actual_stats.json"
SRC = r"D:\VSCode\科研任务\音乐情感研究\data\sample_rates.json"
BAK = r"D:\VSCode\科研任务\音乐情感研究\_archive\actual_stats-2026-11-04-before-sr.json"

sr = json.load(io.open(SRC, encoding="utf-8"))["datasets"]
stats = json.load(io.open(STATS, encoding="utf-8"))

changed, missing = [], []
for ds, row in sorted(stats.items()):
    if not isinstance(row, dict):
        continue
    src = sr.get(ds)
    if not src or not src["sr_counts"]:
        if row.get("audio_files"):
            missing.append(ds)
        continue
    old_sr, old_ch = row.get("sr"), row.get("channels")
    row["sr"] = src["sr_counts"]
    row["channels"] = src["channel_counts"]
    row["sr_audit"] = {
        "date": "2026-11-04",
        "method": "文件头解析（wav/RIF fmt+data、flac/STREAMINFO、mp3/首帧头+Xing），不解码",
        "source": "data/sample_rates.json",
        "probed": src["probed"],
        "unparsed": src["fails"],
    }
    changed.append((ds, old_sr, row["sr"], src["probed"], src["fails"]))

if missing:
    print("!! 有音频但无复核数据:", missing)

if not os.path.exists(BAK):
    shutil.copyfile(STATS, BAK)
    print("backup ->", BAK)

io.open(STATS, "w", encoding="utf-8", newline="\n").write(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

print("更新 %d 库的 sr/channels：" % len(changed))
all_sr = Counter()
for ds, old, new, probed, fails in changed:
    print("  %-19s %-28s -> %-46s probed=%d fails=%d" % (ds, json.dumps(old), json.dumps(new), probed, fails))
    for k, v in new.items():
        all_sr[int(k)] += v
print("\n全库采样率分布（按文件数）：")
tot = sum(all_sr.values())
for k in sorted(all_sr):
    print("  %6d Hz : %6d 文件 (%.1f%%)" % (k, all_sr[k], 100.0 * all_sr[k] / tot))
print("  合计 %d 文件" % tot)
