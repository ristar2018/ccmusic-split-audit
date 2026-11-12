# -*- coding: utf-8 -*-
"""Write the re-checked values from data/sample_rates.json back into sr / channels in data/actual_stats.json.

Only these two fields change (every original value was {"0": n}, left over from soundfile failures); all other fields are untouched.
A backup is written to _archive/actual_stats-2026-11-04-before-sr.json before writing.
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
        "method": "header parsing (wav/RIFF fmt+data, flac/STREAMINFO, mp3 first frame header + Xing); no decoding",
        "source": "data/sample_rates.json",
        "probed": src["probed"],
        "unparsed": src["fails"],
    }
    changed.append((ds, old_sr, row["sr"], src["probed"], src["fails"]))

if missing:
    print("!! audio present but no re-check data:", missing)

if not os.path.exists(BAK):
    shutil.copyfile(STATS, BAK)
    print("backup ->", BAK)

io.open(STATS, "w", encoding="utf-8", newline="\n").write(
    json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

print("updated sr/channels for %d libraries:" % len(changed))
all_sr = Counter()
for ds, old, new, probed, fails in changed:
    print("  %-19s %-28s -> %-46s probed=%d fails=%d" % (ds, json.dumps(old), json.dumps(new), probed, fails))
    for k, v in new.items():
        all_sr[int(k)] += v
print("\nSample-rate distribution over all libraries (by file count):")
tot = sum(all_sr.values())
for k in sorted(all_sr):
    print("  %6d Hz : %6d files (%.1f%%)" % (k, all_sr[k], 100.0 * all_sr[k] / tot))
print("  total %d files" % tot)
