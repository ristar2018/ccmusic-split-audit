#!/usr/bin/env python
"""Rebuild code/dataset_index.json from the ModelScope repo tree API (the manifest source for the labels/audio presets)."""
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = "https://www.modelscope.cn"
ORG = "ccmusic-database"
OUT = Path(__file__).with_name("dataset_index.json")
LABEL_DS = ["CNPM", "Guzheng_Tech99", "instrument_timbre",
            "GZ_IsoTech", "erhu_playing_tech", "timbre_range"]
AUDIO_DS = ["pianos", "music_genre", "chest_falsetto", "acapella",
            "song_structure", "bel_canto", "CTIS"]
META = re.compile(r"(^dataset_infos\.json$|^data/[^/]+\.csv$|^data/label\.zip$)")


def tree(ds):
    url = ("%s/api/v1/datasets/%s/%s/repo/tree?Revision=master&Recursive=True"
           % (ENDPOINT, ORG, ds))
    data = json.load(urllib.request.urlopen(url, timeout=60))
    return [(i["Path"], i.get("Size", 0)) for i in data.get("Data", {}).get("Files", [])
            if i.get("Type") != "tree"]


def main():
    idx = {}
    for ds in LABEL_DS + AUDIO_DS:
        pairs = tree(ds)
        files = [p for p, _ in pairs]
        idx[ds] = {"group": "labels" if ds in LABEL_DS else "audio",
                   "labels": sorted(p for p in files if META.match(p)),
                   "files": files,
                   "sizes": {p: s for p, s in pairs}}
        print("%-18s %d files / %d labels" % (ds, len(files), len(idx[ds]["labels"])))
    OUT.write_text(json.dumps({"datasets": idx}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
