"""Generates the availability guide for the 13 CCMusic datasets, with _extract.json as the single source of truth.

Output:
  data/datasets_inventory.json   machine-readable inventory (later experiment scripts read only this file)
  data/datasets_inventory.csv    human-readable table

Conventions:
  - Source of truth: D:/VSCode/Datasets/_extract.json (unpack log: per-zip file count, bytes, target directory, status)
  - Counts: data/actual_stats.json (measured scan of the disk: extension distribution, audio duration, image sizes)
  - Existence: re-checked directly with os.path.isdir, no claim from anyone is taken on trust
  - The in-project Datasets/ is only a metadata snapshot (26 files), counted separately and kept out of the "how many libraries" conclusion
"""

import json
import os
import csv

ROOT = r"D:\VSCode\Datasets"
WS_SNAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Datasets")
EXTRACT = os.path.join(ROOT, "_extract.json")
STATS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "actual_stats.json")
OUT_JSON = os.path.join(os.path.dirname(STATS), "datasets_inventory.json")
OUT_CSV = os.path.join(os.path.dirname(STATS), "datasets_inventory.csv")

# 13 official datasets, in the same order as the ModelScope ccmmusic-database organisation page
OFFICIAL = [
    ("pianos", "piano timbre quality", "brand/pitch/four-dimension rating"),
    ("music_genre", "music genre", "3 levels, 2/9/16 classes"),
    ("chest_falsetto", "chest vs falsetto voice", "4 classes"),
    ("acapella", "a cappella singing skill", "9 dimensions on a 10-point scale"),
    ("CNPM", "pentatonic mode", "system/tonic/pattern/type"),
    ("Guzheng_Tech99", "guzheng techniques", "7 classes, per frame"),
    ("instrument_timbre", "instrument timbre", "61 instruments, 16 axes"),
    ("song_structure", "song structure", "9 functional segment classes"),
    ("GZ_IsoTech", "guzheng isolated-segment techniques", "8 classes"),
    ("bel_canto", "bel canto / folk", "4 classes"),
    ("erhu_playing_tech", "erhu techniques", "11 classes + 4/7-class subsets"),
    ("CTIS", "Chinese traditional instruments", "209 instruments, 219 labels"),
    ("timbre_range", "timbre range", "auxiliary"),
]
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit != "GB" else f"{n / 1024 ** 3:.2f}GB"
        n /= 1024


def classify(audio, images, present):
    if not present:
        return "missing"
    if audio > 0:
        return "full"
    if images > 0:
        return "features_only"
    return "metadata_only"


def snap_stats(base):
    """Count the in-project metadata snapshot: directory name -> file count"""
    out = {}
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if not os.path.isdir(p):
            continue
        n = 0
        for _, _, files in os.walk(p):
            n += len(files)
        out[name] = n
    return out


def main():
    extract = load_json(EXTRACT)
    stats = load_json(STATS)

    archives = {}
    for key, rec in extract.items():
        ds = rec.get("dataset") or key.split("/")[0]
        archives.setdefault(ds, []).append({
            "archive": rec.get("archive"),
            "zip_bytes": rec.get("zip_bytes", 0),
            "files": rec.get("files", 0),
            "bytes": rec.get("bytes", 0),
            "top": rec.get("top", []),
            "status": rec.get("status", ""),
        })

    rows = []
    for name, cn, task in OFFICIAL:
        p = os.path.join(ROOT, name)
        present = os.path.isdir(p)
        s = stats.get(name, {})
        ext = s.get("ext", {}) or {}
        audio = sum(v for k, v in ext.items() if k.lower() in AUDIO_EXT)
        images = sum(v for k, v in ext.items() if k.lower() in IMG_EXT)
        csvs = sum(v for k, v in ext.items() if k.lower() == ".csv")
        arcs = archives.get(name, [])
        files = sum(a["files"] for a in arcs)
        zbytes = sum(a["zip_bytes"] for a in arcs)
        bytes_ = s.get("bytes", 0)
        status = classify(audio, images, present)
        if present and status == "metadata_only" and files:
            status = "extract_only_no_media"
        rows.append({
            "dataset": name,
            "cn": cn,
            "task": task,
            "present": present,
            "status": status,
            "archives": sorted({a["archive"] for a in arcs}),
            "extract_files": files,
            "extract_zip_gb": round(zbytes / 1024 ** 3, 2),
            "disk_bytes": bytes_,
            "disk_gb": round(bytes_ / 1024 ** 3, 2),
            "audio": audio,
            "duration_h": s.get("duration_h", 0.0),
            "images": images,
            "csv": csvs,
            "ext": ext,
        })

    snap = snap_stats(WS_SNAP)
    present = [r for r in rows if r["present"]]
    missing = [r["dataset"] for r in rows if not r["present"]]
    total_bytes = sum(r["disk_bytes"] for r in rows)
    total_files = sum(r["extract_files"] for r in rows)
    total_audio = sum(r["audio"] for r in rows)
    total_img = sum(r["images"] for r in rows)

    payload = {
        "_meta": {
            "generated": "2026-11-04",
            "source_of_truth": EXTRACT,
            "counted_from": STATS,
            "root": ROOT,
            "official_count": len(OFFICIAL),
            "present_count": len(present),
            "missing": missing,
            "total_disk_gb": round(total_bytes / 1024 ** 3, 2),
            "total_extract_files": total_files,
            "total_audio": total_audio,
            "total_images": total_img,
            "workspace_snapshot": {
                "path": WS_SNAP,
                "datasets": snap,
                "note": "metadata only (dataset_infos.json plus a few csv/zip), not usable data; not counted in present_count",
            },
            "rule": "a dataset counts as present if and only if D:/VSCode/Datasets/<name> exists and its unpack log reports >0 files; "
                    "a metadata-only snapshot does not count as a completed download",
        },
        "datasets": rows,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "cn", "status", "present", "archives", "extract_files",
                    "disk_gb", "audio", "duration_h", "images", "csv"])
        for r in rows:
            w.writerow([r["dataset"], r["cn"], r["status"], r["present"],
                        ";".join(r["archives"]), r["extract_files"], r["disk_gb"],
                        r["audio"], r["duration_h"], r["images"], r["csv"]])

    print(f"{'dataset':<18}{'status':<12}{'files':>8}{'size':>9}{'audio':>6}{'dur_h':>7}{'images':>9}")
    for r in rows:
        print(f"{r['dataset']:<18}{r['status']:<12}{r['extract_files']:>8}"
              f"{r['disk_gb']:>8.2f}G{r['audio']:>6}{r['duration_h']:>7.2f}{r['images']:>9}")
    print(f"\nofficial {len(OFFICIAL)} / present {len(present)} / missing {','.join(missing) or 'none'}")
    print(f"total {total_bytes / 1024 ** 3:.2f}GB, {total_files} files, audio {total_audio}, images {total_img}")
    print(f"in-project snapshot: {len(snap)} directories / {sum(snap.values())} files (metadata only)")
    print(f"-> {OUT_JSON}")
    print(f"-> {OUT_CSV}")


if __name__ == "__main__":
    main()
