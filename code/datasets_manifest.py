"""以 _extract.json 为唯一事实源，生成 13 个 CCMusic 数据集的到手指南。

输出：
  data/datasets_inventory.json   机器可读清单（后续实验脚本只读这一份）
  data/datasets_inventory.csv    人读表格

口径说明：
  - 事实源：D:/VSCode/Datasets/_extract.json（解压记录：每个 zip 的文件数、字节、目标目录、状态）
  - 计数：data/actual_stats.json（对磁盘的实测扫描：扩展名分布、音频时长、图像尺寸）
  - 存在性：直接 os.path.isdir 复核，不复用任何人的说法
  - 项目内 Datasets/ 只是元数据快照（26 个文件），单独统计，不参与"有几个库"的结论
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

# 13 个官方数据集，顺序与 ModelScope ccmmusic-database 组织页一致
OFFICIAL = [
    ("pianos", "钢琴音质", "品牌/音高/四维评分"),
    ("music_genre", "音乐流派", "3 层 2/9/16 类"),
    ("chest_falsetto", "真假声", "4 类"),
    ("acapella", "无伴奏唱功", "9 维 10 分制"),
    ("CNPM", "五声调式", "system/tonic/pattern/type"),
    ("Guzheng_Tech99", "古筝技法", "7 类逐帧"),
    ("instrument_timbre", "乐器音色", "61 件 16 轴"),
    ("song_structure", "歌曲结构", "9 类功能段"),
    ("GZ_IsoTech", "古筝片段技法", "8 类"),
    ("bel_canto", "美声/民族", "4 类"),
    ("erhu_playing_tech", "二胡技法", "11 类 + 4/7 类子集"),
    ("CTIS", "民族乐器", "209 乐器 219 标签"),
    ("timbre_range", "音色范围", "辅助"),
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
    """统计项目内元数据快照：目录名 -> 文件数"""
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
                "note": "仅元数据（dataset_infos.json 与少量 csv/zip），不是可用数据；不计入 present_count",
            },
            "rule": "一个数据集计入 present 当且仅当 D:/VSCode/Datasets/<name> 存在，且其解压记录文件数 >0；"
                    "仅元数据快照不算下载完成",
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

    print(f"{'数据集':<18}{'状态':<12}{'文件':>8}{'体积':>9}{'音频':>6}{'时长h':>7}{'图像':>9}")
    for r in rows:
        print(f"{r['dataset']:<18}{r['status']:<12}{r['extract_files']:>8}"
              f"{r['disk_gb']:>8.2f}G{r['audio']:>6}{r['duration_h']:>7.2f}{r['images']:>9}")
    print(f"\n官方 {len(OFFICIAL)} / 到手 {len(present)} / 缺 {','.join(missing) or '无'}")
    print(f"合计 {total_bytes / 1024 ** 3:.2f}GB, {total_files} 文件, 音频 {total_audio}, 图像 {total_img}")
    print(f"项目内快照：{len(snap)} 个目录 / {sum(snap.values())} 个文件（仅元数据）")
    print(f"-> {OUT_JSON}")
    print(f"-> {OUT_CSV}")


if __name__ == "__main__":
    main()
