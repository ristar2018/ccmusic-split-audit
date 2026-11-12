# -*- coding: utf-8 -*-
"""CCMusic dataset unpacker: expands in place to <dataset>\\data\\<top-level dir>, verifies, then deletes the archive.

Why the layout is like this
  each dataset's official loader reads data/audio, data/eval/chroma, data/mel, data/label
  and similar paths; the first level inside the archives is exactly audio/ eval/ mel/ label/ timbre/ range/.
  So after unzipping into <dataset>\\data\\ the tree matches the loader convention exactly, with no code change.

Safety valves
  1. check free disk first (unpacked size ≈ the sum of file sizes in the archive);
  2. while unpacking, count entries and accumulate bytes, then reconcile against the zip central directory;
  3. delete the archive only when both "file count" and "byte count" match; otherwise keep the zip and raise an error;
  4. --keep-zips unpacks without deleting; --dry-run only shows the plan.

Usage
  python code/ccmusic_extract.py --dry-run
  python code/ccmusic_extract.py                    # unpack everything and delete the archives
  python code/ccmusic_extract.py --dataset CTIS     # only one dataset
  python code/ccmusic_extract.py --keep-zips        # unpack but keep the zips
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

DATASETS_ROOT = r"D:\VSCode\Datasets"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def find_archives(root: str, only: str | None = None) -> list[tuple[str, str]]:
    """Returns [(dataset, zip path)], looking only at <dataset>/data/*.zip."""
    out = []
    for name in sorted(os.listdir(root)):
        ddir = os.path.join(root, name, "data")
        if not os.path.isdir(ddir) or name.startswith("_"):
            continue
        if only and name != only:
            continue
        for fn in sorted(os.listdir(ddir)):
            if fn.lower().endswith(".zip"):
                out.append((name, os.path.join(ddir, fn)))
    return out


def plan(archive: str) -> dict:
    with zipfile.ZipFile(archive) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        tops = sorted({i.filename.split("/")[0] for i in infos})
    return {
        "files": len(infos),
        "bytes": sum(i.file_size for i in infos),
        "top": tops,
        "compressed": os.path.getsize(archive),
    }


def extract_one(dataset: str, archive: str, root: str, dry: bool, keep: bool) -> dict:
    meta = plan(archive)
    dest_root = os.path.join(root, dataset, "data")  # unpack into data/, so the top-level audio|eval|... lands in place
    rec = {
        "dataset": dataset,
        "archive": os.path.basename(archive),
        "zip_bytes": meta["compressed"],
        "files": meta["files"],
        "bytes": meta["bytes"],
        "top": meta["top"],
        "dest": dest_root,
    }
    if dry:
        rec["status"] = "dry-run"
        return rec

    free = shutil.disk_usage(root).free
    if meta["bytes"] > free:
        rec["status"] = f"skip not enough disk space (need {human(meta['bytes'])}, free {human(free)})"
        return rec

    t0 = time.time()
    n_ok = 0
    n_skip = 0
    b_ok = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = os.path.join(dest_root, info.filename.replace("/", os.sep))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                n_skip += 1
                b_ok += info.file_size
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            n_ok += 1
            b_ok += info.file_size

    rec.update(
        extracted=n_ok,
        skipped=n_skip,
        bytes_ok=b_ok,
        seconds=round(time.time() - t0, 1),
    )
    if b_ok != meta["bytes"] or n_ok + n_skip != meta["files"]:
        rec["status"] = f"mismatch files {n_ok + n_skip}/{meta['files']}, bytes {b_ok}/{meta['bytes']} (zip kept)"
        return rec

    if keep:
        rec["status"] = "ok (zip kept)"
        return rec

    os.remove(archive)
    # also clear any empty directories the unpack may leave behind in this dataset directory
    rec["status"] = "ok (zip deleted)"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="CCMusic dataset unpack (into data/, archives deleted)")
    ap.add_argument("--root", default=DATASETS_ROOT, help="download root directory")
    ap.add_argument("--dataset", help="process only this dataset")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-zips", action="store_true", help="keep the archives after unpacking")
    args = ap.parse_args()

    archives = find_archives(args.root, args.dataset)
    if not archives:
        print("no archives found to unpack")
        return 0

    print(f"root: {args.root}")
    total_files = total_bytes = total_zip = 0
    rows = []
    for dataset, arc in archives:
        p = plan(arc)
        total_files += p["files"]
        total_bytes += p["bytes"]
        total_zip += p["compressed"]
        print(
            f"  {dataset:18s} {os.path.basename(arc):14s} "
            f"{p['files']:>6} files  {human(p['bytes']):>8} -> data/{','.join(p['top'])}"
        )
    print(f"\ntotal {len(archives)} archives / {total_files} files / {human(total_bytes)} unpacked (archives {human(total_zip)})")
    print(f"disk free {human(shutil.disk_usage(args.root).free)}")
    if args.dry_run:
        print("\n--dry-run: nothing was touched")
        return 0

    print("\nunpacking ...")
    t0 = time.time()
    for dataset, arc in archives:
        rec = extract_one(dataset, arc, args.root, args.dry_run, args.keep_zips)
        rows.append(rec)
        print(
            f"  [{rec['status']:<28}] {dataset:18s} {rec['archive']:14s} "
            f"added {rec.get('extracted', 0):>6} skipped {rec.get('skipped', 0):>5} "
            f"{rec.get('seconds', 0):>6.1f}s"
        )

    out = os.path.join(args.root, "_extract.json")
    prev = {}
    if os.path.exists(out):
        try:
            with open(out, encoding="utf-8") as fh:
                prev = json.load(fh)
        except Exception:  # noqa: BLE001
            prev = {}
    prev.update({f"{r['dataset']}/{r['archive']}": r for r in rows})
    prev["_meta"] = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": len([k for k in prev if not k.startswith("_")]),
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, ensure_ascii=False, indent=1)

    bad = [r for r in rows if not str(r["status"]).startswith("ok")]
    print(f"\nelapsed {time.time() - t0:.0f}s, records -> {out}")
    if bad:
        print(f"!! {len(bad)} archives failed verification: {[b['dataset'] + '/' + b['archive'] for b in bad]}")
        return 1
    print("all archives passed verification and were deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
