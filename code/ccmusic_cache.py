#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature cache: unify the manifest's mel images into a (N,224,224,3) uint8 memmap.

Design basis (these conventions must not be changed in code):
- `worklog/exp-plan/KICKOFF.md` §3.4: five libraries of mel images, uint8 memmap, 224×224; size groups
  496×369 / 476×349 / 417×128 / 5360×128 / 7220×128; interpolation hard-coded and declared in the caption;
  `num_workers=6, persistent_workers=True`; acceptance = 200 sampled images match the originals by eye + CACHE.json.
- `design note v2.8 :575`: each main run uses 0.60S ≈ 66,249 images
  (224×224×3 uint8 ≈ 10.0 GB); the full 110,415 images ≈ 16.6 GB.
- Same file `:587` (risk table): feature-map sizes are not uniform (496×369 dominant, pianos 128-high strips up to 28,235 wide),
  "resize everything to 224, state the interpolation and aspect-ratio handling in the method; no size-cascade comparison".

Interpolation choice in this script (hard-coded; change it here and in the caption):
- `PIL.Image.BILINEAR`. Pillow's resize is scale-aware (filter support widens with the zoom ratio),
  so a downscale to 224 is antialiased; LANCZOS on spectrograms overshoots at high-frequency edges, BILINEAR does not ring.
- Aspect ratio: **not preserved**, stretched straight to 224×224. Reason: source aspect ratios across the five
  libraries run from 1.34 (496×369) to 42 (5360×128), two orders of magnitude; any "keep ratio + padding" would
  add large black borders to the small images and turn the padding fraction itself into a between-library
  confound. Cost, stated in the method: the time axis of the strip images (pianos) is compressed 1/24, so
  the time resolution is lost in the cache.
- Colour space: always `convert("RGB")` (all sources are already RGB; this step is a defensive assertion).

Usage:
  python code/ccmusic_cache.py --libs CTIS,music_genre --workers 8 --verify 200
  python code/ccmusic_cache.py --libs CTIS --no-verify
  python code/ccmusic_cache.py --libs CTIS --verify 200 --force     # rewrite an existing cache

Outputs (default `data/cache/mel224/`):
  <lib>.u8             uint8 memmap, shape (N,224,224,3), row order = manifest row order
  <lib>_index.csv      row,sample_id,label,recording_uid,src_w,src_h,resized
  CACHE.json           per-library count / shape / dtype / source-size histogram / interpolation / aspect policy / elapsed / verify result
  verify_montage.png   written by --verify: 12 pairs "original (scaled proportionally to 112 high) | cache read-back"
"""

import argparse
import collections
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"D:\VSCode\Datasets"
MANIFEST = os.path.join(PROJ, "data", "manifest")
DEFAULT_OUT = os.path.join(PROJ, "data", "cache", "mel224")

SIZE = 224
CHUNK = 512
RESAMPLE = Image.BILINEAR
RESAMPLE_NAME = "PIL BILINEAR (scale-aware, antialiased downscale)"
ASPECT_POLICY = "stretch to square (aspect NOT preserved)"


def load_manifest(lib):
    path = os.path.join(MANIFEST, lib + ".csv")
    if not os.path.exists(path):
        raise SystemExit("manifest not found: " + path)
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_one(rel, size=SIZE):
    """One image: open → RGB → resize → uint8 HWC. Returns (arr, (w, h))."""
    p = os.path.join(DATA, rel.replace("/", os.sep))
    with Image.open(p) as im:
        wh = im.size
        if im.mode != "RGB":
            im = im.convert("RGB")
        if wh != (size, size):
            im = im.resize((size, size), RESAMPLE)
        arr = np.asarray(im, dtype=np.uint8)
    if arr.shape != (size, size, 3):
        raise ValueError("unexpected shape %s for %s" % (arr.shape, rel))
    return arr, wh


def build_lib(lib, rows, outdir, workers, force):
    n = len(rows)
    path = os.path.join(outdir, lib + ".u8")
    idx = os.path.join(outdir, lib + "_index.csv")
    expect_bytes = n * SIZE * SIZE * 3
    if os.path.exists(path) and not force:
        got = os.path.getsize(path)
        if got == expect_bytes and os.path.exists(idx):
            print("[skip] %s cache already complete (%d rows, %.2f GB)" % (lib, n, got / 2 ** 30))
            return None
        raise SystemExit("%s cache is incomplete (%d/%d bytes); pass --force to rewrite" % (lib, got, expect_bytes))

    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(n, SIZE, SIZE, 3))
    sizes = collections.Counter()
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(idx, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "sample_id", "label", "recording_uid", "src_w", "src_h", "resized"])
        for start in range(0, n, CHUNK):
            block = rows[start:start + CHUNK]
            res = list(ex.map(lambda r: read_one(r["relative_path"]), block))
            for k, ((arr, wh), r) in enumerate(zip(res, block)):
                i = start + k
                mm[i] = arr
                sizes[wh] += 1
                w.writerow([i, r["sample_id"], r["label"], r["recording_uid"],
                            wh[0], wh[1], int(wh != (SIZE, SIZE))])
            done += len(block)
            if done % (CHUNK * 8) == 0 or done == n:
                el = time.time() - t0
                print("  %s %d/%d  %.1f img/s  %.2f GB  elapsed %.0fs"
                      % (lib, done, n, done / max(el, 1e-9), done * SIZE * SIZE * 3 / 2 ** 30, el))
    mm.flush()
    del mm
    got = os.path.getsize(path)
    if got != expect_bytes:
        raise SystemExit("%s byte count mismatch after write: %d != %d" % (lib, got, expect_bytes))
    return {"rows": n, "bytes": got, "gb": round(got / 2 ** 30, 3),
            "src_sizes": {"%dx%d" % k: v for k, v in sizes.most_common()},
            "n_resized": sum(v for k, v in sizes.items() if k != (SIZE, SIZE)),
            "elapsed_s": round(time.time() - t0, 1),
            "img_per_s": round(n / max(time.time() - t0, 1e-9), 1)}


def verify_lib(lib, rows, outdir, k=200, seed=20261105):
    """Sample k rows: cache read-back vs recomputed from source, byte by byte; returns difference statistics."""
    path = os.path.join(outdir, lib + ".u8")
    mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(len(rows), SIZE, SIZE, 3))
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(rows)), min(k, len(rows))))
    diffs = []
    picks = []
    for i in idxs:
        ref, _ = read_one(rows[i]["relative_path"])
        got = np.asarray(mm[i])
        d = int(np.abs(got.astype(np.int16) - ref.astype(np.int16)).max())
        diffs.append(d)
        if len(picks) < 12:
            picks.append((i, ref, got))
    del mm
    bad = [d for d in diffs if d != 0]
    return {"n": len(idxs), "max_abs_diff": max(diffs), "n_nonzero": len(bad),
            "sample_rows": idxs[:5], "picks": picks}


def montage(picks, path):
    """12 pairs: left = original scaled proportionally to 112 high, right = cache read-back at 112 high."""
    h = 112
    tiles = []
    for i, ref, got in picks:
        a = Image.fromarray(ref).resize((int(round(SIZE * h / SIZE)), h), Image.BILINEAR)
        b = Image.fromarray(np.asarray(got)).resize((a.size[0], h), Image.BILINEAR)
        sep = Image.new("RGB", (3, h), (255, 0, 0))
        row = Image.new("RGB", (a.size[0] * 2 + 3, h), (255, 255, 255))
        row.paste(a, (0, 0))
        row.paste(sep, (a.size[0], 0))
        row.paste(b, (a.size[0] + 3, 0))
        tiles.append(row)
    cols = 3
    rows_n = (len(tiles) + cols - 1) // cols
    tw = max(t.size[0] for t in tiles)
    sheet = Image.new("RGB", (tw * cols + 2 * (cols - 1), h * rows_n + 2 * (rows_n - 1)), (255, 255, 255))
    for n_, t in enumerate(tiles):
        r, c = divmod(n_, cols)
        sheet.paste(t, (c * (tw + 2), r * (h + 2)))
    sheet.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", default="CTIS,music_genre")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", type=int, default=200)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    libs = [x.strip() for x in a.libs.split(",") if x.strip()]
    os.makedirs(a.out, exist_ok=True)

    cache = {"generated_by": "code/ccmusic_cache.py",
             "design_ref": ["worklog/exp-plan/KICKOFF.md §3.4",
                            "design note v2.8 :575", "same file :587"],
             "size": SIZE, "dtype": "uint8", "channels": 3,
             "resample": RESAMPLE_NAME, "aspect_policy": ASPECT_POLICY,
             "row_order": "manifest row order (the row column is the manifest row index)",
             "data_root": DATA, "out": a.out, "libraries": {}}
    # On incremental library additions the cached entries must survive: CACHE.json is the whole summary (total_rows / total_gb accumulate from it)
    prev_path = os.path.join(a.out, "CACHE.json")
    if os.path.exists(prev_path):
        try:
            with open(prev_path, encoding="utf-8") as f:
                prev = json.load(f)
            cache["libraries"] = dict(prev.get("libraries", {}))
            print("merging existing CACHE.json: %d libraries (%d rows)"
                  % (len(cache["libraries"]), sum(v.get("rows", 0) for v in cache["libraries"].values())))
        except Exception as e:      # noqa: BLE001  rebuild when the old format cannot be read; do not block
            print("!! cannot read the old CACHE.json (%s); building from scratch" % e)

    for lib in libs:
        rows = load_manifest(lib)
        print("[%s] manifest: %d rows" % (lib, len(rows)))
        info = build_lib(lib, rows, a.out, a.workers, a.force)
        if info is None:
            # On skip we still have to restore the volume and source sizes, else total_gb only counts the libraries built now
            p = os.path.join(a.out, lib + ".u8")
            nb = os.path.getsize(p)
            info = {"rows": len(rows), "bytes": nb, "gb": round(nb / 2 ** 30, 3),
                    "cached": True, "skipped_rebuild": True}
            with open(os.path.join(a.out, "CACHE.json"), encoding="utf-8") as f:
                prev = json.load(f)
            old = prev["libraries"].get(lib, {})
            for k in ("src_sizes", "n_resized"):
                if k in old:
                    info[k] = old[k]
            if "src_sizes" not in info:
                # the old CACHE.json did not record source sizes: rebuild them from src_w/src_h in index.csv
                sizes = collections.Counter()
                nres = 0
                with open(os.path.join(a.out, lib + "_index.csv"), encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        sizes["%sx%s" % (r["src_w"], r["src_h"])] += 1
                        if str(r.get("resized")).lower() in ("true", "1"):
                            nres += 1
                info["src_sizes"] = dict(sizes)
                info["n_resized"] = nres
        cache["libraries"][lib] = info
        if not a.no_verify and a.verify > 0:
            v = verify_lib(lib, rows, a.out, k=a.verify)
            mp = montage(v.pop("picks"), os.path.join(a.out, "verify_montage_%s.png" % lib))
            v["montage"] = os.path.basename(mp)
            cache["libraries"][lib]["verify"] = v
            print("[%s] sampled %d: max_abs_diff=%d, %d mismatched -> %s"
                  % (lib, v["n"], v["max_abs_diff"], v["n_nonzero"], v["montage"]))
            if v["n_nonzero"]:
                raise SystemExit("cache does not match the source images; stopping")

    cache["total_rows"] = sum(v.get("rows", 0) for v in cache["libraries"].values())
    cache["total_gb"] = round(sum(v.get("gb", 0) for v in cache["libraries"].values()), 3)
    cache["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(a.out, "CACHE.json"), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print("→ %s" % os.path.join(a.out, "CACHE.json"))
    print("total %d rows / %.3f GB" % (cache["total_rows"], cache["total_gb"]))


if __name__ == "__main__":
    main()
