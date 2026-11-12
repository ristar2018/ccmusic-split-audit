#!/usr/bin/env python
"""Batch download / verify of the 13 CCMusic (ModelScope ccmusic-database) datasets.

Layout: <dest>/<dataset>/<path inside the repo>; cache pinned to <dest>/.cache (never touches C:).
Engines: modelscope CLI (default) -> in-process modelscope_hub -> direct HTTP (auto degrades in that order).

Presets:
  labels  labels/metadata only, for all 13 datasets (csv, label.zip, dataset_infos.json; 26 files)
  audio   the 7 datasets that need audio, whole repo (pianos / music_genre / chest_falsetto /
          acapella / song_structure / bel_canto / CTIS, about 38 GB)
  all     labels + audio

Usage:
  python code/ccmusic_download.py --project labels --dry-run
  python code/ccmusic_download.py --project labels --dest D:/VSCode/Datasets
  python code/ccmusic_download.py --project audio --via http --dataset CTIS
  python code/ccmusic_download.py --dataset instrument_timbre --file data/Chinese.csv
  python code/ccmusic_download.py --project labels --check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST_DEFAULT = Path(r"D:\VSCode\Datasets")
ENDPOINT = "https://www.modelscope.cn"
ORG = "ccmusic-database"
INDEX = Path(__file__).with_name("dataset_index.json")

PRESETS = {
    "labels": {"scope": "labels", "datasets": None},   # None = all 13 in the index
    "audio": {"scope": "full", "datasets": ["pianos", "music_genre", "chest_falsetto",
                                            "acapella", "song_structure", "bel_canto", "CTIS"]},
}


def load_index():
    if not INDEX.exists():
        sys.exit("index missing: %s (run code/build_index.py to generate it)" % INDEX)
    return json.loads(INDEX.read_text(encoding="utf-8"))["datasets"]


def plan(project, datasets, files, index):
    """Returns [{ds, cmd, check}]: cmd=None means whole repo (no --include)."""
    groups = ["labels", "audio"] if project == "all" else [project]
    jobs, seen = [], {}
    for g in groups:
        spec = PRESETS[g]
        names = spec["datasets"] or list(index)
        if datasets:
            names = [d for d in names if d in datasets]
        for ds in names:
            if ds not in index:
                sys.exit("unknown dataset %s; available: %s" % (ds, ", ".join(index)))
            if files:
                bad = [f for f in files if f not in index[ds]["files"]]
                if bad:
                    sys.exit("%s has no such file: %s" % (ds, bad))
                cmd, chk = list(files), list(files)
            elif spec["scope"] == "full":
                cmd, chk = None, index[ds]["files"]
            else:
                cmd = chk = index[ds]["labels"]
            if ds in seen:                      # all: whole-repo wins, avoid duplicates
                if cmd is None:
                    seen[ds].update(cmd=None, check=chk)
                continue
            seen[ds] = {"ds": ds, "cmd": cmd, "check": chk}
            jobs.append(seen[ds])
    return jobs


def cli_cmd(ds, dest, files):
    cmd = ["modelscope", "download", "--dataset", "%s/%s" % (ORG, ds),
           "--endpoint", ENDPOINT,
           "--cache-dir", str(dest / ".cache"),
           "--local-dir", str(dest / ds)]
    if files:
        cmd += ["--include", *files]   # must come last: nargs='+', mid-list swallows later options
    return cmd


def run_modelscope(ds, dest, files, dry):
    cmd = cli_cmd(ds, dest, files)
    print("  $ " + subprocess.list2cmdline(cmd))
    if dry:
        return True
    p = subprocess.run(cmd, timeout=7200, cwd=str(ROOT))
    if p.returncode != 0:
        print("  [modelscope] exit code %s" % p.returncode)
        return False
    return True


def run_python(ds, dest, files, dry):
    print("  [python] in-process modelscope_hub download_repo(%s, %s)" % (ds, files or "whole repo"))
    if dry:
        return True
    import modelscope_hub
    from modelscope_hub._download import DownloadManager
    cfg = modelscope_hub.get_default_config()
    cfg.endpoint = ENDPOINT
    client = modelscope_hub._legacy_api.LegacyClient(token=None, endpoint=ENDPOINT)
    DownloadManager(client, cfg).download_repo(
        repo_id="%s/%s" % (ORG, ds), repo_type="dataset", revision="master",
        cache_dir=str(dest / ".cache"), local_dir=str(dest / ds),
        allow_patterns=files)
    return True


def run_http(ds, dest, files, dry):
    if not files:
        print("  [http] whole-repo mode not supported; run --project labels first to fetch metadata")
        return False
    ok = True
    for f in files:
        url = ("%s/api/v1/datasets/%s/%s/repo?Revision=master&FilePath=%s"
               % (ENDPOINT, ORG, ds, urllib.parse.quote(f)))
        out = dest / ds / f
        print("  [http] %s" % f)
        if dry:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(out, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as e:
            print("  [http] failed %s: %s: %s" % (f, type(e).__name__, e))
            ok = False
    return ok


ENGINES = {"modelscope": run_modelscope, "python": run_python, "http": run_http}


def download(job, dest, via):
    ds = job["ds"]
    order = ["modelscope", "python", "http"] if via == "auto" else [via]
    for eng in order:
        try:
            if ENGINES[eng](ds, dest, job["cmd"], False):
                return eng
            print("  -> %s failed, degrading" % eng)
        except Exception as e:
            print("  -> %s raised %s: %s, degrading" % (eng, type(e).__name__, e))
    return None


def verify(dest, jobs, index, deep=False):
    total, problems = 0, []
    for job in jobs:
        ds = job["ds"]
        for f in job["check"]:
            p = dest / ds / f
            total += 1
            if not p.exists():
                problems.append("%s/%s missing" % (ds, f))
                continue
            local = p.stat().st_size
            if local == 0:
                problems.append("%s/%s 0 bytes" % (ds, f))
                continue
            want = index[ds]["sizes"].get(f)
            if want and want != local:
                problems.append("%s/%s size mismatch local=%d remote=%d" % (ds, f, local, want))
    print("%d files, %d problems" % (total, len(problems)))
    for x in problems:
        print("  -", x)
    return total, problems


def write_manifest(dest, jobs, via, index):
    mf = dest / "_manifest.json"
    m = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"_meta": {}}
    for job in jobs:
        for f in job["check"]:
            p = dest / job["ds"] / f
            m["%s/%s" % (job["ds"], f)] = {"size": p.stat().st_size if p.exists() else 0}
    m["_meta"] = {"dest": str(dest), "via": via, "n_files": len(m) - 1,
                  "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    mf.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print("manifest: %s  n_files=%d  via=%s" % (mf, m["_meta"]["n_files"], via))


def main(argv=None):
    ap = argparse.ArgumentParser(description="CCMusic 13 datasets: download / verify (labels / audio / all)")
    ap.add_argument("--project", default="labels", choices=["labels", "audio", "all"])
    ap.add_argument("--dataset", action="append", default=None, help="restrict to these datasets (repeatable / comma-separated)")
    ap.add_argument("--file", action="append", default=None, help="restrict to paths inside the repo (use with --dataset)")
    ap.add_argument("--dest", default=str(DEST_DEFAULT))
    ap.add_argument("--via", default="auto", choices=["auto", "modelscope", "python", "http"])
    ap.add_argument("--dry-run", action="store_true", help="print the commands only, download nothing")
    ap.add_argument("--check", action="store_true", help="verify local files only (compare against remote sizes)")
    a = ap.parse_args(argv)

    dest = Path(a.dest)
    datasets = [d.strip() for x in (a.dataset or []) for d in x.split(",") if d.strip()]
    files = [f.strip() for x in (a.file or []) for f in x.split(",") if f.strip()]
    if files and not datasets:
        sys.exit("--file also requires --dataset")

    index = load_index()
    jobs = plan(a.project, datasets, files, index)
    n = sum(len(j["check"]) for j in jobs)
    print("target dir: %s  preset: %s  engine: %s" % (dest, a.project, a.via))
    print("plan: %d datasets / %d files" % (len(jobs), n))

    if a.check:
        _, problems = verify(dest, jobs, index)
        return 0 if not problems else 1

    if a.dry_run:
        for j in jobs:
            print("\n[%s] %d files" % (j["ds"], len(j["check"])))
            if j["cmd"]:
                shown = j["cmd"] if len(j["cmd"]) <= 3 else j["cmd"][:2] + ["... %d more" % (len(j["cmd"]) - 2)]
                for f in shown:
                    print("   ", f)
            else:
                print("    whole repo (no --include)")
            print("  $ " + subprocess.list2cmdline(cli_cmd(j["ds"], dest, j["cmd"])))
        print("\n[dry-run] nothing downloaded")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    used = []
    for j in jobs:
        print("\n==> %s (%d files)" % (j["ds"], len(j["check"])))
        eng = download(j, dest, a.via)
        used.append(eng)
        got = sum(1 for f in j["check"] if (dest / j["ds"] / f).exists()
                  and (dest / j["ds"] / f).stat().st_size > 0)
        print("  -> %s: %d/%d" % (eng, got, len(j["check"])))
        if got == 0:
            print("  !! repos that only return WARNING still exit 0; fetch them explicitly with --via http: %s" % j["ds"])
    via = a.via if a.via != "auto" else (used[0] if used and len(set(used)) == 1 else "mixed")
    write_manifest(dest, jobs, via, index)
    print()
    _, problems = verify(dest, jobs, index)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
