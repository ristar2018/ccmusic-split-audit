# -*- coding: utf-8 -*-
"""Assemble the review-facing reproduction package for the CCMusic slice-unit audit.

Copies the frozen artifacts (manuscript sources and PDFs, figures, code, splits,
manifests, results, sealed predictions, review documents) into

    repro/ccmusic_slice_audit_2026-11-11/

and writes RUN_MANIFEST.json (inventory with sha256 per file, environment facts and
the headline numbers with their source files) plus SHA256SUMS.txt.

The package is assembled, never edited: re-running this script rebuilds it from the
project tree, so the project keeps a single copy of every artifact.

Usage (project root):
    python code/assemble_repro.py            # assemble and hash
    python code/assemble_repro.py --zip      # also write repro_<date>.zip
"""
from __future__ import print_function

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
import zipfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = "2026-11-11"
NAME = "ccmusic_slice_audit_" + STAMP
OUT = os.path.join(PROJ, "repro", NAME)

# (project-relative source, package-relative destination)
# the package mirrors the project layout so that every script runs unmodified from
# the package root (PROJ is derived from the script's own location)
TREES = [
    ("paper", "paper"),
    ("figures", "figures"),
    ("code", "code"),
    ("data/splits", "data/splits"),
    ("data/manifest", "data/manifest"),
    ("data/results", "data/results"),
    ("data/p3", "data/p3"),
    ("runs/sealed_eval", "runs/sealed_eval"),
    ("report/审稿意见/human-2026-11-11", "docs/review_human-2026-11-11"),
    ("worklog", "docs/worklog"),
]

SKIP_EXT = {".aux", ".log", ".out", ".toc", ".synctex.gz", ".fls", ".fdb_latexmk",
            ".pyc", ".pyo", ".bak", ".tmp"}


def keep(path):
    low = path.lower()
    if "__pycache__" in low:
        return False
    for ext in SKIP_EXT:
        if low.endswith(ext):
            return False
    return True


def sha256(path):
    h = hashlib.sha256()
    with io.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src_rel, dst_rel, acc):
    src = os.path.join(PROJ, src_rel)
    if not os.path.isdir(src):
        print("  MISSING %s" % src_rel)
        return
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, src)
            if not keep(rel):
                continue
            target = os.path.join(OUT, dst_rel, rel)
            parent = os.path.dirname(target)
            if not os.path.isdir(parent):
                os.makedirs(parent)
            shutil.copy2(full, target)
            acc.append((os.path.join(dst_rel, rel).replace("\\", "/"), full))
            n += 1
    print("  %-40s %5d files -> %s" % (src_rel, n, dst_rel))


def read_small(rel):
    path = os.path.join(PROJ, rel)
    if not os.path.isfile(path):
        return None
    try:
        return json.loads(io.open(path, encoding="utf-8").read())
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="also write a zip archive")
    args = ap.parse_args(argv)

    if os.path.isdir(OUT):
        print("removing previous package at %s" % OUT)
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    print("copying project trees:")
    files = []
    for src_rel, dst_rel in TREES:
        copy_tree(src_rel, dst_rel, files)

    for extra in ("AGENTS.md", "README.md"):
        src = os.path.join(PROJ, extra)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT, "docs", extra))
            files.append(("docs/" + extra, src))

    # the repository's design plan and execution record travel with the package
    repo_docs = os.path.join(PROJ, "_github", "ccmusic-split-audit", "docs")
    for extra in ("design-plan.md", "execution-changes.md"):
        src = os.path.join(repo_docs, extra)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT, "docs", extra))
            files.append(("docs/" + extra, src))

    # the package README is authored once and copied in, so re-assembling never loses it
    readme_src = os.path.join(PROJ, "repro", "README_package.md")
    if os.path.isfile(readme_src):
        shutil.copy2(readme_src, os.path.join(OUT, "README.md"))
        files.append(("README.md", readme_src))

    print("hashing %d files:" % len(files))
    entries = []
    for rel, full in sorted(files):
        entries.append({"path": rel, "bytes": os.path.getsize(full), "sha256": sha256(full)})
    total = sum(e["bytes"] for e in entries)

    with io.open(os.path.join(OUT, "SHA256SUMS.txt"), "w", encoding="utf-8", newline="\n") as fh:
        for e in entries:
            fh.write("%s  %s\n" % (e["sha256"], e["path"]))

    inv = read_small(os.path.join("data", "datasets_inventory.json")) or {}
    rel_inv = (inv.get("datasets") or inv.get("libraries") or [])
    libs = []
    if isinstance(rel_inv, dict):
        rel_inv = list(rel_inv.values())
    for rec in rel_inv if isinstance(rel_inv, list) else []:
        if isinstance(rec, dict):
            libs.append({"dataset": rec.get("dataset"),
                         "audio_files": rec.get("audio"),
                         "images": rec.get("images"),
                         "disk_gb": rec.get("disk_gb"),
                         "task": rec.get("task")})

    manifest = {
        "package": NAME,
        "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "assembled_by": "code/assemble_repro.py",
        "project_root_at_assembly": PROJ,
        "contents": {
            "paper": "LaTeX sources, sn-jnl class and style files, compiled PDFs (30/8/1 pages)",
            "figures": "figure PDFs and PNGs as embedded in the manuscript",
            "code": "every analysis, training, audit, figure and verification script used in the study",
            "data/splits": "frozen split CSVs (per library and side) and the leak audit summary",
            "data/manifest": "per-slice manifests with recording ids, classes, candidate groups and content hashes",
            "data/results": "metric tables, bootstrap summaries, round-2 analyses and the verification report",
            "data/p3": "cross-instrument experiment artifacts and the 43-class universe",
            "runs/sealed_eval": "sealed test-set predictions (probs, labels, recording ids) per library, arm, backbone and seed",
            "docs/review_human-2026-11-11": "second human review report, item-by-item verification, revision record and the audited article",
            "docs/worklog": "project status cards and the append-only work log (internal; drop before shipping to reviewers)",
            "docs/AGENTS.md, docs/README.md": "workspace rules and directory map",
        },
        "layout_note": "the package mirrors the project layout; scripts resolve the project root from their own location, so they run unmodified from the package root",
        "excluded": [
            "raw audio and image features under D:/VSCode/Datasets (35.47 GB; redistribution not permitted, see README)",
            "feature cache (115,847 images) and trained checkpoints (not needed to re-derive any reported number)",
            "LaTeX intermediate files (*.aux, *.log, *.out) and __pycache__",
        ],
        "environment": {
            "python": "3.10.22",
            "torch": "2.11.0+cu128 (cuDNN 9.19)",
            "gpu": "1 x RTX 5060 Ti 16 GB",
            "os": "Windows 11 25H2",
            "latex": "pdflatex + bibtex, sn-jnl sn-basic bst (Numbered)",
        },
        "headline_numbers": [
            {"quantity": "A1-A0 CTIS slice, AlexNet", "value": "+2.02 [+0.83,+3.35]",
             "source": "DATA/results/e1_delta.csv"},
            {"quantity": "A1-A0 CTIS slice, Swin-T", "value": "+1.70 [+1.28,+2.84]",
             "source": "DATA/results/e1_delta.csv"},
            {"quantity": "A1-C1 CTIS slice, AlexNet", "value": "+1.96 [+0.74,+2.99]",
             "source": "DATA/results/human_review/round2/a1_minus_c1.csv"},
            {"quantity": "A1-C1 CTIS slice, Swin-T", "value": "+1.07 [+0.23,+1.96]",
             "source": "DATA/results/human_review/round2/a1_minus_c1.csv"},
            {"quantity": "exposure split, exposed vs not, AlexNet slice",
             "value": "+4.03 [+2.69,+6.14] vs -0.19 [-1.81,+1.83]",
             "source": "DATA/results/human_review/round2/a1_minus_c1.csv"},
            {"quantity": "A2-A0 CTIS slice, AlexNet / Swin-T",
             "value": "+1.21 [-0.02,+2.44] / +1.75 [+1.18,+2.70]",
             "source": "DATA/results/delta_A2/e1_delta.csv"},
            {"quantity": "injected CTIS slices / recordings / test recordings exposed",
             "value": "4,178 / 753 / 578 of 963",
             "source": "DATA/results/human_review/round2/slice_budget_by_class.csv"},
            {"quantity": "reported-number verification", "value": "270 checks passed, 0 failed",
             "source": "DATA/results/paper_verify.json"},
        ],
        "datasets_inventory_snapshot": libs,
        "file_count": len(entries),
        "total_bytes": total,
        "files": entries,
    }
    with io.open(os.path.join(OUT, "RUN_MANIFEST.json"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("package: %s" % OUT)
    print("files: %d   bytes: %d (%.1f MB)" % (len(entries), total, total / 2.0 ** 20))

    if args.zip:
        zpath = os.path.join(PROJ, "repro", NAME + ".zip")
        if os.path.isfile(zpath):
            os.remove(zpath)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for e in entries:
                zf.write(os.path.join(OUT, e["path"]), os.path.join(NAME, e["path"]))
            zf.write(os.path.join(OUT, "RUN_MANIFEST.json"), os.path.join(NAME, "RUN_MANIFEST.json"))
            zf.write(os.path.join(OUT, "SHA256SUMS.txt"), os.path.join(NAME, "SHA256SUMS.txt"))
        print("zip: %s (%.1f MB)" % (zpath, os.path.getsize(zpath) / 2.0 ** 20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
