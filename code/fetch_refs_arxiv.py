"""Second pass of the reference build: items Crossref does not carry (conference
papers, JMLR, ISMIR) are resolved through the arXiv API instead.

Same discipline as code/fetch_refs.py: the title we ask for is scored against
what the service returns, and only a hit above MIN_SCORE is written out. The
BibTeX entry is assembled from the returned payload (title, authors, year,
journal-ref if the API reports one), never from memory.

    python code/fetch_refs_arxiv.py

Output: paper/refs_arxiv.bib, data/results/refs_arxiv_audit.json
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
BIB = os.path.join(PROJ, "paper", "refs_arxiv.bib")
AUDIT = os.path.join(PROJ, "data", "results", "refs_arxiv_audit.json")
UA = "nuonuo-ref-audit/1.0"
MIN_SCORE = 0.90
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}

WANTED = [
    "FMA: A Dataset for Music Analysis",
    "The Million Song Dataset",
    "The MTG-Jamendo dataset for automatic music tagging",
    "Unsupervised Domain Adaptation by Backpropagation",
    "Domain-Adversarial Training of Neural Networks",
    "Statistical Comparisons of Classifiers over Multiple Data Sets",
    "ESC: Dataset for Environmental Sound Classification",
    "Datasheets for datasets",
    "Do ImageNet Classifiers Generalize to ImageNet?",
    "MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training",
    "Transfer learning for music classification and regression tasks",
    "Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms",
    "Evaluation of CNN-based Automatic Music Tagging Models",
    "Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders",
    "Improving Reproducibility in Machine Learning Research",
    "Leakage in data mining: Formulation, detection, and avoidance",
]


def norm(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def get(url: str, tries: int = 3) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (i + 1))
    raise RuntimeError(str(last))


def search(title: str):
    q = 'ti:"%s"' % title.replace('"', " ")
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q)
           + "&start=0&max_results=5")
    root = ET.fromstring(get(url))
    entries = []
    for e in root.findall("a:entry", NS):
        entries.append({
            "title": " ".join((e.findtext("a:title", default="", namespaces=NS)).split()),
            "authors": [a.findtext("a:name", default="", namespaces=NS)
                        for a in e.findall("a:author", NS)],
            "published": e.findtext("a:published", default="", namespaces=NS),
            "id": e.findtext("a:id", default="", namespaces=NS),
            "journal_ref": (e.findtext("x:journal_ref", default="", namespaces=NS) or "").strip(),
            "comment": (e.findtext("x:comment", default="", namespaces=NS) or "").strip(),
        })
    return entries


def key_of(authors: list[str], year: str) -> str:
    surname = authors[0].split()[-1] if authors else "anon"
    surname = re.sub(r"[^A-Za-z]", "", surname).lower()
    return "%s%s" % (surname, year)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out, audit, keys = [], [], set()
    for title in WANTED:
        try:
            hits = search(title)
        except Exception as exc:  # noqa: BLE001
            audit.append({"title": title, "status": "error", "error": str(exc)})
            print("  ERROR  %-66s %s" % (title[:66], exc))
            continue
        want = norm(title)
        best, score = None, 0.0
        for h in hits:
            s = difflib.SequenceMatcher(None, want, norm(h["title"])).ratio()
            if s > score:
                best, score = h, s
        if best is None or score < MIN_SCORE:
            audit.append({"title": title, "status": "rejected", "score": round(score, 3),
                          "got": (best or {}).get("title")})
            print("  REJECT %-66s %.3f  %s" % (title[:66], score, (best or {}).get("title", "")))
            continue

        aid = best["id"].rsplit("/", 1)[-1]
        year = best["published"][:4]
        authors = " and ".join(a.replace(" and ", " ") for a in best["authors"]) or "anonymous"
        key = key_of(best["authors"], year)
        while key in keys:
            key += "x"
        keys.add(key)
        venue = best["journal_ref"] or best["comment"].split(";")[0][:120]
        entry = [
            "@article{%s," % key,
            "  title = {%s}," % best["title"].replace("&", r"\&"),
            "  author = {%s}," % authors,
            "  year = {%s}," % year,
            "  eprint = {%s}," % aid,
            "  archivePrefix = {arXiv},",
            "  doi = {10.48550/arXiv.%s}," % aid,
        ]
        if venue:
            entry.append("  note = {%s}," % venue.replace("&", r"\&"))
        entry.append("}")
        out.append("\n".join(entry))
        audit.append({"title": title, "status": "ok", "score": round(score, 3), "arxiv": aid,
                      "got": best["title"], "year": year, "authors": best["authors"],
                      "journal_ref": best["journal_ref"], "comment": best["comment"], "key": key})
        print("  ok     %-66s %.3f  arXiv:%s (%s)" % (title[:66], score, aid, year))
        time.sleep(3.0)                                  # arXiv asks for ~3 s between calls

    n_ok = sum(1 for r in audit if r["status"] == "ok")
    print("resolved %d / %d" % (n_ok, len(WANTED)))
    if a.dry_run:
        return 0
    os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
    with io.open(AUDIT, "w", encoding="utf-8") as f:
        json.dump({"min_score": MIN_SCORE, "n_wanted": len(WANTED), "n_ok": n_ok, "entries": audit},
                  f, ensure_ascii=False, indent=1)
    with io.open(BIB, "w", encoding="utf-8", newline="\n") as f:
        f.write("% generated by code/fetch_refs_arxiv.py from the arXiv API; do not hand-edit\n")
        f.write("\n\n".join(out) + "\n")
    print("wrote", BIB)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
