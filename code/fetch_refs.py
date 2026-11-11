"""Build paper/refs.bib from Crossref only.

Every entry is looked up at api.crossref.org, the top hit is scored against the
title we asked for, and the BibTeX record is pulled from Crossref's own
transform endpoint, so volume, pages, DOI and author list are Crossref's text
rather than ours. Entries whose best hit scores below MIN_SCORE are written to
the audit file but kept out of the .bib.

    python code/fetch_refs.py            # write paper/refs_auto.bib + audit json
    python code/fetch_refs.py --dry-run  # score only, touch no files

Output: paper/refs_auto.bib, data/results/refs_audit.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
BIB = os.path.join(PROJ, "paper", "refs_auto.bib")
AUDIT = os.path.join(PROJ, "data", "results", "refs_audit.json")
UA = "nuonuo-ref-audit/1.0 (mailto:noreply@example.org)"
MIN_SCORE = 0.92

# (title, year). Year is only used to break ties between near-identical hits.
WANTED = [
    ("CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research", 2025),
    ("Leakage and the reproducibility crisis in machine-learning-based science", 2023),
    ("Leakage in data mining: Formulation, detection, and avoidance", 2012),
    ("A Dataset and Taxonomy for Urban Sound Research", 2014),
    ("An analysis of the GTZAN music genre dataset", 2012),
    ("Data leakage in deep learning studies of translational EEG", 2024),
    ("A Critical Study on Data Leakage in Recommender System Offline Evaluation", 2023),
    ("GiantMIDI-Piano: A large-scale MIDI dataset for classical piano music", 2022),
    ("FMA: A Dataset for Music Analysis", 2017),
    ("The Million Song Dataset", 2011),
    ("The MTG-Jamendo dataset for automatic music tagging", 2019),
    ("Unsupervised Domain Adaptation by Backpropagation", 2015),
    ("Domain-Adversarial Training of Neural Networks", 2016),
    ("Statistical Comparisons of Classifiers over Multiple Data Sets", 2006),
    ("Audio Set: An ontology and human-labeled dataset for audio events", 2017),
    ("ESC: Dataset for Environmental Sound Classification", 2015),
    ("Datasheets for datasets", 2021),
    ("Do ImageNet Classifiers Generalize to ImageNet?", 2019),
    ("MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training", 2024),
    ("Natural Language Supervision for General-Purpose Audio Representations", 2024),
    ("Transfer learning for music classification and regression tasks", 2017),
    ("Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms", 2017),
    ("Evaluation of CNN-based Automatic Music Tagging Models", 2020),
    ("Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders", 2017),
    ("The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets", 2015),
    ("A Simple Method to Determine if a Music Information Retrieval System is a Horse", 2014),
    ("CNN Architectures for Large-Scale Audio Classification", 2017),
    ("PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", 2020),
    ("Automatic Tagging Using Deep Convolutional Neural Networks", 2016),
    ("Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification", 2017),
    ("Improving Reproducibility in Machine Learning Research", 2021),
    ("Music Information Retrieval: Recent Developments and Applications", 2014),
    ("The Music Information Retrieval Evaluation eXchange: Some Observations and Insights", 2010),
]


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def get(url: str, tries: int = 3) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - network flake, retried
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(str(last))


def search(title: str) -> list[dict]:
    q = urllib.parse.quote(title)
    url = ("https://api.crossref.org/works?query.bibliographic=" + q
           + "&rows=4&select=DOI,title,issued,container-title,author,type")
    msg = json.loads(get(url))["message"]["items"]
    return msg


def pick(title: str, year: int, hits: list[dict]) -> tuple[dict | None, float]:
    best, best_s = None, 0.0
    want = norm(title)
    for h in hits:
        t = (h.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, want, norm(t)).ratio()
        got_year = (h.get("issued", {}).get("date-parts") or [[0]])[0][0] or 0
        if got_year and year and abs(got_year - year) <= 2:
            s += 0.03                      # small bonus, never overrides a bad title
        if s > best_s:
            best, best_s = h, s
    return best, best_s


def bibtex(doi: str) -> str:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi) + "/transform/application/x-bibtex"
    return get(url).decode("utf-8").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out, audit = [], []
    for title, year in WANTED:
        try:
            hits = search(title)
            hit, score = pick(title, year, hits)
            if hit is None or score < MIN_SCORE:
                audit.append({"title": title, "status": "rejected", "score": round(score, 3),
                              "got": (hit or {}).get("title", [""])[0] if hit else None})
                print("  REJECT %-70s %.3f" % (title[:70], score))
                continue
            doi = hit["DOI"]
            b = bibtex(doi)
            out.append(b)
            audit.append({"title": title, "status": "ok", "score": round(score, 3), "doi": doi,
                          "got": (hit.get("title") or [""])[0],
                          "year": (hit.get("issued", {}).get("date-parts") or [[None]])[0][0],
                          "container": (hit.get("container-title") or [""])[0]})
            print("  ok     %-70s %.3f  %s" % (title[:70], score, doi))
        except Exception as exc:  # noqa: BLE001 - report and continue
            audit.append({"title": title, "status": "error", "error": str(exc)})
            print("  ERROR  %-70s %s" % (title[:70], exc))
        time.sleep(0.4)

    n_ok = sum(1 for r in audit if r["status"] == "ok")
    print("resolved %d / %d (min score %.2f)" % (n_ok, len(WANTED), MIN_SCORE))
    if a.dry_run:
        return 0
    os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
    with io.open(AUDIT, "w", encoding="utf-8") as f:
        json.dump({"min_score": MIN_SCORE, "n_wanted": len(WANTED), "n_ok": n_ok, "entries": audit},
                  f, ensure_ascii=False, indent=1)
    with io.open(BIB, "w", encoding="utf-8", newline="\n") as f:
        f.write("% generated by code/fetch_refs.py from Crossref; do not hand-edit\n")
        f.write("\n\n".join(out) + "\n")
    print("wrote", BIB)
    return 0 if n_ok >= 25 else 1


if __name__ == "__main__":
    sys.exit(main())
