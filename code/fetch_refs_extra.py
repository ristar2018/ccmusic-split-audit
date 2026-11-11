"""Third pass of the reference build: entries neither Crossref search nor the
arXiv API returned, plus one whose Crossref *search* hit was the wrong version
(a bioRxiv preprint instead of the journal article).

Two mechanisms, both evidence-first:
  * explicit DOIs -> Crossref's own BibTeX transform endpoint (metadata is
    Crossref's, not ours);
  * OpenAlex -> api.openalex.org/works?search=..., used for ISMIR/JMLR/ACM MM
    items that have no Crossref record. The title is scored against the query
    and only hits above MIN_SCORE are written; venue, year and DOI (when one
    exists) come from the returned record.

DBLP was tried first and rejected: dblp.org serves an anti-bot HTML page
("Making sure you're not a bot!") to scripted clients, so its JSON API is not
reachable from here.

    python code/fetch_refs_extra.py

Output: paper/refs_extra.bib, data/results/refs_extra_audit.json
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
BIB = os.path.join(PROJ, "paper", "refs_extra.bib")
AUDIT = os.path.join(PROJ, "data", "results", "refs_extra_audit.json")
UA = "nuonuo-ref-audit/1.0 (mailto:noreply@example.org)"
MIN_SCORE = 0.90

BY_DOI = [
    ("10.3389/fnins.2024.1373515", "Data leakage in deep learning studies of translational EEG"),
]

BY_TITLE = [
    ("The Million Song Dataset", "Bertin-Mahieux", 2011),
    ("The MTG-Jamendo dataset for automatic music tagging", "Bogdanov", 2019),
    ("Statistical Comparisons of Classifiers over Multiple Data Sets", "Demsar", 2006),
    ("ESC: Dataset for Environmental Sound Classification", "Piczak", 2015),
    ("Improving Reproducibility in Machine Learning Research", "Pineau", 2021),
    ("Leakage in data mining: Formulation, detection, and avoidance", "Kaufman", 2012),
    ("Automatic Tagging Using Deep Convolutional Neural Networks", "Choi", 2016),
]


def norm(t: str) -> str:
    t = re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())
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
            time.sleep(3.0 * (i + 1))
    raise RuntimeError(str(last))


def crossref_bibtex(doi: str) -> str:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi) + "/transform/application/x-bibtex"
    return get(url).decode("utf-8").strip()


def openalex_search(title: str) -> list[dict]:
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(title)
           + "&per_page=8&mailto=noreply@example.org")
    msg = json.loads(get(url))
    return msg.get("results", [])


def key_of(authors: list[str], year: str) -> str:
    surname = authors[0].split()[-1] if authors else "anon"
    surname = re.sub(r"[^A-Za-z]", "", surname).lower()
    return "%s%s" % (surname, year)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out, audit = [], []

    for doi, title in BY_DOI:
        try:
            out.append(crossref_bibtex(doi))
            audit.append({"title": title, "status": "ok", "via": "crossref-doi", "doi": doi})
            print("  ok     %-66s doi %s" % (title[:66], doi))
        except Exception as exc:  # noqa: BLE001
            audit.append({"title": title, "status": "error", "via": "crossref-doi", "error": str(exc)})
            print("  ERROR  %-66s %s" % (title[:66], exc))

    for title, first_author, year in BY_TITLE:
        try:
            hits = openalex_search(title)
        except Exception as exc:  # noqa: BLE001
            audit.append({"title": title, "status": "error", "via": "openalex", "error": str(exc)})
            print("  ERROR  %-66s %s" % (title[:66], exc))
            continue
        want = norm(title)
        best, score = None, 0.0
        for h in hits:
            s = difflib.SequenceMatcher(None, want, norm(h.get("title"))).ratio()
            names = [au.get("author", {}).get("display_name", "") for au in h.get("authorships", [])]
            if names and first_author.lower() in names[0].lower():
                s += 0.03
            if str(h.get("publication_year")) == str(year):
                s += 0.03
            if s > score:
                best, score = h, s
        if best is None or score < MIN_SCORE:
            audit.append({"title": title, "status": "rejected", "via": "openalex",
                          "score": round(score, 3), "got": (best or {}).get("title")})
            print("  REJECT %-66s %.3f  %s" % (title[:66], score, (best or {}).get("title", "")))
            continue

        names = [au.get("author", {}).get("display_name", "") for au in best.get("authorships", [])]
        yr = str(best.get("publication_year") or year)
        venue = ""
        loc = best.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = src.get("display_name") or ""
        doi = (best.get("doi") or "").replace("https://doi.org/", "")
        key = key_of(names, yr)

        entry = ["@inproceedings{%s," % key]
        entry.append("  title = {%s}," % (best.get("title") or title).replace("&", r"\&"))
        entry.append("  author = {%s}," % " and ".join(names).replace("&", r"\&"))
        entry.append("  year = {%s}," % yr)
        if venue:
            entry.append("  booktitle = {%s}," % venue.replace("&", r"\&"))
        if doi:
            entry.append("  doi = {%s}," % doi)
        entry.append("}")
        out.append("\n".join(entry))
        audit.append({"title": title, "status": "ok", "via": "openalex", "score": round(score, 3),
                      "got": best.get("title"), "authors": names, "year": yr, "venue": venue,
                      "doi": doi, "openalex": best.get("id"), "key": key})
        print("  ok     %-66s %.3f  %s | %s" % (title[:66], score, yr, venue[:44]))
        time.sleep(1.0)

    n_ok = sum(1 for r in audit if r["status"] == "ok")
    print("resolved %d / %d" % (n_ok, len(BY_DOI) + len(BY_TITLE)))
    if a.dry_run:
        return 0
    os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
    with io.open(AUDIT, "w", encoding="utf-8") as f:
        json.dump({"min_score": MIN_SCORE, "n_ok": n_ok, "entries": audit}, f, ensure_ascii=False, indent=1)
    with io.open(BIB, "w", encoding="utf-8", newline="\n") as f:
        f.write("% generated by code/fetch_refs_extra.py from Crossref (by DOI) and OpenAlex; do not hand-edit\n")
        f.write("\n\n".join(out) + "\n")
    print("wrote", BIB)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
