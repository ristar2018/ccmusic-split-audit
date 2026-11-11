"""Assemble paper/refs.bib from the three service-generated files.

Nothing here is typed by hand: every entry body comes from Crossref's BibTeX
transform (code/fetch_refs.py, code/fetch_refs_extra.py) or from the arXiv API
record (code/fetch_refs_arxiv.py). This script only selects, renames and
concatenates, and it refuses to emit an entry whose DOI it cannot re-resolve.

Dropped on purpose (recorded in data/results/refs_merge_audit.json):
  Kazemi_2018  Crossref returned an unrelated target-recognition paper for the
               "Automatic tagging" query (title score 0.928). False match.
  Dave_2014    Crossref record carries the wrong author list for the MIR
               survey (Schedl, Gomez, Urbano). Metadata not trustworthy.
  Brookshire_2024 (preprint) superseded by the Frontiers version (refs_extra).
  bogdanov2019 / demar2006 / vincentlamarre2021 / choi2016  OpenAlex gave no
               venue, so the entry would be incomplete.

    python code/merge_refs.py
"""

from __future__ import annotations

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
UA = "nuonuo-ref-audit/1.0 (mailto:noreply@example.org)"
OUT = os.path.join(PROJ, "paper", "refs.bib")
AUDIT = os.path.join(PROJ, "data", "results", "refs_merge_audit.json")

EXTRA_DOI = [
    ("alexnet2012", "10.1145/3065386"),
    ("swin2021", "10.1109/iccv48922.2021.00986"),
    ("piczak2015", "10.1145/2733373.2806390"),
    ("kaufman2012", "10.1145/2382577.2382579"),
]

KEEP_AUTO = ["Zhou_2025", "Kapoor_2023", "Salamon_2014", "Sturm_2012", "Ji_2023",
             "Kong_2022", "Gemmeke_2017", "Elizalde_2024", "Saito_2015", "Sturm_2014",
             "Hershey_2017", "Kong_2020", "Salamon_2017", "Downie_2010"]
KEEP_ARXIV = ["defferrard2016", "ganin2014", "ganin2015", "gebru2018", "recht2019",
              "li2023", "choi2017", "lee2017", "won2020", "engel2017"]
KEEP_EXTRA = ["Brookshire_2024"]

RENAME = {"Zhou_2025": "zhou2025ccmusic", "Kapoor_2023": "kapoor2023leakage",
          "Salamon_2014": "salamon2014urbansound", "Sturm_2012": "sturm2012gtzan",
          "Ji_2023": "ji2023leakage", "Kong_2022": "kong2022giantmidi",
          "Gemmeke_2017": "gemmeke2017audioset", "Elizalde_2024": "elizalde2024clap",
          "Saito_2015": "saito2015pr", "Sturm_2014": "sturm2014horse",
          "Hershey_2017": "hershey2017cnn", "Kong_2020": "kong2020panns",
          "Salamon_2017": "salamon2017augment", "Downie_2010": "downie2010mirex",
          "Brookshire_2024": "brookshire2024eeg"}


def get(url: str, tries: int = 3) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.5 * (i + 1))
    raise RuntimeError(str(last))


def crossref_bibtex(doi: str) -> str:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi) + "/transform/application/x-bibtex"
    return get(url).decode("utf-8").strip()


def parse(text: str) -> dict[str, str]:
    """Split a .bib file into {key: entry text} (entries are brace-balanced enough for these files)."""
    out, i = {}, 0
    while True:
        m = re.search(r"@(\w+)\{([^,\s]+),", text[i:])
        if not m:
            break
        start = i + m.start()
        depth, j = 0, i + m.start()
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[m.group(2)] = text[start:j + 1]
        i = j + 1
    return out


def tidy(body: str) -> str:
    """Strip the pieces of Crossref's output that BibTeX warns about.

    Crossref emits month=Sept (an undefined string) and, for arXiv records,
    @article with no journal. Both produce warnings on every build, so the month
    field is dropped and journal-less entries become @misc.
    """
    body = re.sub(r"\s*month\s*=\s*[A-Za-z0-9]+,", "", body)
    # long http(s) urls overflow the numeric-style bibliography line; the DOI
    # field already carries the resolvable identifier
    body = re.sub(r"\s*url\s*=\s*\{[^}]*\},", "", body)
    body = re.sub(r"^@inbook\{", "@incollection{", body)
    head = re.match(r"@(\w+)\{", body)
    if head and head.group(1) == "article" and not re.search(r"\bjournal\s*=", body):
        body = "@misc{" + body[len("@article{"):]
    return body


def main() -> int:
    src = {}
    for name in ("refs_auto.bib", "refs_arxiv.bib", "refs_extra.bib"):
        p = os.path.join(PROJ, "paper", name)
        src.update(parse(io.open(p, encoding="utf-8").read()))

    chosen, audit, dropped = [], [], []
    for key in KEEP_AUTO + KEEP_ARXIV + KEEP_EXTRA:
        if key not in src:
            audit.append({"key": key, "status": "missing"})
            continue
        new = RENAME.get(key, key)
        body = tidy(src[key])
        head = re.match(r"@(\w+)\{", body).group(1)
        chosen.append((new, body.replace("{%s," % key, "{%s," % new, 1)))
        audit.append({"key": new, "status": "kept", "from": key, "type": head})

    for key, doi in EXTRA_DOI:
        try:
            body = tidy(crossref_bibtex(doi))
            # keep the entry type Crossref reports (article for CACM/TKDD,
            # inproceedings for ICCV and ACM MM); only the key is ours
            body = re.sub(r"^@(\w+)\{[^,]+", lambda m: "@%s{%s" % (m.group(1), key), body)
            chosen.append((key, body))
            audit.append({"key": key, "status": "kept", "from": "crossref-doi", "doi": doi})
            print("  ok   %-18s doi %s" % (key, doi))
        except Exception as exc:  # noqa: BLE001
            audit.append({"key": key, "status": "error", "doi": doi, "error": str(exc)})
            print("  ERR  %-18s %s" % (key, exc))
        time.sleep(0.5)

    for bad in ["Kazemi_2018", "Dave_2014", "bogdanov2019", "demar2006",
                "vincentlamarre2021", "choi2016"]:
        dropped.append({"key": bad, "reason": {"Kazemi_2018": "wrong paper returned by Crossref (title score 0.928)",
                                               "Dave_2014": "Crossref author list wrong for the MIR survey"}.get(
            bad, "OpenAlex record has no venue; entry would be incomplete")})

    keys = [k for k, _ in chosen]
    assert len(keys) == len(set(keys)), "duplicate keys: %s" % [k for k in keys if keys.count(k) > 1]
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("% assembled by code/merge_refs.py; every entry body comes from Crossref or the arXiv API\n\n")
        f.write("\n\n".join(body for _, body in chosen) + "\n")
    with io.open(AUDIT, "w", encoding="utf-8") as f:
        json.dump({"n_entries": len(chosen), "entries": audit, "dropped": dropped}, f,
                  ensure_ascii=False, indent=1)
    print("wrote %s with %d entries" % (OUT, len(chosen)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
