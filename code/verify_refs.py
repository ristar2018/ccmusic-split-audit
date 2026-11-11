"""Verify references against OpenAlex (DOIs) and the arXiv API (preprints).

Both are reachable over plain HTTP from this host; HTTPS to foreign hosts is
blocked, so https://api.crossref.org cannot be used.

Usage: python code/verify_refs.py code/refs_all.json
"""
import json
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "ref-check/1.0 (mailto:research@example.org)"}


def get(url, timeout=25):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def openalex(doi):
    d = json.loads(get("http://api.openalex.org/works/doi:" + doi))
    src = ((d.get("primary_location") or {}).get("source") or {})
    auth = d.get("authorships") or []
    first = ""
    if auth:
        a = auth[0].get("author") or {}
        first = a.get("display_name", "")
    return {
        "found": True,
        "title": d.get("title") or d.get("display_name") or "",
        "year": d.get("publication_year"),
        "container": src.get("display_name", "") or "",
        "first_author": first,
        "type": d.get("type", ""),
        "cited_by": d.get("cited_by_count"),
    }


def arxiv(aid):
    raw = get(f"http://export.arxiv.org/api/query?id_list={aid}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(raw)
    e = root.find("a:entry", ns)
    if e is None:
        return {"found": False, "error": "no entry"}
    title = " ".join((e.find("a:title", ns).text or "").split())
    if title.lower().startswith("error") or not title:
        return {"found": False, "error": "arxiv error entry"}
    authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    pub = (e.find("a:published", ns).text or "")[:10]
    return {"found": True, "title": title, "year": pub[:4], "container": "arXiv",
            "first_author": authors[0] if authors else "", "type": "preprint",
            "arxiv_id": aid}


def main():
    refs = json.load(open(sys.argv[1], encoding="utf-8"))
    out = []
    for r in refs:
        rec = {"key": r["key"], "claim": r.get("claim", ""), "id": r.get("doi") or r.get("arxiv")}
        try:
            rec.update(openalex(r["doi"]) if r.get("doi") else arxiv(r["arxiv"]))
        except Exception as exc:  # noqa: BLE001
            rec.update({"found": False, "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
        out.append(rec)
        flag = "OK  " if rec.get("found") else "MISS"
        print(f"{flag} {r['key'][:30]:30s} {str(rec.get('year'))[:4]:4s} "
              f"{(rec.get('title') or rec.get('error', ''))[:78]}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "refs_checked.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    ok = sum(1 for x in out if x.get("found"))
    print(f"\nverified {ok}/{len(out)}")


if __name__ == "__main__":
    main()
