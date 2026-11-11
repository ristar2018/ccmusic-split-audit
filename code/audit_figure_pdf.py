"""Ground-truth check of the exported figure PDFs.

The matplotlib renderer mis-measures mathtext bounding boxes in the PDF backend,
so the artist-level audit reports one false "clipped" xlabel on fig4. This script
checks what is actually in the file: every word box from `pdftotext -bbox-latex`
must lie inside the page, and no two word boxes may overlap.

A word box outside the page means the label is cut off; an overlap means two
labels collide. Both are what a reader would see.
"""
import json
import os
import subprocess
import xml.etree.ElementTree as ET

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = ["fig1_split_units.pdf", "fig2_p3_decomposition.pdf", "fig4_three_arms.pdf"]
TEXBIN = r"D:\texlive\2026\bin\windows\pdftotext.exe"


def _iter_tag(root, name):
    """Iterate elements whose local name matches, ignoring any XML namespace."""
    for el in root.iter():
        if el.tag.split("}")[-1] == name:
            yield el


def word_boxes(pdf):
    import tempfile
    fd, xmlpath = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    try:
        subprocess.run([TEXBIN, "-bbox", pdf, xmlpath], capture_output=True,
                       check=True)
        with open(xmlpath, encoding="utf-8", errors="replace") as fh:
            xml = fh.read()
    finally:
        try:
            os.remove(xmlpath)
        except OSError:
            pass
    root = ET.fromstring(xml)
    pages = []
    for page in _iter_tag(root, "page"):
        w, h = float(page.get("width")), float(page.get("height"))
        words = []
        for wd in _iter_tag(page, "word"):
            words.append({
                "text": (wd.text or "").strip(),
                "x0": float(wd.get("xMin")), "x1": float(wd.get("xMax")),
                "y0": float(wd.get("yMin")), "y1": float(wd.get("yMax")),
            })
        pages.append({"width": w, "height": h, "words": words})
    return pages


def overlaps(a, b, tol=0.4):
    dx = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    dy = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
    return dx > tol and dy > tol


report = []
for name in FIGS:
    path = os.path.join(PROJ, "figures", name)
    pages = word_boxes(path)
    assert len(pages) == 1, (name, len(pages))
    page = pages[0]
    words = [w for w in page["words"] if w["text"]]
    problems = []
    for w in words:
        if (w["x0"] < -0.5 or w["y0"] < -0.5 or w["x1"] > page["width"] + 0.5
                or w["y1"] > page["height"] + 0.5):
            problems.append("OUTSIDE page: %r x=[%.2f,%.2f] y=[%.2f,%.2f] page=%.2fx%.2f"
                            % (w["text"], w["x0"], w["x1"], w["y0"], w["y1"],
                               page["width"], page["height"]))
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if overlaps(words[i], words[j]):
                problems.append("OVERLAP %r vs %r" % (words[i]["text"], words[j]["text"]))
    heights = sorted({round(w["y1"] - w["y0"], 2) for w in words})
    report.append({
        "figure": name,
        "page_pt": [round(page["width"], 2), round(page["height"], 2)],
        "page_mm": [round(page["width"] / 72 * 25.4, 1), round(page["height"] / 72 * 25.4, 1)],
        "words": len(words),
        "word_heights_pt": heights[:4],
        "problems": problems,
    })

bad = [r for r in report if r["problems"]]
out = os.path.join(PROJ, "data", "results", "human_review", "figure_pdf_audit.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump({"figures": report, "figures_with_problems": len(bad)}, fh,
              ensure_ascii=False, indent=2)

for r in report:
    print("%-28s %6.1f x %6.1f mm  words=%3d  heights(pt)=%s  %s"
          % (r["figure"], r["page_mm"][0], r["page_mm"][1], r["words"],
             r["word_heights_pt"], "PROBLEMS=%d" % len(r["problems"]) if r["problems"] else "ok"))
    for p in r["problems"][:10]:
        print("   -", p)
print("report:", os.path.relpath(out, PROJ))
raise SystemExit(1 if bad else 0)
