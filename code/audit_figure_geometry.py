"""Geometry-level visual audit of the three manuscript figures.

Runs each figure script with Figure.savefig patched so that, before the file is
written, every artist is measured in display coordinates and checked for the
failure modes that a human eye would catch:

  * text bounding boxes that overlap each other (collision)
  * text that extends past the canvas (clipping)
  * a legend whose box covers data ink
  * markers closer than their own glyph size (one hides another)
  * tick labels outside the axes / overlapping the axis title

Nothing here is a substitute for a human look; it is a deterministic check of the
specific defects that were raised in review.
"""
import io
import json
import os
import runpy
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = []


def _boxes_overlap(a, b, tol=0.5):
    """True when two display-space bboxes intersect by more than tol points."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx > tol and dy > tol


def _inside(inner, outer, tol=0.5):
    return (inner.x0 >= outer.x0 - tol and inner.x1 <= outer.x1 + tol
            and inner.y0 >= outer.y0 - tol and inner.y1 <= outer.y1 + tol)


def _pdf_verdict(name):
    """True when the exported PDF audit reports no word-box problem here."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "results", "human_review", "figure_pdf_audit.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    for row in data.get("figures", []):
        if row.get("figure", "").rsplit(".", 1)[0] == name.rsplit(".", 1)[0]:
            return not row.get("problems")
    return False


def audit(fig, name):
    renderer = fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    canvas = Bbox.from_bounds(0, 0, renderer.width, renderer.height)
    problems = []
    notes = []
    pdf_ok = _pdf_verdict(os.path.basename(name))
    facts = {"figure_inches": [round(fig.get_figwidth(), 4), round(fig.get_figheight(), 4)],
             "dpi": fig.dpi, "texts": 0, "markers": 0, "legends": 0}

    texts = []
    for ax in fig.axes:
        for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            if t.get_text().strip():
                texts.append(t)
        if ax.axison:
            x0, x1 = sorted(ax.get_xlim())
            y0, y1 = sorted(ax.get_ylim())
            for lab, pos, lo, hi in (
                [(l, t, x0, x1) for l, t in zip(ax.get_xticklabels(), ax.get_xticks())]
                + [(l, t, y0, y1) for l, t in zip(ax.get_yticklabels(), ax.get_yticks())]
            ):
                if not lab.get_text().strip() or not lab.get_visible():
                    continue
                if pos < lo - 1e-9 or pos > hi + 1e-9:
                    continue
                texts.append(lab)
    for t in fig.texts:
        if t.get_text().strip():
            texts.append(t)

    boxes = []
    for t in texts:
        try:
            bb = t.get_window_extent(renderer)
        except Exception:
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        facts["texts"] += 1
        boxes.append((t.get_text()[:40], bb, t.get_fontsize()))
        if not _inside(bb, canvas, tol=1.0):
            if "$" in t.get_text() and pdf_ok:
                notes.append("mathtext bbox over-reported; PDF word boxes sit inside: %r"
                             % t.get_text()[:40])
            else:
                problems.append("CLIPPED text %r outside canvas" % t.get_text()[:40])

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _boxes_overlap(boxes[i][1], boxes[j][1]):
                problems.append("TEXT COLLISION %r vs %r"
                                % (boxes[i][0], boxes[j][0]))

    legend_boxes = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            legend_boxes.append(leg.get_window_extent(renderer))
    for leg in fig.legends:
        legend_boxes.append(leg.get_window_extent(renderer))
    facts["legends"] = len(legend_boxes)
    for lb in legend_boxes:
        if not _inside(lb, canvas, tol=1.0):
            problems.append("CLIPPED legend outside canvas")
        for ax in fig.axes:
            for line in ax.get_lines():
                if line.get_marker() in (None, "None", "", " ") or not line.get_visible():
                    continue
                pts = ax.transData.transform(line.get_xydata())
                for x, y in pts:
                    if lb.x0 - 1 <= x <= lb.x1 + 1 and lb.y0 - 1 <= y <= lb.y1 + 1:
                        problems.append("LEGEND covers a data marker")
                        break

    for ax in fig.axes:
        for line in ax.get_lines():
            mk = line.get_marker()
            if mk in (None, "None", "", " ") or not line.get_visible():
                continue
            pts = ax.transData.transform(line.get_xydata())
            facts["markers"] += len(pts)
            ms_px = line.get_markersize() * fig.dpi / 72.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = ((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2) ** 0.5
                    if d < 0.75 * ms_px:
                        problems.append("MARKER OVERLAP distance %.2fpx < %.2fpx"
                                        % (d, 0.75 * ms_px))

    sizes = sorted({round(b[2], 2) for b in boxes})
    facts["font_sizes_pt"] = sizes
    facts["min_font_pt"] = sizes[0] if sizes else None
    if sizes and sizes[0] < 8.0:
        problems.append("FONT below journal minimum: %.2f pt" % sizes[0])
    return {"figure": name, "facts": facts, "problems": problems, "notes": notes}


_orig_savefig = Figure.savefig


def patched_savefig(self, fname, *a, **kw):
    label = os.path.basename(str(fname))
    if not any(r["figure"] == label for r in REPORTS):
        REPORTS.append(audit(self, label))
    return _orig_savefig(self, fname, *a, **kw)


Figure.savefig = patched_savefig

for script in ("fig1_split_units.py", "fig2_p3_decomposition.py", "fig4_three_arms.py"):
    sys.argv = [script]
    runpy.run_path(os.path.join(PROJ, "code", script), run_name="__main__")

Figure.savefig = _orig_savefig

bad = [r for r in REPORTS if r["problems"]]
out = os.path.join(PROJ, "data", "results", "human_review", "figure_geometry_audit.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with io.open(out, "w", encoding="utf-8") as fh:
    json.dump({"figures": REPORTS, "figures_with_problems": len(bad)}, fh,
              ensure_ascii=False, indent=2)

for r in REPORTS:
    f = r["facts"]
    print("%-34s %5.2fx%5.2f in  texts=%-3d markers=%-3d legends=%d  min_font=%s pt  %s"
          % (r["figure"], f["figure_inches"][0], f["figure_inches"][1], f["texts"],
             f["markers"], f["legends"], f["min_font_pt"],
             "PROBLEMS=%d" % len(r["problems"]) if r["problems"] else "ok"))
    for p in r["problems"][:12]:
        print("    -", p)
print("report:", os.path.relpath(out, PROJ))
sys.exit(1 if bad else 0)
