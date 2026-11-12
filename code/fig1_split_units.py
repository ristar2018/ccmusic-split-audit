"""Figure 1: what the two split units do to one recording, and what the exposure arm adds.

Three rows, read top to bottom.

Row 1 -- the split as the framework states it, drawn over image files (8:1:1) on a
slice-level reading: one recording is a row of slices, and because the split is drawn
over files, slices of the same recording land on all three sides.

Row 2 -- the recording-level unit used in this paper. A recording is never split: each
one goes wholly to training (C), validation (R) or the sealed test panel (Q). Inside the
test recordings, half of the slices form the sealed panel and half are held out into the
pool.

Row 3 -- the exposure arm (A1). The held-out pool slices replace an equal number of C
slices, so the training budget is unchanged; some, but not necessarily all, test
recordings contribute to the pool.

Slice and recording counts are schematic: the figure shows where the slices of one
recording land, not the 8:1:1 arithmetic.

Exported at 120 mm width; the smallest type in the figure is 9 pt.
"""

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
OUT = os.path.join(PROJ, "figures")

MM = 1 / 25.4
FIG_W, FIG_H = 120 * MM, 110 * MM

# Okabe-Ito palette: colour-blind safe, and separable in greyscale by the labels.
C_TRAIN = "#0072B2"
C_VAL = "#56B4E9"
C_TEST = "#9E9E9E"
C_POOL = "#D55E00"

FS = 9.0
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS,
    "pdf.fonttype": 42,
})

SQ = 3.8
GAP = 0.7
GAP_REC = 3.0
X0 = 30.0
WRAP = 60

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def strip(y, colours, x_start, sq=SQ, gap=GAP, hollow=()):
    x = x_start
    for i, col in enumerate(colours):
        if i in hollow:
            ax.add_patch(Rectangle((x, y), sq, sq, facecolor="white", edgecolor=col,
                                   linestyle=(0, (1.3, 1.0)), linewidth=0.7))
        else:
            ax.add_patch(Rectangle((x, y), sq, sq, facecolor=col, edgecolor="none"))
        x += sq + gap
    return x - gap


def title(y, text):
    ax.text(0.0, y, text, ha="left", va="baseline", fontsize=9.6, color="#111111",
            fontweight="bold")


def para(y_top, text, colour="#444444"):
    """Wrapped left-aligned paragraph; returns the y of its last baseline."""
    lines = textwrap.wrap(text, WRAP)
    y = y_top
    for line in lines:
        ax.text(0.0, y, line, ha="left", va="baseline", fontsize=FS, color=colour)
        y -= 4.0
    return y + 4.0


# ---------------------------------------------------------------- row 1: slice level
Y1 = 80.0
row1 = [C_TRAIN] * 12
row1[4] = C_TEST
row1[7] = C_VAL
row1[10] = C_TEST
strip(Y1, row1, X0)

title(87.5, "Slice-level split")
para(74.5, "On a slice-level reading of the stated ratios, the slices of one "
           "recording land in training, validation and test alike.")

# ----------------------------------------------------------- row 2: recording level
Y2 = 53.5
xa = X0
xA_end = strip(Y2, [C_TRAIN] * 4, xa)
xb = xA_end + GAP + GAP_REC
xB_end = strip(Y2, [C_VAL] * 4, xb)
xc = xB_end + GAP + GAP_REC
xC_end = strip(Y2, [C_TEST, C_TEST, C_POOL, C_POOL], xc)

for mid, tag in (((xa + xA_end) / 2, "C"), ((xb + xB_end) / 2, "R"),
                 ((xc + xC_end) / 2, "Q")):
    ax.text(mid, Y2 + SQ + 0.8, tag, ha="center", va="bottom", fontsize=FS, style="italic",
            color="#333333")

title(61.5, "Recording-level split")
para(45.5, "As used here, a recording is never split: training recordings C, "
           "validation recordings R, and test recordings Q, whose slices are half "
           "sealed and half held out as a pool.")

# --------------------------------------------------------------- row 3: exposure arm
Y3 = 22.5
xa3 = X0
xC3_half = strip(Y3, [C_TRAIN] * 2, xa3)
x_inj = xC3_half + GAP
x_inj_end = strip(Y3, [C_POOL] * 2, x_inj)
xb3 = x_inj_end + GAP + GAP_REC
xB3_end = strip(Y3, [C_VAL] * 4, xb3)
xc3 = xB3_end + GAP + GAP_REC
xC3_end = strip(Y3, [C_TEST, C_TEST, C_POOL, C_POOL], xc3, hollow=(2, 3))

for mid, tag in (((xa3 + x_inj_end) / 2, "C + pool"),
                 ((xb3 + xB3_end) / 2, "R"), ((xc3 + xC3_end) / 2, "Q")):
    ax.text(mid, Y3 + SQ + 0.8, tag, ha="center", va="bottom", fontsize=FS, style="italic",
            color="#333333")

title(30.5, "Exposure arm (A1)")
para(18.0, "Pool slices (orange) replace an equal number of C slices, so the budget is "
           "unchanged; the dashed outlines in Q mark the same slices at their old "
           "position. Not every test recording contributes to the pool.")

# row 3 is read against row 2 without a connector: the solid orange blocks in C are the
# Q pool drawn at its new position, and the dashed outlines in Q are the same slices at
# their old position. A connector would have to cross a label or a strip.

# ------------------------------------------------------------------------- legend
handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")
           for c in (C_TRAIN, C_VAL, C_TEST, C_POOL)]
handles.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=C_POOL,
                         linestyle=(0, (1.3, 1.0)), linewidth=0.7))
ax.legend(handles,
          ["training slice", "validation slice", "sealed test slice",
           "held-out pool slice", "moved to training"],
          loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False,
          fontsize=FS, handlelength=1.0, handleheight=1.0, handletextpad=0.45,
          columnspacing=2.0, borderaxespad=0.0, labelspacing=0.45)

para(1.5, "Schematic: the counts are illustrative, not the 8:1:1 ratio.",
     colour="#666666")

for ext in ("pdf", "png"):
    path = os.path.join(OUT, "fig1_split_units.%s" % ext)
    fig.savefig(path, dpi=600 if ext == "png" else None, facecolor="white")
    print("wrote", path)

# ---------------------------------------------------------------- geometry self-check
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
sizes = [t.get_fontsize() for t in texts]
boxes = []
for t in texts:
    bb = t.get_window_extent(rend)
    boxes.append((t.get_text()[:28], bb, inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])))
outside = [(n, x) for n, bb, x in boxes
           if x[0][0] < -0.002 or x[0][1] < -0.002 or x[1][0] > 1.002 or x[1][1] > 1.002]
overlaps = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        a, b = boxes[i][1], boxes[j][1]
        if a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1:
            overlaps.append((boxes[i][0], boxes[j][0]))
print("min fontsize: %.1f pt" % min(sizes))
print("texts: %d | outside canvas: %d | overlapping pairs: %d"
      % (len(texts), len(outside), len(overlaps)))
for n in outside:
    print("  OUTSIDE:", n[0])
for p in overlaps:
    print("  OVERLAP:", p)

plt.close(fig)
