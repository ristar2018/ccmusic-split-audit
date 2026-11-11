"""Figure 3: how much of the cross-instrument gap is recording-level leakage.

The pooled macro-F1 of four partitions of the same CTIS technique task. The display
labels follow the partitions the review asked for, and they are deliberately not the
A0/A1/A2 names used in the main experiment, because the arms mean something different
here:

    R-slice     random slice-level folds          (code key A0   -- the leaky baseline)
    R-rec       recording-grouped folds, the
                instrument present on both sides  (code key A1r)
    L-inst      instrument-withheld folds         (code key A1)
    L-inst+Adv  the adversarial arm on top of L-inst  (code key A2)

The two gaps are drawn separately because they measure different things: R-slice ->
R-rec is within-instrument recording leakage, R-rec -> L-inst is instrument novelty.

Bars are means over two seeds with the 95 % recording-cluster bootstrap interval.
Two stacked panels (AlexNet above, Swin-T below) rather than two columns, so that the
four display labels fit at 9 pt without rotation.

Exported at 120 mm width; the smallest type in the figure is 9 pt.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RES = os.path.join(PROJ, "data", "results")
OUT = os.path.join(PROJ, "figures")

MM = 1 / 25.4
FIG_W, FIG_H = 120 * MM, 96 * MM

C_A0 = "#0072B2"
C_A1R = "#009E73"
C_A1 = "#D55E00"
C_A2 = "#CC79A7"

FS = 9.0
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FS,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
})


def read_metrics(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["arm"], r["bb"])] = r
    return rows


main = read_metrics(os.path.join(RES, "p3", "p3_metrics.csv"))
leak = read_metrics(os.path.join(RES, "p3b_leak", "p3_metrics.csv"))

# (code key in the CSVs, display label, colour)
ARMS = [("A0", "R-slice", C_A0),
        ("A1r", "R-rec", C_A1R),
        ("A1", "L-inst", C_A1),
        ("A2", "L-inst+Adv", C_A2)]

fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharey=True)
fig.subplots_adjust(left=0.135, right=0.985, top=0.925, bottom=0.135, hspace=0.34)

for ax, bb, title in zip(axes, ("alexnet", "swin_t"), ("AlexNet", "Swin-T")):
    vals, los, his, cols = [], [], [], []
    for arm, _, colour in ARMS:
        src = leak if arm == "A1r" else main
        r = src[(arm, bb)]
        vals.append(float(r["macro_f1_slice"]) * 100)
        los.append(float(r["slice_ci_lo"]) * 100)
        his.append(float(r["slice_ci_hi"]) * 100)
        cols.append(colour)
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.60, color=cols, edgecolor="none", zorder=2)
    ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                fmt="none", ecolor="#333333", elinewidth=0.55, capsize=1.8, capthick=0.55,
                zorder=3)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.65, len(vals) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels([a[1] for a in ARMS], fontsize=FS)
    ax.set_title(title, fontsize=9.6, pad=3.0, loc="left")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="y", labelsize=FS, length=2.0, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=2.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", lw=0.35, color="#DDDDDD", zorder=0)

fig.supylabel("pooled macro-F1 on held-out folds, %", fontsize=FS, x=0.028)

# Report the two gaps above the bars, one per backbone.
for ax, bb, gaps in ((axes[0], "alexnet", ("-2.9", "-36.7")),
                     (axes[1], "swin_t", ("-3.4", "-31.4"))):
    ax.annotate("", xy=(1, 88), xytext=(0, 88),
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color="#444444",
                                mutation_scale=4))
    ax.text(0.5, 90.0, gaps[0], ha="center", va="bottom", fontsize=FS, color="#444444")
    ax.annotate("", xy=(2, 88), xytext=(1, 88),
                arrowprops=dict(arrowstyle="-|>", lw=0.55, color="#444444",
                                mutation_scale=4))
    ax.text(1.5, 90.0, gaps[1], ha="center", va="bottom", fontsize=FS, color="#444444")

for ext in ("pdf", "png"):
    path = os.path.join(OUT, "fig2_p3_decomposition.%s" % ext)
    fig.savefig(path, dpi=600 if ext == "png" else None, facecolor="white")
    print("wrote", path)

# ---------------------------------------------------------------- geometry self-check
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()

texts = []
for ax in axes:
    xlo, xhi = ax.get_xlim()
    texts.extend([t for t in ax.get_xticklabels()
                  if t.get_text().strip() and t.get_visible()])
    ylo, yhi = ax.get_ylim()
    texts.extend([t for t in ax.get_yticklabels()
                  if t.get_text().strip() and t.get_visible()
                  and ylo - 1e-9 <= t.get_position()[1] <= yhi + 1e-9])
    for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
        if t.get_text().strip():
            texts.append(t)

boxes = []
for t in texts:
    bb = t.get_window_extent(rend)
    fr = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    boxes.append((t.get_text()[:26], bb, fr))
outside = [(n, round(float(fr[0][0]), 3), round(float(fr[1][0]), 3))
           for n, bb, fr in boxes
           if fr[0][0] < -0.002 or fr[0][1] < -0.002
           or fr[1][0] > 1.002 or fr[1][1] > 1.002]
overlaps = []
for i in range(len(boxes)):
    for j in range(i + 1, len(boxes)):
        p, q = boxes[i][1], boxes[j][1]
        if p.x0 < q.x1 and q.x0 < p.x1 and p.y0 < q.y1 and q.y0 < p.y1:
            overlaps.append((boxes[i][0], boxes[j][0]))
print("min fontsize: %.1f pt" % min(t.get_fontsize() for t in texts))
print("texts: %d | outside canvas: %d | overlapping pairs: %d"
      % (len(texts), len(outside), len(overlaps)))
for n in outside:
    print("  OUTSIDE:", n)
for o in overlaps:
    print("  OVERLAP:", o)

plt.close(fig)
