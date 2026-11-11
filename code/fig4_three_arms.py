"""Figure 2: per-library macro-F1 of the three training arms on the sealed test panel.

Five stacked panels (one per library) x two rows (AlexNet above, Swin-T below). Each
panel keeps its own x scale: the A0 baselines span 0.69-0.94 across the five libraries,
so a shared axis would flatten the 2-point CTIS effect into a single line.

Markers per row: A0 (filled circle), A1 (filled triangle), A2 (filled square), and the
A2 self-test score (open square). The three arms are dodged vertically by +-0.34 of a
row so that the markers and their whiskers cannot fall on top of each other, and a thin
grey segment joins A0 to A1 to make Delta1 readable without a table.

Whiskers are +-1 s.d. over the three training seeds, for all four series. They are seed
dispersion, not confidence intervals: e1_metrics.csv carries no interval column, and the
paired Delta intervals that the paper quotes are computed from the sealed predictions in
Table 2, on a different (recording-cluster bootstrap) basis. The legend says so.

Data sources:
  - A0/A1/A2 on Q_test: data/results/e1_metrics.csv (per-seed rows; mean and s.d. over
    the three seeds; slice-level macro-F1)
  - A2 self-test: runs/sealed_eval/<lib>_A2_<bb>/seed<N>_a2_selftest.npz (recomputed per
    seed, then mean and s.d.)

Exported at 120 mm width; the smallest type in the figure is 9 pt.

Usage:
  python code/fig4_three_arms.py [--out-dir figures]
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from analyze_e1 import macro_f1, conf_slice, load_npz  # noqa: E402

MM = 1 / 25.4
LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto"]
# Display labels: the paper writes the library names as words, not as directory keys.
LIB_LABELS = {"CTIS": "CTIS", "music_genre": "music genre", "pianos": "pianos",
              "CNPM": "CNPM", "bel_canto": "bel canto"}
BBS = ["alexnet", "swin_t"]
BBS_LAB = {"alexnet": "AlexNet", "swin_t": "Swin-T"}
ARMS = ["A0", "A1", "A2"]

# Okabe-Ito palette: colour-blind safe, and the marker shapes stay distinguishable in
# greyscale print.
C_A0 = "#0072B2"
C_A1 = "#D55E00"
C_A2 = "#009E73"
C_SELF = "#555555"
MARK = {"A0": "o", "A1": "^", "A2": "s"}
MARK_SZ = 4.0

# Vertical dodge inside one backbone row, so the three arms never share a y position.
DY = {"A0": 0.34, "A1": 0.00, "A2": -0.34}
DY_SELF = 0.66

FS = 9.0
plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42})


def read_metrics(path):
    """Mean and s.d. over the per-seed rows of e1_metrics.csv."""
    acc = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["library"] not in LIBS or r["arm"] not in ARMS:
                continue
            k = (r["library"], r["arm"], r["bb"])
            acc.setdefault(k, []).append(float(r["macro_f1_slice"]))
    out = {}
    for k, v in acc.items():
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        out[k] = (float(np.mean(v)), sd, len(v))
    return out


def selftest_scores(nc_by_lib):
    """A2 self-test: recompute slice-level macro-F1 per seed, then mean and s.d."""
    out = {}
    for lib in LIBS:
        for bb in BBS:
            vals = []
            for p in sorted(glob.glob(os.path.join(
                    PROJ, "runs", "sealed_eval", f"{lib}_A2_{bb}", "seed*_a2_selftest.npz"))):
                probs, y, _ = load_npz(p)          # load_npz returns (probs, y, records)
                nc = nc_by_lib[lib]
                conf = conf_slice(y, probs.argmax(1), nc)
                vals.append(macro_f1(conf)[0])     # macro_f1 returns (score, per-class f1, keep)
            if vals:
                out[(lib, bb)] = (float(np.mean(vals)),
                                  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                  len(vals))
    return out


def whisker(ax, x, y, sd, colour):
    """+-1 s.d. bar; cap size is set in points, not in data units."""
    if sd <= 0:
        return
    # The uncertainty is in the score (x), so the bar runs horizontally.
    ax.errorbar(x, y, xerr=sd, fmt="none", ecolor=colour, elinewidth=0.6,
                capsize=1.8, capthick=0.6, zorder=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(PROJ, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    met = read_metrics(os.path.join(PROJ, "data", "results", "e1_metrics.csv"))
    nc_by_lib = {}
    with open(os.path.join(PROJ, "data", "splits", "v2", "class_universe.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("kept", "").lower() in ("true", "1"):
                nc_by_lib[r["library"]] = nc_by_lib.get(r["library"], 0) + 1
    self_ = selftest_scores(nc_by_lib)

    W = 120.0 * MM
    H = 118.0 * MM
    fig, axes = plt.subplots(len(LIBS), 1, figsize=(W, H))
    fig.subplots_adjust(left=0.185, right=0.965, top=0.845, bottom=0.115, hspace=0.86)

    for ax, lib in zip(axes, LIBS):
        ys = {"alexnet": 1.0, "swin_t": 0.0}
        xs_all = []
        for bb in BBS:
            for arm in ARMS:
                xs_all.append(met[(lib, arm, bb)][0])
            if (lib, bb) in self_:
                xs_all.append(self_[(lib, bb)][0])
        lo, hi = min(xs_all), max(xs_all)
        pad = max(0.012, (hi - lo) * 0.18)
        ax.set_xlim(lo - pad, min(hi + pad, 1.0))

        for bb in BBS:
            y = ys[bb]
            a0 = met[(lib, "A0", bb)][0]
            a1 = met[(lib, "A1", bb)][0]
            ax.plot([a0, a1], [y + DY["A0"], y + DY["A1"]], color="#B0B0B0", lw=0.8,
                    zorder=1, solid_capstyle="butt")
            for arm in ARMS:
                v, sd, _ = met[(lib, arm, bb)]
                colour = {"A0": C_A0, "A1": C_A1, "A2": C_A2}[arm]
                whisker(ax, v, y + DY[arm], sd, colour)
                ax.scatter(v, y + DY[arm], marker=MARK[arm], s=MARK_SZ ** 2 * 1.4,
                           color=colour, linewidths=0.6, edgecolors="white",
                           zorder=3.0 + {"A2": 0.1, "A1": 0.2}.get(arm, 0.0))
            if (lib, bb) in self_:
                xs, sd, _ = self_[(lib, bb)]
                whisker(ax, xs, y + DY_SELF, sd, C_SELF)
                ax.scatter(xs, y + DY_SELF, marker="s", s=MARK_SZ ** 2 * 1.4,
                           facecolors="none", edgecolors=C_SELF, linewidths=0.9, zorder=4)

        ax.set_ylim(-0.92, 2.02)
        ax.set_yticks([1.0, 0.0])
        # The dodge puts the markers off the tick line; the label marks the row centre.
        ax.set_yticklabels(["AlexNet", "Swin-T"], fontsize=FS)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", labelsize=FS, length=2.0, pad=1.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="x", color="#E8E8E8", lw=0.5, zorder=0)
        ax.set_title(LIB_LABELS[lib], loc="left", fontsize=9.6, pad=3.0)

    axes[-1].set_xlabel("macro-F1 (slice level)\nfilled: sealed panel $Q_{\\mathrm{test}}$;"
                        " hollow: A2 self-test (different panel)",
                        fontsize=FS, labelpad=2.2)

    handles = [
        Line2D([], [], marker="o", ls="", color=C_A0, ms=MARK_SZ,
               label="A0 ($C$ only)"),
        Line2D([], [], marker="^", ls="", color=C_A1, ms=MARK_SZ,
               label="A1 (+$Q_{\\mathrm{pool}}$)"),
        Line2D([], [], marker="s", ls="", color=C_A2, ms=MARK_SZ,
               label="A2 (random split)"),
        Line2D([], [], marker="s", ls="", mfc="none", mec=C_SELF, color=C_SELF,
               ms=MARK_SZ, label="A2 self-test"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=FS,
               frameon=False, bbox_to_anchor=(0.5, 0.985), handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0.2, labelspacing=0.45)

    pdf = os.path.join(a.out_dir, "fig4_three_arms.pdf")
    png = os.path.join(a.out_dir, "fig4_three_arms.png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, dpi=600)

    # ------------------------------------------------------------ geometry self-check
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def collect():
        """Current (not stale) text artists: tick labels are re-read from each axes."""
        out = []
        for ax_ in axes:
            # get_xticklabels() can still list labels of ticks the current locator
            # no longer emits; keep only the ones that match a live tick value.
            xlo_, xhi_ = ax_.get_xlim()
            live = {"%.2f" % v for v in ax_.get_xticks() if xlo_ - 1e-9 <= v <= xhi_ + 1e-9}
            out.extend([t for t in ax_.get_xticklabels()
                        if t.get_text().strip() and t.get_visible()
                        and t.get_text() in live])
            ylo_, yhi_ = ax_.get_ylim()
            out.extend([t for t in ax_.get_yticklabels()
                        if t.get_text().strip() and t.get_visible()
                        and ylo_ - 1e-9 <= t.get_position()[1] <= yhi_ + 1e-9])
            for t in (ax_.title, ax_.xaxis.label, ax_.yaxis.label):
                if t.get_text().strip():
                    out.append(t)
        out.extend([t for t in fig.legends[0].get_texts()
                    if t.get_text().strip() and t.get_visible()])
        return out

    texts = collect()
    boxes = []
    for t in texts:
        bb = t.get_window_extent(rend)
        fr = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        boxes.append((t.get_text()[:24].replace("\n", " "), bb, fr))
    outside = [(n, round(float(fr[0][0]), 3), round(float(fr[1][0]), 3))
               for n, bb, fr in boxes
               if fr[0][0] < -0.002 or fr[0][1] < -0.002
               or fr[1][0] > 1.002 or fr[1][1] > 1.002]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            q, r = boxes[i][1], boxes[j][1]
            if q.x0 < r.x1 and r.x0 < q.x1 and q.y0 < r.y1 and r.y0 < q.y1:
                overlaps.append((boxes[i][0], boxes[j][0]))
    print("min fontsize: %.1f pt" % min(t.get_fontsize() for t in texts))
    print("texts: %d | outside canvas: %d | overlapping pairs: %d"
          % (len(texts), len(outside), len(overlaps)))
    for n in outside:
        print("  OUTSIDE:", n)
    for o in overlaps[:12]:
        print("  OVERLAP:", o)

    plt.close(fig)
    for lib in LIBS:
        row = []
        for bb in BBS:
            r = met[(lib, "A0", bb)], met[(lib, "A1", bb)], met[(lib, "A2", bb)]
            s = self_.get((lib, bb))
            row.append("%s: A0 %.3f+-%.3f A1 %.3f+-%.3f A2 %.3f+-%.3f self %s" % (
                bb, r[0][0], r[0][1], r[1][0], r[1][1], r[2][0], r[2][1],
                ("%.3f+-%.3f(n=%d)" % s) if s else "NA"))
        print(lib, "|", " || ".join(row))


if __name__ == "__main__":
    main()
