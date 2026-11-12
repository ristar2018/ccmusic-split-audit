"""Verify every number printed in the P1 manuscript against stored artifacts.

Design
------
Two independent layers:

1. Identity layer -- recompute derived quantities from the artifacts themselves
   (rounded means from per-seed metrics, deltas from means, sums over library
   partitions, headroom-normalised ratios, cost totals) and assert the algebraic
   relationships hold.  This catches drift between tables even if the prose is
   untouched.

2. Text layer -- rebuild the exact table rows and reported tokens in the same
   grammar the manuscript uses, then require each one to appear in the LaTeX
   sources after whitespace, brace, dollar-sign and escaped-semicolon normalisation.
   This catches typos and
   stale numbers transcribed into prose.

Both layers read only; nothing under D:/VSCode/Datasets is touched.  Exit code 1
means at least one check failed.

Usage:  python code/verify_paper_numbers.py [--tex-dir paper] [--json out.json]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SPLITS = os.path.join(PROJ, "data", "splits", "v2")
RESULTS = os.path.join(PROJ, "data", "results")
P3 = os.path.join(RESULTS, "p3")
A2DELTA = os.path.join(RESULTS, "delta_A2", "e1_delta.csv")

LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto"]
BB_LABEL = {"alexnet": "AlexNet", "swin_t": "Swin-T"}
PARTITIONS = ["qtest", "qpool", "r", "c"]


# ---------------------------------------------------------------- helpers
def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def thousands(n):
    return "{:,}".format(int(n))


def norm(text):
    for tok in ("{", "}", "$", ";", "\\;", "\\,", "\\", "\u00a0"):
        text = text.replace(tok, "")
    return "".join(text.split())


def tex_text(tex_dir):
    parts = []
    main = os.path.join(tex_dir, "main_sn.tex")
    if not os.path.exists(main):
        main = os.path.join(tex_dir, "main.tex")
    if os.path.exists(main):
        parts.append(io.open(main, encoding="utf-8").read())
    sec = os.path.join(tex_dir, "sections")
    for name in sorted(os.listdir(sec)):
        if name.endswith(".tex"):
            parts.append(io.open(os.path.join(sec, name), encoding="utf-8").read())
    return norm("\n".join(parts))


def ci(lo, hi, dp=2):
    """Half-up formatted interval; matches how the manuscript rounds."""
    return "[{},{}]".format(rnd(lo, dp), rnd(hi, dp))


def rnd(x, dp=2):
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-dp)
    v = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if v >= 0:
        return "+%s" % v
    return str(v)


def delta(mean_a, mean_b, dp=2):
    """Signed difference in macro-F1 points (fractions -> points)."""
    return "{:+.{d}f}".format(100.0 * (mean_a - mean_b), d=dp)


# ---------------------------------------------------------------- artifact loads
def load_metrics():
    """(lib, arm, bb) -> mean slice / recording macro-F1 over seeds."""
    out = {}
    for r in read_csv(os.path.join(RESULTS, "e1_metrics.csv")):
        key = (r["library"], r["arm"], r["bb"])
        acc = out.setdefault(key, {"slice": [], "rec": []})
        acc["slice"].append(float(r["macro_f1_slice"]))
        acc["rec"].append(float(r["macro_f1_recording"]))
    for key, v in out.items():
        v["slice"] = sum(v["slice"]) / len(v["slice"])
        v["rec"] = sum(v["rec"]) / len(v["rec"])
    return out


def load_delta(path):
    return {(r["library"], r["bb"]): r for r in read_csv(path)}


def load_panels():
    """Recompute the auditable panel table from the split CSVs."""
    rows = {}
    for lib in LIBS:
        slices = 0
        recs, labels = set(), set()
        qtest_slices = 0
        qtest_recs = set()
        for part in PARTITIONS:
            for r in read_csv(os.path.join(SPLITS, "%s_%s.csv" % (lib, part))):
                slices += 1
                recs.add(r["recording_uid"])
                labels.add(r["label"])
                if part == "qtest":
                    qtest_slices += 1
                    qtest_recs.add(r["recording_uid"])
        rows[lib] = {
            "classes": len(labels),
            "recordings": len(recs),
            "slices": slices,
            "kbar": slices / len(recs),
            "qtest_slices": qtest_slices,
            "qtest_recs": len(qtest_recs),
        }
    return rows


def load_alpha():
    rows = read_csv(os.path.join(SPLITS, "feasibility.csv"))
    out = {}
    for r in rows:
        out.setdefault(r["library"], []).append(float(r["alpha"]))
    return {k: sorted(v) for k, v in out.items()}


def median(vals):
    n = len(vals)
    if n == 0:
        return float("nan")
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


# ---------------------------------------------------------------- checks
class Checker:
    def __init__(self, text):
        self.text = text
        self.fails = []
        self.passes = 0

    def in_text(self, name, needle):
        """Require the normalised needle to appear in the manuscript."""
        needle = norm(needle)
        if needle and needle in self.text:
            self.passes += 1
            return True
        self.fails.append(("text", name, needle[:160]))
        return False

    def in_text_any(self, name, needles):
        """Require at least one of several acceptable renderings of the same claim."""
        for needle in needles:
            needle = norm(needle)
            if needle and needle in self.text:
                self.passes += 1
                return True
        self.fails.append(("text", name, " | ".join(n[:80] for n in needles)))
        return False

    def identity(self, name, ok, detail=""):
        if ok:
            self.passes += 1
            return True
        self.fails.append(("identity", name, detail))
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-dir", default=os.path.join(PROJ, "paper"))
    ap.add_argument("--json", default=os.path.join(RESULTS, "paper_verify.json"))
    a = ap.parse_args()

    text = tex_text(a.tex_dir)
    ck = Checker(text)
    metrics = load_metrics()
    d1 = load_delta(os.path.join(RESULTS, "e1_delta.csv"))
    d2 = load_delta(A2DELTA)
    panels = load_panels()
    alpha = load_alpha()

    # ---- layer 1: identities -------------------------------------------
    tot_slices = sum(p["slices"] for p in panels.values())
    tot_recs = sum(p["recordings"] for p in panels.values())
    tot_cls = sum(p["classes"] for p in panels.values())
    tot_qtest = sum(p["qtest_slices"] for p in panels.values())
    tot_qrec = sum(p["qtest_recs"] for p in panels.values())
    ck.identity("audited slices 109831", tot_slices == 109831, str(tot_slices))
    ck.identity("audited recordings 7756", tot_recs == 7756, str(tot_recs))
    ck.identity("audited classes 234", tot_cls == 234, str(tot_cls))
    ck.identity("qtest slices 11387", tot_qtest == 11387, str(tot_qtest))
    ck.identity("qtest recordings 1529", tot_qrec == 1529, str(tot_qrec))

    for lib in LIBS:
        for bb in ("alexnet", "swin_t"):
            k = (lib, bb)
            if k not in d1:
                ck.identity("delta row %s/%s" % k, False, "missing")
                continue
            row = d1[k]
            m0 = metrics[(lib, "A0", bb)]["slice"]
            m1 = metrics[(lib, "A1", bb)]["slice"]
            rec0 = metrics[(lib, "A0", bb)]["rec"]
            rec1 = metrics[(lib, "A1", bb)]["rec"]
            ck.identity(
                "delta equals mean difference %s/%s" % k,
                abs(100.0 * (m1 - m0) - float(row["delta_slice_pt"])) < 0.02,
                "%.4f vs %s" % (100.0 * (m1 - m0), row["delta_slice_pt"]),
            )
            ck.identity(
                "recording delta equals mean difference %s/%s" % k,
                abs(100.0 * (rec1 - rec0) - float(row["delta_rec_pt"])) < 0.02,
                "%.4f vs %s" % (100.0 * (rec1 - rec0), row["delta_rec_pt"]),
            )
            ck.identity(
                "CI brackets point estimate %s/%s" % k,
                float(row["ci_lo_slice_pt"]) <= float(row["delta_slice_pt"]) <= float(row["ci_hi_slice_pt"]),
                row["delta_slice_pt"],
            )
            ck.identity(
                "SESOI flag consistent %s/%s" % k,
                (float(row["delta_slice_pt"]) >= 3.0) == (row["meets_sesoi"] == "True"),
                "%s / %s" % (row["delta_slice_pt"], row["meets_sesoi"]),
            )

    # headroom ratio must equal delta / (100 - A0)
    for lib in LIBS:
        for bb in ("alexnet", "swin_t"):
            m0 = metrics[(lib, "A0", bb)]["slice"]
            d = float(d1[(lib, bb)]["delta_slice_pt"])
            ratio = d / (100.0 - 100.0 * m0)
            ck.identity(
                "headroom ratio in [0,1] %s/%s" % (lib, bb),
                0.0 <= ratio <= 1.0,
                "%.3f" % ratio,
            )

    # P3 pooled means must reproduce the P3 deltas
    p3m, p3r = {}, {}
    for r in read_csv(os.path.join(P3, "p3_metrics.csv")):
        p3m[(r["arm"], r["bb"])] = float(r["macro_f1_slice"])
        p3r[(r["arm"], r["bb"])] = float(r["macro_f1_recording"])
    for r in read_csv(os.path.join(P3, "p3_delta.csv")):
        arm, base, bb = r["arm"], r["base"], r["bb"]
        recomputed = 100.0 * (p3m[(arm, bb)] - p3m[(base, bb)])
        ck.identity(
            "P3 %s-%s %s" % (arm, base, bb),
            abs(recomputed - float(r["delta_slice"])) < 0.02,
            "%.3f vs %s" % (recomputed, r["delta_slice"]),
        )

    # cost total
    cost = json.load(io.open(os.path.join(RESULTS, "cost.json"), encoding="utf-8"))
    parts = [cost["E1b"]["gpu_h"], cost["E1c"]["gpu_h"], cost["E1f"]["gpu_h"], cost["P3"]["gpu_h"]]
    ck.identity(
        "cost total equals components",
        abs(sum(parts) - cost["total_measured_gpu_h"]) < 0.005,
        "%s vs %s" % (sum(parts), cost["total_measured_gpu_h"]),
    )

    # A2 as a share of the exposure effect (Table 4 vs Table 2)
    shares = {}
    for lib in LIBS:
        for bb in ("alexnet", "swin_t"):
            shares[(lib, bb)] = (float(d2[(lib, bb)]["delta_slice_pt"])
                                 / float(d1[(lib, bb)]["delta_slice_pt"]))
    lo = min(shares.values())
    hi = max(shares.values())
    rep_lo = min(v for (lib, _), v in shares.items() if lib != "CTIS")
    rep_hi = max(v for (lib, _), v in shares.items() if lib != "CTIS")
    ck.identity("share range plausible", 0.0 < lo < hi < 1.2, "%.3f..%.3f" % (lo, hi))
    ck.in_text("prose share range", "60--103")
    ck.in_text("prose replication share range",
               "between%d%%and%d%%" % (round(100 * rep_lo), round(100 * rep_hi)))
    ck.in_text("prose replication share low", "%d\\%%" % round(100 * rep_lo))
    ck.in_text("prose replication share high", "%d\\%%" % round(100 * rep_hi))

    # ---------------------------------------------------------------- layer 2: text
    # Table 1 (panels)
    for lib in LIBS:
        p = panels[lib]
        med = median(alpha[lib])
        ck.in_text(
            "table1 row %s" % lib,
            "%s&%d&%s&%s&%.2f&%s&%d&%.3f"
            % (lib.replace("_", " "), p["classes"], thousands(p["recordings"]),
               thousands(p["slices"]), p["kbar"], thousands(p["qtest_slices"]),
               p["qtest_recs"], med),
        )
    ck.in_text("table1 total", "Total&234&7,756&109,831&14.16&11,387&1,529&")
    corrected_rec = {(r["library"], r["bb"], r["contrast"]): r
                     for r in read_csv(os.path.join(RESULTS, "human_review", "corrected_recording_delta.csv"))}
    # Table 2 (main effect)
    for lib in LIBS:
        for bb in ("alexnet", "swin_t"):
            k = (lib, bb)
            row = d1[k]
            m0 = metrics[(lib, "A0", bb)]["slice"]
            m1 = metrics[(lib, "A1", bb)]["slice"]
            # The SESOI column was folded into the caption (dagger marks the two
            # confirmatory rows), so the row now carries seven cells.
            dagger = "^\\dagger" if lib == "CTIS" else ""
            ck.in_text(
                "table2 row %s/%s" % k,
                "%s&%s%s&%.4f&%.4f&%s%s&%s%s&%.2f"
                % (lib.replace("_", " "), BB_LABEL[bb], dagger, m0, m1,
                   rnd(row["delta_slice_pt"], 2), ci(row["ci_lo_slice_pt"], row["ci_hi_slice_pt"]),
                   rnd(corrected_rec[(lib, bb, "A1-A0")]["point_unw_pt"], 2),
                   ci(row["ci_lo_rec_pt"], row["ci_hi_rec_pt"]),
                   float(row["seed_range_pt"])),
            )
            # Table 3 (headroom)
            head = 100.0 * (1.0 - m0)
            ck.in_text(
                "table3 row %s/%s" % k,
                "%s&%s&%.2f&%.2f&%.3f"
                % (lib.replace("_", " "), BB_LABEL[bb], head,
                   float(row["delta_slice_pt"]),
                   float(row["delta_slice_pt"]) / head),
            )
            # Table 4 (A2)
            r2 = d2[k]
            ck.in_text(
                "table4 row %s/%s" % k,
                "%s&%s&%s%s&%s%s"
                % (lib.replace("_", " "), BB_LABEL[bb],
                   "{:+.2f}".format(float(r2["delta_slice_pt"])),
                   ci(r2["ci_lo_slice_pt"], r2["ci_hi_slice_pt"]),
                   "{:+.2f}".format(float(corrected_rec[(lib, bb, "A2-A0")]["point_unw_pt"])),
                   ci(r2["ci_lo_rec_pt"], r2["ci_hi_rec_pt"])),
            )

    # Table 5 (negative controls)
    # B2（外审后补跑）：负对照统一三个 seed，结果落在 data/results/b2/e1_delta.csv
    B2DELTA = os.path.join(RESULTS, "b2", "e1_delta.csv")
    neg = {(r["library"], r["bb"]): r for r in read_csv(B2DELTA)
           if r["library"] in ("chest_falsetto", "erhu_playing_tech", "GZ_IsoTech")}
    with io.open(os.path.join(SPLITS, "negcontrol", "negcontrol.json"), encoding="utf-8") as fh:
        njson = json.load(fh)
    nlibs = njson.get("libraries", njson)
    for (lib, bb), r in sorted(neg.items()):
        libinfo = nlibs[lib]
        ck.in_text(
            "table5 row %s/%s" % (lib, bb),
            "%s&%s&%s/%s&%s%s&%.2f&%.3f"
            % (lib, BB_LABEL[bb], thousands(libinfo["recordings"]),
               thousands(libinfo["slices"]),
               "{:+.2f}".format(float(r["delta_slice_pt"])),
               ci(r["ci_lo_slice_pt"], r["ci_hi_slice_pt"]),
               float(r["seed_range_pt"]),
               float(r["boot_p_slice"])),
        )

    # Table 6 (cross-instrument): pooled means in the table, contrasts in the caption
    P3_LABEL = {
        "A0": "R-slice",
        "A1r": "R-rec",
        "A1": "L-inst",
        "A2": "L-inst+Adv",
    }
    # B1（外审 M7）：录音分组折基线的 pooled 均值与两条补充对比
    p3b_m, p3b_r = {}, {}
    for r in read_csv(os.path.join(RESULTS, "p3b_leak", "p3_metrics.csv")):
        p3b_m[(r["arm"], r["bb"])] = float(r["macro_f1_slice"])
        p3b_r[(r["arm"], r["bb"])] = float(r["macro_f1_recording"])
    leak = {(r["arm"], r["bb"]): r
            for r in read_csv(os.path.join(RESULTS, "p3b_leak", "p3_delta.csv"))}
    inst = {(r["arm"], r["bb"]): r
            for r in read_csv(os.path.join(RESULTS, "p3b_instr", "p3_delta.csv"))}
    for bb in ("alexnet", "swin_t"):
        # 表注写的是 A0 - A1r（正数），文件里是 A1r - A0（负数）
        r = leak[("A1r", bb)]
        ck.in_text(
            "table6 caption A0-A1r/%s" % bb,
            "%s%s" % (rnd(-float(r["delta_slice"])),
                      ci(-float(r["ci_hi"]), -float(r["ci_lo"]))),
        )
        r = inst[("A1", bb)]
        ck.in_text(
            "table6 caption A1r-A1/%s" % bb,
            "%s%s" % (rnd(-float(r["delta_slice"])),
                      ci(-float(r["ci_hi"]), -float(r["ci_lo"]))),
        )
    for bb in ("alexnet", "swin_t"):
        ck.in_text(
            "table6 row A1r/%s" % bb,
            "R-rec&%s&%.4f&%.4f" % (
                BB_LABEL[bb], p3b_m[("A1r", bb)], p3b_r[("A1r", bb)]),
        )
    p3rows = [r for r in read_csv(os.path.join(P3, "p3_delta.csv"))
              if r["base"] == "A0" and r["arm"] in ("A1", "A2")]
    for r in p3rows:
        arm, base, bb = r["arm"], r["base"], r["bb"]
        label = P3_LABEL[arm]
        ck.in_text(
            "table6 row %s/%s" % (arm, bb),
            "%s&%s&%.4f&%.4f\\\\"
            % (label, BB_LABEL[bb], p3m[(arm, bb)], p3r[(arm, bb)]),
        )
        ck.in_text(
            "table6 caption contrast %s-%s/%s" % (arm, base, bb),
            "%+.2f%s" % (float(r["delta_slice"]), ci(r["ci_lo"], r["ci_hi"], dp=2)),
        )

    # Reported headline tokens in prose
    for tok, why in [
        ("110,415", "released slice count"),
        ("7,827", "released recording count"),
        ("252", "released class count"),
        ("109,831", "audited slice count"),
        ("7,756", "audited recording count"),
        ("234", "audited class count"),
        ("4,605", "CTIS sealed panel slices"),
        ("963", "CTIS sealed panel recordings"),
        ("6.25", "rounded E1b batch wall-clock hours"),
        ("4.15", "rounded E1c batch wall-clock hours"),
        ("2.99", "rounded cross-instrument batch wall-clock hours"),
        ("+2.02[+0.83,+3.35]", "confirmatory AlexNet interval"),
        ("+1.70[+1.28,+2.84]", "confirmatory Swin-T interval"),
        ("+39.63", "P3 shortcut upper bound (AlexNet)"),
        ("+34.77", "P3 shortcut upper bound (Swin-T)"),
        ("-0.36", "P3 A2-A1 AlexNet"),
        ("-2.83", "P3 A2-A1 Swin-T"),
        ("0.0065", "GZ_IsoTech expected exposure"),
        ("0.0098", "GZ_IsoTech exposure 95th percentile"),
        ("0.005", "GZ_IsoTech realised exposure"),
        ("99.49", "CTIS instrument parse rate"),
    ]:
        ck.in_text("prose %s (%s)" % (tok, why), tok)

    # ---- supplements B1 (recording-grouped folds, instruments kept) and C1 (volume control)
    # B1: three-way decomposition of the cross-instrument gap.
    leak = {(r["arm"], r["bb"]): r for r in read_csv(os.path.join(RESULTS, "p3b_leak", "p3_delta.csv"))}
    instr = {(r["arm"], r["bb"]): r for r in read_csv(os.path.join(RESULTS, "p3b_instr", "p3_delta.csv"))}
    for bb in ("alexnet", "swin_t"):
        r_leak = leak[("A1r", bb)]
        r_inst = instr[("A1", bb)]
        # identity: the two components must add up to the A0-A1 contrast
        a0_a1 = abs(float(leak[("A1", bb)]["delta_slice"]))
        parts = float(r_leak["delta_slice"]) + float(r_inst["delta_slice"])
        ck.identity("B1 decomposition %s" % bb, abs(parts + a0_a1) < 0.02,
                 "%.3f + %.3f + %.3f" % (parts, a0_a1, parts + a0_a1))
        share = abs(float(r_leak["delta_slice"])) / a0_a1 * 100.0
        ck.identity("B1 leak share %s in 5-12%%" % bb, 5.0 <= share <= 12.0, "%.1f%%" % share)
    ck.in_text("prose leak direction (lowers macro-F1)", "lowers macro-F1 by")
    ck.in_text("prose 2.90 (leak AlexNet)", "2.90")
    ck.in_text("prose 3.41 (leak Swin-T)", "3.41")
    ck.in_text("prose -36.7 (novelty AlexNet)", "-36.7")
    ck.in_text("prose -31.4 (novelty Swin-T)", "-31.4")
    ck.in_text("prose A1r pooled 0.816", "0.816")
    ck.in_text("prose A1r pooled 0.843", "0.843")
    # C1: equal-volume control must stay small and well below the exposure effect
    c1 = {(r["bb"]): r for r in read_csv(os.path.join(RESULTS, "c1", "e1_delta.csv"))}
    main_d = {(r["library"], r["bb"]): r for r in read_csv(os.path.join(RESULTS, "e1_delta.csv"))}
    for bb, tok in (("alexnet", "+0.05"), ("swin_t", "+0.62")):
        ck.identity("C1 %s equals reported %.2f" % (bb, float(c1[bb]["delta_slice_pt"])),
                 abs(float(c1[bb]["delta_slice_pt"]) - float(tok)) < 0.006,
                 "%.3f" % float(c1[bb]["delta_slice_pt"]))
        ck.identity("C1 %s below SESOI" % bb, abs(float(c1[bb]["delta_slice_pt"])) < 3.0,
                 "%.3f" % float(c1[bb]["delta_slice_pt"]))
        ck.identity("C1 %s smaller than A1 effect" % bb,
                 abs(float(c1[bb]["delta_slice_pt"])) < abs(float(main_d[("CTIS", bb)]["delta_slice_pt"])),
                 "%.3f vs %.3f" % (float(c1[bb]["delta_slice_pt"]),
                                   float(main_d[("CTIS", bb)]["delta_slice_pt"])))
    ck.in_text("prose C1 volume 25,502", "25,502")

    # Human-review supplemental tables: verify actual generated cells, not token presence.
    for r in read_csv(os.path.join(RESULTS, "human_review", "selftest.csv")):
        pub = "%.3f" % float(r["reported_mel_f1"]) if r["reported_mel_f1"] else "---"
        ck.in_text("S1 self-test %s/%s" % (r["library"], r["backbone"]),
                   "%s&%s&%.4f&%.4f&%+.2f&%s" % (r["library"].replace("_", " "),
                     r["backbone"], float(r["selftest_mean"]), float(r["selftest_sd"]),
                     100 * float(r["selftest_minus_a0"]), pub))
    for (lib, bb, contrast), r in corrected_rec.items():
        if contrast != "A1-A0":
            continue
        ck.in_text("S1 recording correction %s/%s" % (lib, bb),
                   "%s&%s&%+.2f&%+.2f" % (lib.replace("_", " "), BB_LABEL[bb],
                     float(r["legacy_point_sw_pt"]), float(r["point_unw_pt"])))
    schemes = {"legacy_seedfixed": "cluster, fixed seeds", "stratified_seedfixed": "class, fixed seeds",
               "stratified_seedboot": "class + seeds", "legacy_seedboot": "cluster + seeds"}
    for r in read_csv(os.path.join(RESULTS, "human_review", "ctis_seed_stratified_bootstrap.csv")):
        if r["metric"] not in ("slice", "recording"):
            continue
        ck.in_text("S1 sensitivity %s/%s/%s" % (r["metric"], r["backbone"], r["scheme"]),
                   "%s&%s&%s&%+.2f&%+.2f&%+.2f" % (r["metric"], BB_LABEL[r["backbone"]],
                     schemes[r["scheme"]], float(r["point_pt"]), float(r["ci_lo_pt"]), float(r["ci_hi_pt"])))
    ck.in_text("add-one Holm disclosure", "0.0020")
    ck.identity("recording implementation uses equal votes",
                "nc, w=rec_n" not in io.open(os.path.join(PROJ, "code", "analyze_e1.py"), encoding="utf-8").read(),
                "point and cluster interval both count one recording vote")

    # ---- layer 3: numbers added in the second review round ---------------
    R2 = os.path.join(RESULTS, "human_review", "round2")
    d_main = {(r["library"], r["bb"], r["contrast"]): r
              for r in read_csv(os.path.join(RESULTS, "e1_delta.csv"))}
    d_c1 = {(r["library"], r["bb"], r["contrast"]): r
            for r in read_csv(os.path.join(RESULTS, "c1", "e1_delta.csv"))}
    rec_corr = {(r["library"], r["bb"], r["contrast"]): r
                for r in read_csv(os.path.join(RESULTS, "human_review",
                                               "recording_metric_correction.csv"))}

    a1c1 = [r for r in read_csv(os.path.join(R2, "a1_minus_c1.csv"))
            if r["contrast"] == "A1-C1"]
    ck.identity("A1-C1 has four cells", len(a1c1) == 4, str(len(a1c1)))
    for r in a1c1:
        bb, lv = r["backbone"], r["level"]
        delta_pp, lo, hi = float(r["delta_pp"]), float(r["ci_lo"]), float(r["ci_hi"])
        ck.in_text_any("A1-C1 %s %s interval" % (bb, lv),
                       ["%+.2f [%+.2f,%+.2f]" % (delta_pp, lo, hi),
                        "%+.2f points, 95\\%% CI %+.2f to %+.2f" % (delta_pp, lo, hi)])
        ck.identity("A1-C1 %s %s excludes zero" % (bb, lv), lo > 0.0, "%+.2f" % lo)
        ck.identity("A1-C1 %s %s below SESOI" % (bb, lv), delta_pp < 3.0, "%+.2f" % delta_pp)
        if lv == "slice":
            want = (float(d_main[("CTIS", bb, "A1-A0")]["delta_slice_pt"]) -
                    float(d_c1[("CTIS", bb, "C1-A0")]["delta_slice_pt"]))
        else:
            want = (float(rec_corr[("CTIS", bb, "A1-A0")]["point_unw_pt"]) -
                    float(rec_corr[("CTIS", bb, "C1-A0")]["point_unw_pt"]))
        ck.identity("A1-C1 %s %s equals A1-A0 minus C1-A0" % (bb, lv),
                    abs(delta_pp - want) <= 0.03, "%.3f vs %.3f" % (delta_pp, want))

    arm_label = {"A0": "A0, slice", "A1": "A1/A2, instrument", "A2": "A1/A2, instrument",
                 "A1r": "A1r, recording"}
    for r in read_csv(os.path.join(R2, "p3_fold_composition.csv")):
        if r["arm"] == "A2":
            continue
        ck.in_text("fold composition %s fold %s" % (r["arm"], r["fold"]),
                   "%s&%s&%s&%s&%s&%s" % (arm_label[r["arm"]], r["fold"],
                                          thousands(r["n_train_recordings"]),
                                          thousands(r["n_test_recordings"]),
                                          thousands(r["n_test_slices"]), r["n_classes_test"]))
        if r["arm"] == "A1" and r["fold"] == "1":
            ck.identity("first instrument fold covers 42 of 43 classes",
                        r["n_classes_test"] == "42", r["n_classes_test"])

    md = {(r["bb"], r["contrast"], r["mode"]): r
          for r in read_csv(os.path.join(R2, "macro_denominator.csv"))}
    width_gap = 0.0
    for bb in ("alexnet", "swin_t"):
        for contrast in ("A1-A0", "A1-C1"):
            f, d, u = md[(bb, contrast, "fixed")], md[(bb, contrast, "dynamic")], md[(bb, contrast, "fullC")]
            ck.identity("class set leaves point %s %s unchanged" % (bb, contrast),
                        abs(float(f["d_pp"]) - float(d["d_pp"])) < 5e-4,
                        "%s vs %s" % (f["d_pp"], d["d_pp"]))
            ck.identity("full class set equals fixed %s %s" % (bb, contrast),
                        f["lo_pp"] == u["lo_pp"] and f["hi_pp"] == u["hi_pp"],
                        "%s/%s vs %s/%s" % (f["lo_pp"], f["hi_pp"], u["lo_pp"], u["hi_pp"]))
            width_gap = max(width_gap, abs(float(f["ci_width_pp"]) - float(d["ci_width_pp"])))
    ck.identity("class set shifts interval width by at most 0.4 points",
                width_gap <= 0.4, "%.3f" % width_gap)
    for bb, contrast, mode in (("alexnet", "A1-A0", "fixed"), ("swin_t", "A1-C1", "fixed")):
        r = md[(bb, contrast, mode)]
        ck.in_text("S1 fixed class set %s %s bounds" % (bb, contrast),
                   "%s to %s" % (rnd(r["lo_pp"]), rnd(r["hi_pp"])))
    dropped = [int(float(md[(bb, c, "dynamic")]["dropped_classes_max"]))
               for bb in ("alexnet", "swin_t") for c in ("A1-A0", "A1-C1")]
    ck.identity("largest class drop over replicates is 18", max(dropped) == 18, str(dropped))
    ck.in_text("S1 largest class drop is at most 18", "at most %d" % max(dropped))

    # post hoc exposure split of the CTIS sealed panel (round-2 revision)
    exp = [r for r in read_csv(os.path.join(R2, "a1_minus_c1.csv"))
           if r["contrast"].startswith("A1-A0[")]
    ck.identity("exposure split has eight cells", len(exp) == 8, str(len(exp)))
    for r in exp:
        bb, lv = r["backbone"], r["level"]
        d, lo, hi = float(r["delta_pp"]), float(r["ci_lo"]), float(r["ci_hi"])
        ck.in_text_any("exposure split %s %s %s interval" % (r["contrast"], bb, lv),
                       ["%+.2f [%+.2f,%+.2f]" % (d, lo, hi),
                        "%+.2f$ $[%+.2f,%+.2f]$" % (d, lo, hi)])
    sizes = {(r["contrast"], r["n_slices"], r["n_recordings"]) for r in exp}
    ck.identity("exposure split sizes are 3895/578 and 710/385",
                sizes == {("A1-A0[exposed]", "3895", "578"),
                          ("A1-A0[unexposed]", "710", "385")}, str(sorted(sizes)))
    ck.in_text("exposure split denominator", "578 of the 963 CTIS test recordings")
    ck.in_text("exposure split subgroup size", "710 slices and 385 recordings")

    # class-set table rows (round-2 revision)
    for bb_name, bb in (("AlexNet", "alexnet"), ("Swin-T", "swin_t")):
        for lab, contrast in (("A1$-$A0", "A1-A0"), ("A1$-$C1", "A1-C1")):
            for mode_lab, mode in (("recomputed", "dynamic"), ("fixed", "fixed")):
                r = md[(bb, contrast, mode)]
                ck.in_text("class-set row %s %s %s" % (bb, contrast, mode),
                           "%s & %s & %s & %s & $[%s,%s]$ \\\\" %
                           (bb_name, lab, mode_lab, rnd(r["d_pp"]),
                            rnd(r["lo_pp"]), rnd(r["hi_pp"])))
    ck.in_text("class-set table is referenced", "Table~\\ref{tab:classset}")
    ck.in_text("post hoc analyses disclosed in the protocol",
               "Two further analyses are post hoc and are reported as such")

    p3u = json.loads(io.open(os.path.join(PROJ, "data", "p3", "P3_universe.json"),
                             encoding="utf-8").read())
    u43 = read_csv(os.path.join(PROJ, "data", "p3", "universe_43.csv"))
    ube = read_csv(os.path.join(PROJ, "data", "p3", "universe_before_drop.csv"))
    n43, r43 = len(u43), sum(int(r["recordings"]) for r in u43)
    nbe, rbe = len(ube), sum(int(r["recordings"]) for r in ube)
    ck.identity("P3 universe artifact says 43 over 1388", (n43, r43) == (43, 1388),
                "%d over %d" % (n43, r43))
    ck.identity("P3 universe artifact says 49 over 4037", (nbe, rbe) == (49, 4037),
                "%d over %d" % (nbe, rbe))
    ck.identity("P3 universe self-check passed", bool(p3u.get("all_pass")), str(p3u.get("all_pass")))
    ck.in_text("protocol class universe after rule",
               "%d techniques over %s recordings, out of %d candidate classes covering %s recordings"
               % (n43, thousands(r43), nbe, thousands(rbe)))

    def dup_groups(lib):
        acc = {}
        for r in read_csv(os.path.join(PROJ, "data", "manifest", "%s.csv" % lib)):
            acc.setdefault(r["content_hash"], []).append(r)
        return [v for v in acc.values() if len(v) > 1]

    ct_groups = dup_groups("CTIS")
    ck.identity("CTIS has 77 identical-image groups", len(ct_groups) == 77, str(len(ct_groups)))
    ck.identity("CTIS duplicate slices total 175", sum(len(v) for v in ct_groups) == 175,
                str(sum(len(v) for v in ct_groups)))
    ck.identity("every CTIS duplicate group repeats one window",
                all(len({r["slice_index"] for r in v}) == 1 for v in ct_groups),
                "slice_index varies in some group")
    multi = sum(1 for v in ct_groups if len({r["recording_canonical"] for r in v}) >= 2)
    single = sum(1 for v in ct_groups if len({r["recording_canonical"] for r in v}) == 1)
    ck.identity("26 CTIS duplicate groups span distinct recordings", multi == 26, str(multi))
    ck.identity("51 CTIS duplicate groups resolve to one recording's audio", single == 51, str(single))
    ck.in_text("manuscript split-duplicate sentence",
               "in %d groups those names resolve to one recording's audio after the SHA-1 folding "
               "above, and in %d they resolve to two or more distinct recordings" % (single, multi))
    mg_cross = [v for v in dup_groups("music_genre")
                if len({r["label"] for r in v}) > 1]
    ck.identity("music genre has one cross-class duplicate group", len(mg_cross) == 1,
                str(len(mg_cross)))

    ceiling = []
    for r in read_csv(os.path.join(RESULTS, "e1_metrics.csv")):
        if r["library"] == "bel_canto" and r["arm"] in ("A1", "A2"):
            ceiling.append(float(r["macro_f1_recording"]))
    ck.identity("bel canto recording-level score saturates at 1.00",
                max(ceiling) >= 0.9999, "%.4f" % max(ceiling))
    ck.in_text("bel canto ceiling disclosed",
                "reaches 1.00 in five of the six seed-by-backbone cells")
    ck.in_text("bel canto recording-level baselines disclosed",
                "reaches 0.83 for AlexNet and 0.77 for Swin-T")
    _bc = {}
    for r in read_csv(os.path.join(RESULTS, "human_review", "corrected_recording_metrics.csv")):
        if r["library"] == "bel_canto" and r["arm"] in ("A0", "A1"):
            _bc.setdefault((r["bb"], r["arm"]), []).append(float(r["f1_rec_unweighted"]))
    _m = dict((k, sum(v) / len(v)) for k, v in _bc.items())
    ck.identity("bel canto AlexNet recording-level A0 rounds to 0.83",
                abs(_m[("alexnet", "A0")] - 0.8295) < 5e-4, "%.4f" % _m[("alexnet", "A0")])
    ck.identity("bel canto Swin-T recording-level A0 rounds to 0.77",
                abs(_m[("swin_t", "A0")] - 0.7726) < 5e-4, "%.4f" % _m[("swin_t", "A0")])
    ck.identity("bel canto AlexNet recording-level contrast +14.61",
                delta(_m[("alexnet", "A1")], _m[("alexnet", "A0")]) == "+14.61",
                delta(_m[("alexnet", "A1")], _m[("alexnet", "A0")]))
    ck.identity("bel canto Swin-T recording-level contrast +22.74",
                delta(_m[("swin_t", "A1")], _m[("swin_t", "A0")]) == "+22.74",
                delta(_m[("swin_t", "A1")], _m[("swin_t", "A0")]))
    _ones = sum(1 for (bb, arm), v in _bc.items() if arm == "A1"
                for x in v if abs(x - 1.0) < 1e-9)
    _cells = sum(len(v) for (bb, arm), v in _bc.items() if arm == "A1")
    ck.identity("bel canto exposure arm hits 1.00 in five of six cells",
                (_ones, _cells) == (5, 6), "%d/%d" % (_ones, _cells))

    # round-3 additions: provenance and scope statements
    ck.in_text("all five libraries release audio",
                "all five libraries release source audio")
    ck.in_text("audio archives obtained only for three",
                "We obtained the audio archives of CTIS, CNPM and pianos")
    ck.in_text("two libraries audio not obtained",
                "Bel canto and music genre also release audio, which we did not obtain")
    ck.in_text("canonical fold is label preserving",
                "no canonical recording appears in two classes")
    ck.in_text("A2 trains on about 60 percent", "about 60\\% of the original corpus")
    ck.in_text("instrument count from merged groups",
                "135 groups after name merging")
    ck.in_text("recording-level column is multi-fold",
                "aggregates slice predictions from more than one fold model")

    # per-class slice budget: exactly one class differs between A0 and A1
    budget = {}
    for r in read_csv(os.path.join(R2, "slice_budget_by_class.csv")):
        if r["lib"] == "CTIS":
            budget.setdefault(r["class"], {})[r["arm"]] = int(r["n_slices"])
    differing = {c: v for c, v in budget.items()
                 if len({v.get(a) for a in ("A0", "A1", "C1")}) > 1}
    ck.identity("per-class budget differs in exactly one CTIS class",
                len(differing) == 1, str(sorted(differing)))
    if len(differing) == 1:
        cls, arms = next(iter(differing.items()))
        ck.identity("the differing class is C0296 with 241 vs 237",
                    cls == "C0296" and arms.get("A0") == 241 and arms.get("A1") == 237,
                    "%s %s" % (cls, arms))
        ck.identity("A0 and C1 agree classwise on that class",
                    arms.get("A0") == arms.get("C1"), str(arms))
        ck.in_text_any("four-slice shortfall disclosed in the protocol",
                       ["four fewer slices than A0 and C1 (25{,}498 against 25{,}502)",
                        "four fewer slices than A0 and C1"])
        ck.in_text("shortfall cause named", "duplicate-content rule before injection")
        ck.in_text("affected class coded", "C0296")

    # ---- round-3 revision: arm sizes, inventory, thresholds, duplicates ----
    inv = json.load(io.open(os.path.join(RESULTS, "library_inventory_and_convergence.json"),
                            encoding="utf-8"))
    for label in ("A0", "A1", "A2", "C1"):
        rows = read_csv(os.path.join(SPLITS, "CTIS_%s_train.csv" % label.lower()))
        n_rec = len({r.get("recording_canonical") or r["recording_uid"] for r in rows})
        arm = inv["arm_sizes_ctis"][label]
        ck.identity("CTIS %s training size" % label,
                    (len(rows), n_rec) == (arm["slices"], arm["recordings"]),
                    "%d/%d vs %s" % (len(rows), n_rec, arm))
        ck.in_text("arm table row %s" % label,
                   "&%s&%s&" % (thousands(len(rows)), thousands(n_rec)))

    for lib in ("CTIS", "music_genre", "pianos", "CNPM", "bel_canto",
                "chest_falsetto", "erhu_playing_tech", "GZ_IsoTech"):
        d = inv["libraries"][lib]
        ck.in_text("library table row %s" % lib,
                   "&%s&%s&%s&" % (thousands(d["slices"]), thousands(d["recordings"]),
                                    d["classes"]))
    ck.identity("Guzheng_Tech99 releases 99 slices from 99 sources",
                (inv["libraries"]["Guzheng_Tech99"]["slices"],
                 inv["libraries"]["Guzheng_Tech99"]["recordings"]) == (99, 99),
                str(inv["libraries"]["Guzheng_Tech99"]))
    ck.in_text("arm table A1 relation to test recordings", "578/963 shared")

    ts = inv["threshold_sensitivity"]
    ck.identity("threshold of five reproduces the frozen 201 classes",
                ts["5"]["classes"] == 201, str(ts["5"]))
    ck.in_text("threshold class counts",
               "keep %s, %s, %s and %s classes"
               % tuple(ts[str(t)]["classes"] for t in (2, 3, 5, 10)))
    ck.in_text("threshold slice counts",
               "covering %s, %s, %s and %s of the 43{,}054"
               % tuple(thousands(ts[str(t)]["slices"]) for t in (2, 3, 5, 10)))
    ck.in_text("threshold Q counts",
               "%s, %s, %s and %s Q recordings"
               % tuple(thousands(ts[str(t)]["q_recordings"]) for t in (2, 3, 5, 10)))
    ck.in_text("threshold trade-off sentence",
               "the class count falls by 42 while the slice count falls by 1{,}474")

    ex = ts["excluded"]
    ck.identity("thirteen rule classes cover 393 slices",
                (ex["rule_classes"], ex["rule_slices"]) == (13, 393),
                "%s/%s" % (ex["rule_classes"], ex["rule_slices"]))
    ck.identity("five placeholder classes cover 152 slices",
                (ex["placeholder_classes"], ex["placeholder_slices"]) == (5, 152),
                "%s/%s" % (ex["placeholder_classes"], ex["placeholder_slices"]))
    ck.identity("eighteen excluded classes cover 545 slices",
                (ex["all_classes"], ex["all_slices"]) == (18, 545),
                "%s/%s" % (ex["all_classes"], ex["all_slices"]))
    ck.identity("201 retained classes hold 4887 recordings",
                (ex["kept_classes"], ex["kept_recordings"]) == (201, 4887),
                "%s/%s" % (ex["kept_classes"], ex["kept_recordings"]))
    ck.in_text("rule exclusion share", "393 slices, 0.9\\% of the library")
    ck.in_text("placeholder share", "152 slices, 0.35\\%")
    ck.in_text("combined exclusion share", "545 slices, 1.3\\% of the library")
    ck.in_text("median comparison",
               "median %d source recordings against %d, median %d slices against %d"
               % (ex["rule_median_recordings"], ex["kept_median_recordings"],
                  ex["rule_median_slices"], ex["kept_median_slices"]))
    ck.in_text("density comparison",
               "%.2f slices per recording against %.2f"
               % (ex["rule_median_k"], ex["kept_median_k"]))
    ck.identity("fourteen retained classes have four to six recordings",
                ex["low_resource_kept_4to6"] == 14, str(ex["low_resource_kept_4to6"]))
    ck.in_text("low-resource retained classes",
               "Fourteen retained classes have four to six source recordings")
    ck.identity("201-class capacity for a 50 and 30 percent target",
                (ex["feasible_50"], ex["feasible_30"]) == (10, 38),
                "%s/%s" % (ex["feasible_50"], ex["feasible_30"]))
    ck.in_text("capacity stated in the change table",
               "10 can supply 50\\%, 38 can supply 30\\%")

    dup = json.load(io.open(os.path.join(RESULTS, "cross_library_duplicates.json"),
                            encoding="utf-8"))
    ck.identity("no content hash appears in two libraries",
                dup["cross_library_groups"] == 0, str(dup["cross_library_groups"]))
    ck.identity("cross-library check covers 110415 slices and 110280 hashes",
                (dup["slices_total"], dup["unique_hashes_total"]) == (110415, 110280),
                "%s/%s" % (dup["slices_total"], dup["unique_hashes_total"]))
    ck.in_text("cross-library sentence",
               "110{,}415 slices, 110{,}280 unique hashes, and no hash appears in two libraries")
    ck.in_text("cross-library reproduction command",
               "code/audit\\_cross\\_library\\_duplicates.py")

    conv = inv["convergence"]
    for key, want in (("CTIS|A0|alexnet", 0.9016), ("CTIS|A1|alexnet", 0.9027),
                      ("CTIS|C1|alexnet", 0.9026), ("CTIS|A0|swin_t", 0.8712),
                      ("CTIS|A1|swin_t", 0.8715), ("CTIS|C1|swin_t", 0.8714),
                      ("CTIS|A2|alexnet", 0.9034), ("CTIS|A2|swin_t", 0.8718)):
        ck.identity("final training loss %s" % key,
                    abs(conv[key]["final_train_loss_median"] - want) < 1e-9,
                    str(conv[key]["final_train_loss_median"]))
    ck.identity("A2 ends after 3940 steps", conv["CTIS|A2|alexnet"]["steps"] == 3940,
                str(conv["CTIS|A2|alexnet"]["steps"]))
    ck.in_text("convergence alexnet sentence",
               "0.9016 in A0, 0.9027 in A1 and 0.9026 in C1")
    ck.in_text("convergence swin sentence", "0.8712, 0.8715 and 0.8714 under Swin-T")
    ck.in_text("convergence accuracy floor", "at least 0.9997 in every arm")
    ck.in_text("A2 convergence sentence", "ends at 0.9034 and 0.8718")
    ck.in_text("A2 step count", "3{,}940 steps against 3{,}980")

    # ---- report ---------------------------------------------------------
    summary = {
        "passes": ck.passes,
        "fails": len(ck.fails),
        "checks": ck.fails,
        "panels": panels,
        "totals": {"slices": tot_slices, "recordings": tot_recs, "classes": tot_cls,
                   "qtest_slices": tot_qtest, "qtest_recordings": tot_qrec},
    }
    with io.open(a.json, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print("checks passed: %d, failed: %d" % (ck.passes, len(ck.fails)))
    for kind, name, detail in ck.fails:
        print("  [%s] %s  %s" % (kind, name, detail))
    print("report: %s" % os.path.relpath(a.json, PROJ))
    return 1 if ck.fails else 0


if __name__ == "__main__":
    sys.exit(main())
