"""A2 self-test scores against the MEL F1 values reported by the CCMusic paper.

Independent, read-only. Reads the sealed predictions already on disk
(``runs/sealed_eval/<lib>_A2_<bb>/seed*_a2_selftest.npz``) plus the metric table
``data/results/e1_metrics.csv``, recomputes macro-F1 per seed, and joins the result
to the F1 values printed in the version of record of the audited framework
(TISMIR 8(1):22-38, doi:10.5334/tismir.194).

Only cells whose dataset, label task and input representation could be verified
against the published table are filled. Everything else stays empty with a reason
in the ``note`` column. Nothing here is a claim that the reconstruction is
faithful: these are point estimates from three seeds, with no interval.

Usage
-----
    python code/human_review_selftest.py                 # writes the CSV, prints report
    python code/human_review_selftest.py --check-only    # verify seeds only, no file

Exit code 0 on success, 1 if any library x backbone cell does not hold exactly
three seeds (the check this script exists to make).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
SEALED = PROJ / "runs" / "sealed_eval"
METRICS = PROJ / "data" / "results" / "e1_metrics.csv"
OUT_DIR = PROJ / "data" / "results" / "human_review"
OUT_CSV = OUT_DIR / "selftest.csv"

sys.path.insert(0, str(PROJ / "code"))
from analyze_e1 import conf_slice, load_npz, macro_f1  # noqa: E402

LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto"]
BBS = [("alexnet", "AlexNet"), ("swin_t", "Swin-T")]

EXPECTED_SEEDS = 3

# What the published F1 values measure, stated as carefully as the source allows.
# The paper says (printed p.26): "we randomly split the dataset into training,
# validation, and testing sets at a ratio of 8:1:1", immediately after the audio
# clip count, and it names no grouping constraint anywhere. It does not publish the
# split itself, so which slices the reported numbers were computed on cannot be
# verified; arm A2 here is this audit's own reconstruction of the stated protocol,
# not a copy of theirs. Every comparison below is therefore "their number under
# their stated protocol" against "our number under our reconstruction of it".
REPORTED_PROTOCOL = ("paper states a random 8:1:1 split over audio clips, names no "
                     "grouping constraint, publishes no split file; arm A2 here is a "
                     "reconstruction of the stated protocol, not a replay of their split")

# Label space of each library, verified against the version of record. Quoted
# evidence is in the module README; page numbers are printed page numbers.
LABEL_SPACE = {
    "CTIS": ("instrument recognition; 219 labels = 209 instrument types plus variants "
             "(seven instruments with two variants, Yangqin with four), section 3.1.2 "
             "p.26; this audit keeps 201 of the 219"),
    "CNPM": ("mode classification; five pentatonic modes, 287 recordings, section 3.5; "
             "this audit keeps 5 classes and 286 of 287 recordings"),
    "bel_canto": ("singing style classification; four styles (bel canto male/female, "
                  "folk singing male/female), section 3.6; this audit keeps 4 classes "
                  "and 202 of 203 recordings"),
    "music_genre": "not described as a CCMusic task; no result table published",
    "pianos": "not described as a CCMusic task; no result table published",
}

# Values transcribed from the version of record of the audited framework:
# Zhou et al., "CCMusic: An open and diverse database for Chinese music
# information retrieval research", TISMIR 8(1):22-38 (2025),
# doi:10.5334/tismir.194. Printed page = PDF page index + 21. Column = "Mel".
# Table 3 body and caption are both on printed p.33 (the section text that
# introduces them is on p.32; the earlier note that said p.32 was wrong).
REPORTED = {
    ("CTIS", "AlexNet"): dict(table="Table 3", page=33, dataset="CTIS", mel=0.936,
                              task="Instrument recognition"),
    ("CTIS", "Swin-T"): dict(table="Table 3", page=33, dataset="CTIS", mel=0.956,
                             task="Instrument recognition"),
    ("bel_canto", "AlexNet"): dict(table="Table 8", page=35,
                                   dataset="Bel Canto & Chinese Folk Singing", mel=0.919,
                                   task="Singing style classification"),
    ("bel_canto", "Swin-T"): dict(table="Table 8", page=35,
                                  dataset="Bel Canto & Chinese Folk Singing", mel=0.906,
                                  task="Singing style classification"),
    ("CNPM", "AlexNet"): dict(table="Table 7", page=35, dataset="CNPM", mel=0.742,
                              task="Mode classification"),
}

# Every library x backbone pair that has no verified published counterpart.
NO_REPORTED = {
    ("CNPM", "Swin-T"): "Swin-T is not among the seven backbones in the published CNPM table "
                        "(Table 7 lists ViT-L-32, ViT-L-16, VGG11-BN, RegNet-Y-16GF, "
                        "Wide-ResNet50-2, AlexNet, ShuffleNet-V2-X2.0)",
    ("music_genre", "AlexNet"): "the audited framework publishes no music-genre result table",
    ("music_genre", "Swin-T"): "the audited framework publishes no music-genre result table",
    ("pianos", "AlexNet"): "the audited framework publishes no piano result table; the piano "
                           "panel here is an eight-way brand task",
    ("pianos", "Swin-T"): "the audited framework publishes no piano result table; the piano "
                          "panel here is an eight-way brand task",
}


def read_metrics():
    """(library, arm, backbone) -> macro-F1 at slice level, mean over seeds.

    ``e1_metrics.csv`` has one row per seed; the mean is taken here rather than
    being read from a summary file, so the aggregation is visible.
    """
    vals = {}
    with METRICS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["library"], row["arm"], row["bb"])
            vals.setdefault(key, []).append(float(row["macro_f1_slice"]))
    return {k: statistics.fmean(v) for k, v in vals.items()}


def n_classes(lib):
    """Class count actually used for the sealed panel, from the frozen universe."""
    path = PROJ / "data" / "splits" / "v2" / "class_universe.csv"
    if not path.exists():
        return None
    n = 0
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("library") == lib and row.get("kept", "").strip().lower() in ("true", "1"):
                n += 1
    return n


def selftest_per_seed(lib, bb):
    """macro-F1 on the A2 self-test panel, one value per seed, in seed order."""
    d = SEALED / f"{lib}_A2_{bb}"
    files = sorted(d.glob("seed*_a2_selftest.npz"))
    nc = n_classes(lib)
    if nc is None:
        raise SystemExit(f"class universe missing for {lib}; cannot score a confusion matrix")
    vals = []
    for p in files:
        probs, y, _ = load_npz(p)
        conf = conf_slice(y, probs.argmax(1), nc)
        vals.append(float(macro_f1(conf)[0]))
    return files, vals


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check-only", action="store_true", help="verify seeds, write nothing")
    args = ap.parse_args()

    metrics = read_metrics()
    rows = []
    bad = []

    for lib in LIBS:
        for bb, bb_label in BBS:
            files, vals = selftest_per_seed(lib, bb)
            n = len(vals)
            if n != EXPECTED_SEEDS:
                bad.append(f"{lib}/{bb_label}: found {n} seeds, expected {EXPECTED_SEEDS}")

            mean = statistics.fmean(vals) if vals else float("nan")
            sd = statistics.stdev(vals) if n > 1 else float("nan")
            a0 = metrics.get((lib, "A0", bb))
            a2 = metrics.get((lib, "A2", bb))

            rep = REPORTED.get((lib, bb_label))
            if rep is not None:
                venue = f"{rep['table']} p.{rep['page']} ({rep['dataset']})"
                mel = rep["mel"]
                d_vs_a0 = mel - a0 if a0 is not None else float("nan")
                d_vs_self = mel - mean
                comparability = (
                    "contextual only: same task and same label space up to dropped "
                    "classes, but the published split is not available, so this is not a "
                    "same-protocol validation"
                )
                note = f"paper task: {rep['task']}. Label space: {LABEL_SPACE[lib]}."
            else:
                venue = ""
                mel = float("nan")
                d_vs_a0 = float("nan")
                d_vs_self = float("nan")
                comparability = "no published value: nothing to compare"
                note = NO_REPORTED.get((lib, bb_label), "")

            rows.append(dict(
                library=lib,
                backbone=bb_label,
                n_seeds=n,
                seed_values=";".join(f"{v:.6f}" for v in vals),
                selftest_mean=round(mean, 6),
                selftest_sd=round(sd, 6) if n > 1 else "",
                a0_slice=round(a0, 6) if a0 is not None else "",
                a2_slice=round(a2, 6) if a2 is not None else "",
                selftest_minus_a0=round(mean - a0, 6) if a0 is not None else "",
                reported_source=venue,
                reported_mel_f1=mel if mel == mel else "",
                reported_minus_a0=round(d_vs_a0, 6) if d_vs_a0 == d_vs_a0 else "",
                reported_minus_selftest=round(d_vs_self, 6) if d_vs_self == d_vs_self else "",
                comparability=comparability,
                label_space=LABEL_SPACE.get(lib, ""),
                reported_protocol=REPORTED_PROTOCOL,
                note=note,
            ))

    if not args.check_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"{'library':<12}{'bb':<9}{'n':>3}{'self-test':>11}{'sd':>9}{'A0':>9}"
          f"{'reported':>10}{'rep-A0':>9}{'rep-self':>10}")
    for r in rows:
        def f(v, nd=4):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ("—" if v == "" else str(v))
        print(f"{r['library']:<12}{r['backbone']:<9}{r['n_seeds']:>3}"
              f"{f(r['selftest_mean']):>11}{f(r['selftest_sd']):>9}{f(r['a0_slice']):>9}"
              f"{f(r['reported_mel_f1'], 3):>10}{f(r['reported_minus_a0'], 3):>9}"
              f"{f(r['reported_minus_selftest'], 3):>10}")

    filled = sum(1 for r in rows if r["reported_mel_f1"] != "")
    print()
    print(f"cells: {len(rows)}; with a verified published value: {filled}; blank: {len(rows) - filled}")
    if not args.check_only:
        print(f"wrote {OUT_CSV.relative_to(PROJ)}")

    if bad:
        print("\nSEED CHECK FAILED:")
        for line in bad:
            print("  -", line)
        return 1
    print(f"\nseed check passed: every cell holds exactly {EXPECTED_SEEDS} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
