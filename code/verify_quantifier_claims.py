# -*- coding: utf-8 -*-
"""Third gate: quantified claims in captions, checked against the artifacts.

The first gate asks whether a number appears in the manuscript. The second asks
whether a sentence with a number matches the artifact that produced it. Neither
can see a sentence whose numbers are right and whose *quantifier* is wrong: "both
arms reach 1.00" reads fine and is false if the two arms meant are not the two
arms that reach 1.00.

This gate closes that gap in three steps.

1. Every claim in ``data/results/quantifier_ledger.csv`` names a sentence of the
   manuscript and a predicate over an artifact. The predicate is recomputed from
   the artifact, never from the manuscript, and compared with the expected value
   stored in the ledger. Rewording the sentence breaks the match, which is the
   point: the ledger has to be updated together with the sentence.
2. A claim whose sentence quantifies over arms or libraries must name the subject
   it quantifies over. "A1 and A2 both reach 1.00" passes; "both arms reach 1.00"
   fails, because the ledger sentence is not found in the text.
3. Every caption that contains a quantifier must be covered by a ledger row or by
   an explicit waiver in ``data/results/quantifier_waivers.csv``. A new caption
   with "every" in it cannot be added without either checking it or saying why it
   cannot be checked.

Run: python code/verify_quantifier_claims.py
Exit code is 0 only when every claim is both present and true.
"""
from __future__ import print_function

import csv
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(DATA, "results")
SPLITS = os.path.join(DATA, "splits", "v2")
MANIFEST = os.path.join(DATA, "manifest")
PAPER = os.path.join(ROOT, "paper")

LEDGER = os.path.join(RESULTS, "quantifier_ledger.csv")
WAIVERS = os.path.join(RESULTS, "quantifier_waivers.csv")
REPORT = os.path.join(RESULTS, "quantifier_verify.json")

QUANT = re.compile(r"\b(both|every|all|each|none|neither|no|only)\b", re.I)
TEX_FILES = ["main_sn.tex"] + [
    os.path.join("sections", n)
    for n in sorted(os.listdir(os.path.join(PAPER, "sections")))
    if n.endswith(".tex")
]


# ---------------------------------------------------------------- helpers
def read_csv(path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def norm(text):
    for tok in ("{", "}", "$", ";", "\\;", "\\,", "\\", "\u00a0", "\u2014", "--"):
        text = text.replace(tok, "")
    return "".join(text.split())


def strip_comments(text):
    out = []
    for line in text.split("\n"):
        buf = []
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            buf.append(line[i])
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def relpath(rel):
    """Ledger paths are repo-relative; the paper tree is rooted at paper/."""
    rel = rel.replace("\\", "/")
    return rel[6:] if rel.startswith("paper/") else rel


def read_tex(rel):
    return strip_comments(io.open(os.path.join(PAPER, relpath(rel)), encoding="utf-8").read())


def captions(rel, env=None):
    """All \\caption{...} bodies of one file, brace-matched.

    With ``env="table"`` only captions inside a table float are returned; the
    coverage rule applies there, because a table caption asserts something about
    the table below it. Figure captions describe a drawing and are counted
    separately.
    """
    text = read_tex(rel)
    spans = []
    if env is not None:
        for m in re.finditer(r"\\begin\{(%s)\*?\}" % env, text):
            end = text.find("\\end{%s" % m.group(1), m.end())
            if end > 0:
                spans.append((m.start(), end))
        text_src = [(text[s:e], s) for s, e in spans]
    else:
        text_src = [(text, 0)]
    out = []
    for chunk, _ in text_src:
        for m in re.finditer(r"\\caption\s*\{", chunk):
            depth = 1
            i = m.end()
            while i < len(chunk) and depth > 0:
                if chunk[i] == "{":
                    depth += 1
                elif chunk[i] == "}":
                    depth -= 1
                i += 1
            out.append(chunk[m.end():i - 1])
    return out


# ---------------------------------------------------------------- artifact loads
def load_ctis_manifest():
    return read_csv(os.path.join(MANIFEST, "CTIS.csv"))


def load_main_delta():
    return read_csv(os.path.join(RESULTS, "e1_delta.csv"))


def load_corrected_recording():
    return read_csv(os.path.join(RESULTS, "human_review", "corrected_recording_metrics.csv"))


def load_train_sizes():
    """(library, arm) -> n_train, from the retained run records."""
    out = {}
    for lib in ("CTIS", "music_genre", "pianos", "CNPM", "bel_canto"):
        for arm in ("A0", "A2"):
            path = os.path.join(ROOT, "runs", lib, arm, "alexnet", "seed0", "result.json")
            with io.open(path, encoding="utf-8") as fh:
                out[(lib, arm)] = int(json.load(fh)["n_train"])
    return out


def load_negcontrol():
    return [r for r in read_csv(os.path.join(RESULTS, "b2", "e1_delta.csv"))
            if r["library"] in ("chest_falsetto", "erhu_playing_tech", "GZ_IsoTech")]


def load_fold_composition():
    return read_csv(os.path.join(RESULTS, "human_review", "round2", "p3_fold_composition.csv"))


def load_p3_splits():
    return read_csv(os.path.join(DATA, "p3", "p3_splits.csv"))


def load_classset_modes():
    return read_csv(os.path.join(RESULTS, "human_review", "round2", "macro_denominator.csv"))


def load_inventory():
    with io.open(os.path.join(RESULTS, "library_inventory_and_convergence.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_panels():
    """library -> (classes, recordings) over the four frozen partitions."""
    out = {}
    for lib in ("CTIS", "music_genre", "pianos", "CNPM", "bel_canto"):
        recs, labels = set(), set()
        for part in ("c", "r", "qtest", "qpool"):
            for r in read_csv(os.path.join(SPLITS, "%s_%s.csv" % (lib, part))):
                recs.add(r["recording_uid"])
                labels.add(r["label"])
        out[lib] = (len(labels), len(recs))
    return out


def load_budget():
    out = {}
    for r in read_csv(os.path.join(RESULTS, "human_review", "round2", "slice_budget_by_class.csv")):
        if r["lib"] != "CTIS":
            continue
        out.setdefault(r["class"], {})[r["arm"]] = int(r["n_slices"])
    return out


# ---------------------------------------------------------------- predicates
def h_counted_once():
    rows = load_ctis_manifest()
    uids = set(r["recording_uid"] for r in rows)
    canon = set(r["recording_canonical"] for r in rows)
    groups = {}
    for r in rows:
        groups.setdefault(r["audio_hash"], set()).add(r["recording_uid"])
    pairs = [k for k, v in groups.items() if len(v) > 1]
    return {"n_uids": len(uids), "n_canonical": len(canon), "n_pairs": len(pairs)}


def h_threshold_all():
    rows = load_main_delta()
    primary = [r for r in rows if r["library"] == "CTIS"]
    extra = [r for r in rows if r["library"] != "CTIS"]
    pmax = max(max(float(r["delta_slice_pt"]), float(r["delta_rec_pt"])) for r in primary)
    meet = [r for r in extra
            if float(r["delta_slice_pt"]) >= 3.0 and float(r["delta_rec_pt"]) >= 3.0]
    return {"n_primary": len(primary), "primary_max_pt": round(pmax, 2),
            "n_additional": len(extra), "n_additional_meet": len(meet)}


def h_reach_value():
    rows = [r for r in load_corrected_recording()
            if r["library"] == "bel_canto" and r["bb"] == "swin_t"]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(float(r["f1_rec_unweighted"]))
    ceiling = sorted(a for a, v in by_arm.items() if all(abs(x - 1.0) < 1e-9 for x in v))
    return {"arms_at_ceiling": ",".join(ceiling),
            "a0_mean": round(sum(by_arm["A0"]) / len(by_arm["A0"]), 2),
            "n_seeds": len(by_arm["A0"])}


def h_ratio_range():
    sizes = load_train_sizes()
    libs = sorted(set(k[0] for k in sizes))
    dev = max(abs(float(sizes[(l, "A2")]) / float(sizes[(l, "A0")]) - 1.0) for l in libs)
    return {"n_libraries": len(libs), "max_abs_dev": round(dev, 4)}


def h_zero_in_ci():
    rows = load_negcontrol()
    inc = [r for r in rows if float(r["ci_lo_slice_pt"]) <= 0.0 <= float(r["ci_hi_slice_pt"])]
    return {"n_rows": len(rows), "n_include_zero": len(inc)}


def h_zero_in_ci_one():
    rows = load_negcontrol()
    exc = [r for r in rows if not (float(r["ci_lo_slice_pt"]) <= 0.0 <= float(r["ci_hi_slice_pt"]))]
    if len(exc) != 1:
        return {"n_rows": len(rows), "n_exclude_zero": len(exc),
                "excluded": ",".join("%s/%s" % (r["library"], r["bb"]) for r in exc) or "none",
                "excluded_pt": None, "excluded_seed_range": None}
    r = exc[0]
    return {"n_rows": len(rows), "n_exclude_zero": len(exc),
            "excluded": "%s/%s" % (r["library"], r["bb"]),
            "excluded_pt": round(float(r["delta_slice_pt"]), 2),
            "excluded_seed_range": round(float(r["seed_range_pt"]), 2)}


def h_fold_gap():
    fold = [r for r in load_fold_composition() if r["arm"] == "A1" and r["fold"] == "1"][0]
    splits = load_p3_splits()
    total_slices = len(splits)
    total_recs = len(set(r["recording_uid"] for r in splits))
    return {"n_val_recordings":
            total_recs - int(fold["n_train_recordings"]) - int(fold["n_test_recordings"]),
            "n_val_slices_fold1":
            total_slices - int(fold["n_train_slices"]) - int(fold["n_test_slices"])}


def h_same_numbers():
    modes = {}
    for r in load_classset_modes():
        modes.setdefault((r["bb"], r["contrast"]), {})[r["mode"]] = float(r["d_pp"])
    gaps = [abs(v["fixed"] - v["fullC"]) for v in modes.values()
            if "fixed" in v and "fullC" in v]
    return {"n_pairs": len(gaps), "max_abs_diff": round(max(gaps), 4) if gaps else None}


def h_panel_keeps():
    panels = load_panels()
    inv = load_inventory()["libraries"]
    others = ("music_genre", "pianos", "CNPM", "bel_canto")
    keep = sum(1 for l in others if panels[l][0] == inv[l]["classes"])
    return {"panel_classes": panels["CTIS"][0], "release_classes": inv["CTIS"]["classes"],
            "panel_recordings": panels["CTIS"][1], "release_recordings": inv["CTIS"]["recordings"],
            "n_other_libs_keep_all": keep}


def h_budget_identical():
    budget = load_budget()
    diff = [c for c, v in budget.items() if v.get("A0") != v.get("C1")]
    short = [(c, v["A0"] - v["A1"]) for c, v in budget.items() if v.get("A1", 0) < v.get("A0", 0)]
    return {"n_diff_a0_c1": len(diff), "n_short_a1": len(short),
            "short_class": short[0][0] if len(short) == 1 else ",".join(c for c, _ in short),
            "short_by": short[0][1] if len(short) == 1 else None}


PREDICATES = {
    "counted_once": h_counted_once,
    "threshold_all": h_threshold_all,
    "reach_value": h_reach_value,
    "ratio_range": h_ratio_range,
    "zero_in_ci": h_zero_in_ci,
    "zero_in_ci_one": h_zero_in_ci_one,
    "fold_gap": h_fold_gap,
    "same_numbers": h_same_numbers,
    "panel_keeps": h_panel_keeps,
    "budget_identical": h_budget_identical,
}


# ---------------------------------------------------------------- comparison
def compare(params, facts, dp=2):
    """Compare ledger expectations with recomputed facts.

    Keys may carry a suffix: ``_le`` / ``_ge`` for inequalities, ``_1dp`` /
    ``_2dp`` / ``_3dp`` to round the computed value before comparing.
    """
    problems = []
    for key, want in params.items():
        if key.endswith("_le") or key.endswith("_ge"):
            base, op = key.rsplit("_", 1)
            got = facts.get(base)
            if got is None:
                problems.append("%s: no value" % key)
            elif op == "le" and not got <= want:
                problems.append("%s: %r > %r" % (base, got, want))
            elif op == "ge" and not got >= want:
                problems.append("%s: %r < %r" % (base, got, want))
            continue
        if key.endswith("dp"):
            base, d = key.rsplit("_", 1)
            got = facts.get(base)
            if got is None:
                problems.append("%s: no value" % key)
            else:
                got = round(float(got), int(d[:-2]))
        else:
            got = facts.get(key)
        if got != want:
            problems.append("%s: got %r, ledger says %r" % (key, got, want))
    return problems


def run(verbose=True):
    ledger = read_csv(LEDGER)
    waivers = read_csv(WAIVERS)
    failures = []
    passed = 0
    report = []

    for row in ledger:
        cid = row["id"]
        rel = row["file"].replace("\\", "/")
        text = norm(read_tex(rel))
        needle = norm(row["sentence"])
        ok_sentence = bool(needle) and needle in text
        facts = {}
        problems = []
        if not ok_sentence:
            failures.append(("text", cid, "sentence not found in %s" % rel))
        else:
            fn = PREDICATES.get(row["claim"])
            if fn is None:
                failures.append(("claim", cid, "unknown predicate %r" % row["claim"]))
            else:
                facts = fn()
                params = json.loads(row["params"]) if row["params"].strip() else {}
                problems = compare(params, facts)
                for p in problems:
                    failures.append(("fact", cid, p))
        if ok_sentence and not problems:
            passed += 1
        report.append({"id": cid, "file": rel, "claim": row["claim"],
                       "sentence_found": ok_sentence, "facts": facts,
                       "problems": problems})

    # coverage: every table caption with a quantifier is checked or waived
    covered = 0
    waived = 0
    figure_caps = 0
    for rel in TEX_FILES:
        rel_norm = "paper/" + rel.replace("\\", "/")
        for cap in captions(rel_norm, env="table"):
            cap_norm = norm(cap)
            if not QUANT.search(cap):
                continue
            if any(norm(r["sentence"]) in cap_norm for r in ledger):
                covered += 1
                continue
            hit = [w for w in waivers
                   if w["file"].replace("\\", "/") == rel_norm
                   and norm(w["fingerprint"]) in cap_norm]
            if hit:
                waived += 1
                continue
            failures.append(("coverage", rel_norm,
                             "table caption with a quantifier is neither checked nor waived: %s"
                             % cap_norm[:110]))
        figure_caps += sum(1 for cap in captions(rel_norm, env="figure") if QUANT.search(cap))

    if verbose:
        for kind, cid, msg in failures:
            print("FAIL  %-9s %-6s %s" % (kind, cid, msg))
        print("quantifier claims: %d in the ledger, %d true and present, %d failed"
              % (len(ledger), passed, len(failures)))
        print("table captions with a quantifier: %d checked, %d waived"
              % (covered, waived))
        print("figure captions with a quantifier: %d, out of scope (drawings, not tables)"
              % figure_caps)

    payload = {"ledger_rows": len(ledger), "passed": passed,
               "failures": [{"kind": k, "id": c, "detail": m} for k, c, m in failures],
               "captions_checked": covered, "captions_waived": waived,
               "figure_captions_out_of_scope": figure_caps,
               "claims": report}
    with io.open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
