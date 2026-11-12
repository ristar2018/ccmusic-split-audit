# -*- coding: utf-8 -*-
"""Negative tests for the quantifier gate.

A gate that only ever prints PASS is indistinguishable from a gate that checks
nothing. Each case below breaks exactly one thing --- an artifact row, a
manuscript sentence, a caption --- and asserts that the gate notices, naming the
claim it should have caught. The first case is the control: nothing is broken and
the gate must stay green.

Run: python code/negtest_quantifier_claims.py
"""
from __future__ import print_function

import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_quantifier_claims as V  # noqa: E402

REAL_REPORT = V.REPORT
V.REPORT = os.path.join(tempfile.gettempdir(), "quantifier_negtest.json")

CASES = []


def load_failures():
    with io.open(V.REPORT, encoding="utf-8") as fh:
        payload = json.load(fh)
    return [(f["kind"], f["id"], f["detail"]) for f in payload["failures"]]


def case(fn):
    CASES.append(fn)
    return fn


def snapshot():
    return dict((n, getattr(V, n)) for n in
                ("load_main_delta", "load_corrected_recording", "read_tex", "captions"))


def restore(snap):
    for k, v in snap.items():
        setattr(V, k, v)


def expect(failures, kind, ident=None):
    for k, cid, msg in failures:
        if k == kind and (ident is None or cid == ident):
            return True, "%s %s: %s" % (k, cid, msg)
    return False, "no %s failure for %s (got %r)" % (kind, ident, failures[:3])


@case
def fact_one_library_short():
    """Drop one additional-library row: 'every additional-library point estimate
    does [reach the threshold]' must stop being true."""
    original = V.load_main_delta

    def patched():
        rows = original()
        drop = [r for r in rows if r["library"] == "pianos" and r["bb"] == "swin_t"][0]
        return [r for r in rows if r is not drop]

    V.load_main_delta = patched
    code = V.run(verbose=False)
    return code, expect(load_failures(), "fact", "q02")


@case
def fact_ceiling_broken():
    """Push one A1 recording-level score off the ceiling: 'both reach 1.00 ... in
    every seed' must stop being true."""
    original = V.load_corrected_recording

    def patched():
        out = []
        for r in original():
            if (r["library"] == "bel_canto" and r["bb"] == "swin_t"
                    and r["arm"] == "A1" and r["seed"] == "1"):
                r = dict(r, f1_rec_unweighted="0.9900")
            out.append(r)
        return out

    V.load_corrected_recording = patched
    code = V.run(verbose=False)
    return code, expect(load_failures(), "fact", "q03")


@case
def sentence_reworded():
    """Reword the checked sentence: the ledger must demand a new entry rather
    than silently passing on the old one."""
    original = V.read_tex

    def patched(rel):
        return original(rel).replace("A1 and A2 both reach 1.00", "A1 and A2 reach 1.00")

    V.read_tex = patched
    code = V.run(verbose=False)
    return code, expect(load_failures(), "text", "q03")


@case
def caption_without_entry():
    """Add a table caption that quantifies over libraries without a ledger row:
    coverage must refuse it."""
    original = V.captions

    def patched(rel, env=None):
        caps = original(rel, env=env)
        if env == "table" and rel.endswith("06_results.tex"):
            caps = caps + ["Every library agrees with this reading."]
        return caps

    V.captions = patched
    code = V.run(verbose=False)
    return code, expect(load_failures(), "coverage")


def main():
    failures = []
    base = snapshot()

    # control: an unbroken tree must pass
    restore(base)
    code = V.run(verbose=False)
    if code != 0:
        got = load_failures()
        failures.append("control: clean tree reported %d failure(s): %r" % (len(got), got[:3]))
    print("%-26s %s" % ("control (clean tree)", "PASS" if code == 0 else "FAIL"))

    for fn in CASES:
        restore(base)
        try:
            code, (ok, msg) = fn()
        except Exception as exc:  # noqa: BLE001 - a crash is a test failure, not an error
            code, ok, msg = -1, False, "%s: %s" % (type(exc).__name__, exc)
        good = code == 1 and ok
        print("%-26s %s  %s" % (fn.__name__, "PASS" if good else "FAIL", msg))
        if not good:
            failures.append("%s: %s" % (fn.__name__, msg))

    restore(base)
    V.REPORT = REAL_REPORT
    print("negtest: %d case(s), %d failed" % (len(CASES) + 1, len(failures)))
    for f in failures:
        print("  FAILED", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
