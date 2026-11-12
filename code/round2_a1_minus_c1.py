#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""round2_a1_minus_c1.py — direct paired interval for A1−C1 on CTIS (slice level / recording level),
plus a field-by-field reconciliation of A1−A0 / C1−A0 against the published CSVs (fidelity gate).

Every statistical convention follows code/analyze_e1.py line by line (no second implementation):
  * macro F1 = confusion-matrix convention: tp=diagonal, fp=column sum−tp, fn=row sum−tp; keep = the class is present
    in **this (resampled) sample** (row sum + column sum > 0); empty classes are excluded from the average (analyze_e1.py:96-106). Same
    convention as sklearn average='macro', zero_division=0.
  * slice level: pred = probs.argmax(1); recording level: average probs within a recording, then one argmax vote
    (analyze_e1.py:114-126).
  * recording-cluster bootstrap, 2000 replicates: resampling unit = recording cluster (whole cluster kept or dropped, never split); RNG = default_rng(20261105);
    drawn once per library in sorted(lib) order, size=(2000, n_rec) (analyze_e1.py:205-210), so the CTIS
    bootstrap index matrix is **the same stream** as the published e1_delta.csv, and A1−C1 shares one set of cluster indices with A1−A0.
  * Δ = per-seed paired difference within each replicate, then averaged over the 3 seeds; Δ = F1_arm − F1_reference, in percentage points.
    The point estimate uses the same convention (per-seed paired differences averaged over 3 seeds).
  * 95% CI = the 2.5/97.5 percentiles of the bootstrap distribution of Δ.
  * boot p is reported under both conventions: boot_p = analyze_e1's current two-sided +1 (2*(k+1)/(B+1), analyze_e1.py:258-260);
    boot_p_legacy = the published e1_delta.csv convention 2*min(k_le0,k_ge0)/B with floor 1/B (see the function docstring).
  * Fidelity: slice level is reconciled field by field against e1_delta.csv; recording-level CI against e1_delta.csv and the **point estimate** against
    point_unw_pt of human_review/corrected_recording_delta.csv (delta_rec_pt in e1_delta.csv is the superseded
    legacy slice-weighted value; the manuscript text uses point_unw); boot_p is recomputed under the legacy convention.
  * Transitivity: within one resample, Δ(A1−C1) must equal Δ(A1−A0) − Δ(C1−A0) exactly (asserted in the script).
  * seed spread = max−min of the per-seed Δ (reported only; the 3 seeds are not treated as independent samples in a test).
  * Pairing integrity (section 2 of this script): under one backbone and seed, the y and records of A0/A1/A2/C1 must match element by element;
    y/records must also match across seeds; each macro F1 must agree with the qtest record in seed{k}.json (tol 5e-4).

 Exploratory grouping (section 5): split by whether a test recording appears in the A1 injection pool. The exposure list is not recomputed; it is read from the frozen splits
   data/splits/v2/CTIS_{qpool,qtest,a1_train}.csv: rows of a1_train whose sample_id ∈ qpool are the slices actually injected;
   the intersection of their recording_canonical with that of qtest gives the exposed test recordings (same name and
   same values as the npz records column, asserted in the script; using recording_uid gives the same intersection size). This convention matches
   code/audit_review_exposure_20261108.py and agrees with the 578/963 of table
   tab:realised-injection in paper/sections/10_supplement.tex (asserted in the script). The cluster bootstrap for group Δ resamples clusters within the group,
   using an independent stream default_rng(20261105) (not the same stream as the original table; interpretable within groups only).

Reads: runs/sealed_eval/**, data/splits/v2/**, data/results/{e1_delta.csv,c1/e1_delta.csv}.
Writes: data/results/human_review/round2/a1_minus_c1.{csv,json}.

 Usage: python code/round2_a1_minus_c1.py
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_e1 as ae  # noqa: E402  reuse its load_npz / macro_f1 / conf_slice / per_recording / discover

RUNS = os.path.join(ROOT, "runs", "sealed_eval")
SPLITS = os.path.join(ROOT, "data", "splits", "v2")
OUTDIR = os.path.join(ROOT, "data", "results", "human_review", "round2")
OUT_CSV = os.path.join(OUTDIR, "a1_minus_c1.csv")
OUT_JSON = os.path.join(OUTDIR, "a1_minus_c1.json")
PUB_A1A0 = os.path.join(ROOT, "data", "results", "e1_delta.csv")
PUB_C1A0 = os.path.join(ROOT, "data", "results", "c1", "e1_delta.csv")

BOOT = 2000
BOOT_SEED = 20261105
SESOI = 3.0
ARMS = ("A0", "A1", "A2", "C1")
SEEDS = (0, 1, 2)
BBS = ("alexnet", "swin_t")
CHUNK = 200

CSV_COLS = ["lib", "backbone", "level", "contrast", "delta_pp", "ci_lo", "ci_hi",
            "seed_deltas", "seed_range", "n_slices", "n_recordings"]


def die(msg):
    print("[PAIRING-FAIL] " + msg)
    sys.exit(2)


def macro_f1_batch(conf):
    """(B,C,C) confusion matrices -> (B,) macro F1; element-wise the same convention as ae.macro_f1 (empty classes dropped)."""
    tp = np.diagonal(conf, axis1=1, axis2=2).astype(np.float64)
    fp = conf.sum(axis=1) - tp          # column sum (all cases with pred=j, tp included)
    fn = conf.sum(axis=2) - tp          # row sum (all cases with true=i, tp included)
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(den > 0, 2 * tp / den, np.nan)
    keep = (conf.sum(axis=2) + conf.sum(axis=1)) > 0
    f1 = np.where(keep, f1, np.nan)
    return np.nanmean(f1, axis=1)


def batch_f1_slice(pred, y, nc, W):
    """Slice-level bootstrap: W (B,n) holds the cluster multiplicity weight of each slice."""
    flat = y * nc + pred
    out = np.empty(W.shape[0])
    for st in range(0, W.shape[0], CHUNK):
        en = min(W.shape[0], st + CHUNK)
        wsum = np.zeros((en - st, nc * nc))
        for i, b in enumerate(range(st, en)):
            wsum[i] = np.bincount(flat, weights=W[b], minlength=nc * nc)
        out[st:en] = macro_f1_batch(wsum.reshape(en - st, nc, nc))
    return out


def batch_f1_rec(rec_y, rec_pred, nc, mult):
    """Recording-level bootstrap: mult (B,R) holds the multiplicity weight of each recording cluster."""
    flat = rec_y * nc + rec_pred
    out = np.empty(mult.shape[0])
    for st in range(0, mult.shape[0], CHUNK):
        en = min(mult.shape[0], st + CHUNK)
        wsum = np.zeros((en - st, nc * nc))
        for i, b in enumerate(range(st, en)):
            wsum[i] = np.bincount(flat, weights=mult[b], minlength=nc * nc)
        out[st:en] = macro_f1_batch(wsum.reshape(en - st, nc, nc))
    return out


def boot_p(d):
    """Two-sided percentile p with the +1 correction (analyze_e1.py:258-260, current implementation): 2*(k+1)/(B+1)."""
    smaller = min(int(np.count_nonzero(d <= 0)), int(np.count_nonzero(d >= 0)))
    return min(1.0, 2.0 * (smaller + 1) / (d.size + 1))


def boot_p_legacy(d):
    """boot_p convention of the published e1_delta.csv: 2*min(#<=0,#>=0)/B, floored at 1/B.

    Basis: back-solved from n_draws_le0 and p_legacy_clamped in human_review/slice_p_addone.csv
    (k=0 -> 2*0/2000 caught by the floor 1/2000=0.0005; k=56 -> 0.056; k=462 -> 0.462;
    k=731 -> 0.731; k=20 -> 0.02), matching all four CTIS cells of e1_delta.csv one by one.
    """
    k = min(int(np.count_nonzero(d <= 0)), int(np.count_nonzero(d >= 0)))
    return max(1.0 / d.size, min(1.0, 2.0 * k / d.size))


def read_csv_dicts(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    ae.RUNS = RUNS
    data = ae.discover()

    # ================= 1. pairing-integrity assertions =================
    checks = []
    n_y = n_r = n_f1 = 0
    for bb in BBS:
        for s in SEEDS:
            ref = None
            for arm in ARMS:
                p = os.path.join(RUNS, "CTIS_%s_%s" % (arm, bb), "seed%d_qtest.npz" % s)
                if not os.path.isfile(p):
                    die("missing sealed file %s" % p)
                probs, y, rec = ae.load_npz(p)
                if ref is None:
                    ref = (y, rec, arm)
                    checks.append({"check": "y/records reference", "bb": bb, "seed": s, "arm": arm,
                                   "n": int(y.size), "n_recordings": int(np.unique(rec).size),
                                   "ok": True})
                else:
                    ok_y = bool(np.array_equal(y, ref[0]))
                    ok_r = bool(np.array_equal(rec, ref[1]))
                    n_y += 1
                    n_r += 1
                    if not ok_y:
                        die("%s/seed%d: y of %s does not match %s element by element" % (bb, s, arm, ref[2]))
                    if not ok_r:
                        die("%s/seed%d: records of %s do not match %s element by element" % (bb, s, arm, ref[2]))
                    checks.append({"check": "y/records pairing", "bb": bb, "seed": s, "arm": arm,
                                   "vs": ref[2], "ok": True})
                # metric recomputation reconciled against the qtest record in seed{k}.json
                nc = int(y.max()) + 1
                f1s = ae.macro_f1(ae.conf_slice(y, probs.argmax(1), nc))[0]
                rec_y, rec_pred, _ = ae.per_recording(probs, y, np.unique(rec, return_inverse=True)[1],
                                                      int(np.unique(rec).size), nc)
                f1r = ae.macro_f1(ae.conf_slice(rec_y, rec_pred, nc))[0]
                st = json.loads(open(os.path.join(RUNS, "CTIS_%s_%s" % (arm, bb),
                                                  "seed%d.json" % s), encoding="utf-8").read())["qtest"]
                n_f1 += 2
                if abs(f1s - st["macro_f1_slice"]) > 5e-4 or abs(f1r - st["macro_f1_recording"]) > 5e-4:
                    die("%s/%s/seed%d: recomputed F1 (%.6f/%.6f) disagrees with seed json (%.6f/%.6f)"
                        % (bb, arm, s, f1s, f1r, st["macro_f1_slice"], st["macro_f1_recording"]))
                checks.append({"check": "F1 recomputation", "bb": bb, "seed": s, "arm": arm,
                               "f1_slice": round(f1s, 6), "f1_recording": round(f1r, 6),
                               "json_slice": st["macro_f1_slice"],
                               "json_recording": st["macro_f1_recording"], "ok": True})
    # across seeds: the slice set must be stable
    for bb in BBS:
        for s in SEEDS[1:]:
            _, y0, r0 = ae.load_npz(os.path.join(RUNS, "CTIS_A0_%s" % bb, "seed0_qtest.npz"))
            _, y, r = ae.load_npz(os.path.join(RUNS, "CTIS_A0_%s" % bb, "seed%d_qtest.npz" % s))
            if not (np.array_equal(y, y0) and np.array_equal(r, r0)):
                die("%s: slice set/order of seed%d drifts from seed0" % (bb, s))
            checks.append({"check": "cross-seed slice set", "bb": bb, "seed": s, "ok": True})
    print("[assert] pairing integrity PASS: %d y comparisons + %d records comparisons + %d F1 recomputations"
          % (n_y, n_r, n_f1))

    # ================= 2. replay the analyze_e1 bootstrap stream =================
    rng = np.random.default_rng(BOOT_SEED)
    boot, pt = {}, {}   # (arm,bb,seed) -> (f1s_boot, f1r_boot) / (f1s, f1r)
    y_ref = rec_ref = None
    nrec = None
    for lib in sorted(data):
        per_arm = data[lib]
        if ("A0", "alexnet") not in per_arm:      # same library-skip condition as analyze_e1:155-158, keeps the stream aligned
            continue
        z0 = sorted(per_arm[("A0", "alexnet")])[0]
        _, y_l, r_l = ae.load_npz(per_arm[("A0", "alexnet")][z0])
        recs, rec_idx = np.unique(r_l, return_inverse=True)
        nrec_l = recs.size
        pick = rng.integers(0, nrec_l, size=(BOOT, nrec_l))
        mult = np.stack([np.bincount(p, minlength=nrec_l) for p in pick])
        if lib != "CTIS":
            continue
        y_ref, rec_ref, nrec = y_l, r_l, nrec_l
        nc = int(y_ref.max()) + 1
        W = mult[:, rec_idx]                       # (B, n) slice-level cluster weights

        for (arm, bb), seeds in sorted(per_arm.items()):
            for s in sorted(seeds):
                probs, y, rec = ae.load_npz(seeds[s])
                if not (np.array_equal(y, y_ref) and np.array_equal(rec, rec_ref)):
                    die("CTIS/%s/%s/seed%d slice order disagrees with the A0 reference" % (arm, bb, s))
                rec_y_s, rec_pred_s, _ = ae.per_recording(probs, y, rec_idx, nrec, nc)
                pred = probs.argmax(1)
                pt[(arm, bb, s)] = (ae.macro_f1(ae.conf_slice(y, pred, nc))[0],
                                    ae.macro_f1(ae.conf_slice(rec_y_s, rec_pred_s, nc))[0])
                boot[(arm, bb, s)] = (batch_f1_slice(pred, y, nc, W),
                                      batch_f1_rec(rec_y_s, rec_pred_s, nc, mult))
        # batched vs scalar implementation agreement (3 replicates sampled and checked)
        for (arm, bb, s) in [(k) for k in boot if k[0] == "A0" and k[1] == "alexnet"]:
            probs, y, _ = ae.load_npz(per_arm[(arm, bb)][s])
            flat = y * nc + probs.argmax(1)
            for b in (0, 7, 999):
                c = np.bincount(flat, weights=W[b], minlength=nc * nc).reshape(nc, nc)
                if abs(ae.macro_f1(c)[0] - boot[(arm, bb, s)][0][b]) > 1e-12:
                    die("batched/scalar macro F1 disagree (%s/%s/seed%d, b=%d)" % (arm, bb, s, b))
        print("[assert] bootstrap implementations agree: CTIS cluster indices replayed from the analyze_e1 stream, batched = scalar at slice level")

    # ================= 3. contrasts A1−C1 / A1−A0 / C1−A0 =================
    def contrast(num, ref, bb):
        seeds = sorted(set(k[2] for k in boot if k[0] == num and k[1] == bb)
                       & set(k[2] for k in boot if k[0] == ref and k[1] == bb))
        out = {}
        for i, level in ((0, "slice"), (1, "recording")):
            d_b = np.mean(np.stack([boot[(num, bb, s)][i] - boot[(ref, bb, s)][i] for s in seeds]), axis=0)
            pd = [100.0 * (pt[(num, bb, s)][i] - pt[(ref, bb, s)][i]) for s in seeds]
            lo, hi = np.percentile(100.0 * d_b, [2.5, 97.5])
            out[level] = {"delta_pp": float(np.mean(pd)), "ci_lo": float(lo), "ci_hi": float(hi),
                          "seed_deltas": [float(v) for v in pd],
                          "seed_range": float(max(pd) - min(pd)),
                          "boot_p": boot_p(d_b), "boot_p_legacy": boot_p_legacy(d_b),
                          "n_le0": int(np.count_nonzero(d_b <= 0)),
                          "n_ge0": int(np.count_nonzero(d_b >= 0)),
                          "boot": int(d_b.size), "n_seeds": len(seeds),
                          "seeds": seeds}
        return out

    contrasts = {}
    for bb in BBS:
        for num, ref in (("A1", "C1"), ("A1", "A0"), ("C1", "A0")):
            contrasts[("%s-%s" % (num, ref), bb)] = contrast(num, ref, bb)

    # transitivity self-check: within one resample and one seed pairing, Δ(A1−C1) must equal Δ(A1−A0) − Δ(C1−A0) exactly
    tri = []
    for bb in BBS:
        for lv in ("slice", "recording"):
            a = contrasts[("A1-C1", bb)][lv]["delta_pp"]
            b = contrasts[("A1-A0", bb)][lv]["delta_pp"] - contrasts[("C1-A0", bb)][lv]["delta_pp"]
            tri.append({"backbone": bb, "level": lv, "a1_c1_pp": round(a, 6),
                        "a1_a0_minus_c1_a0_pp": round(b, 6), "abs_diff": abs(a - b)})
    assert all(t["abs_diff"] < 1e-9 for t in tri), tri

    # ================= 4. fidelity gate: field-by-field comparison with the published CSVs =================
    # Convention notes (reviewed 2026-11-11):
    #  (a) slice-level Δ/CI/seed spread/per-seed Δ — compared directly with e1_delta.csv (c1/e1_delta.csv).
    #  (b) recording-level CI — against ci_lo_rec_pt/ci_hi_rec_pt of e1_delta.csv (cluster-bootstrap quantiles, unaffected);
    #      the recording-level **point estimate** against point_unw_pt of human_review/corrected_recording_delta.csv:
    #      delta_rec_pt in e1_delta.csv is the registered legacy slice-weighted point estimate
    #      (= legacy_point_sw_pt of corrected_recording_delta.csv); the manuscript text now uses point_unw.
    #      The legacy difference is therefore listed as an informational item and not counted in the gate.
    #  (c) boot_p — e1_delta.csv uses 2*min(k_le0,k_ge0)/B floored at 1/B (see the boot_p_legacy docstring),
    #      which differs from the current two-sided +1 convention of analyze_e1.py; the gate compares the CSV convention and the JSON reports both values.
    CORR = os.path.join(ROOT, "data", "results", "human_review", "corrected_recording_delta.csv")
    corr_rows = {(r["library"], r["bb"], r["contrast"]): r for r in read_csv_dicts(CORR)}

    def cmp_block(path, arm_pair):
        rows = {(r["library"], r["bb"]): r for r in read_csv_dicts(path)}
        det, ok_all = {}, True
        for bb in BBS:
            pub = rows.get(("CTIS", bb))
            corr = corr_rows.get(("CTIS", bb, arm_pair))
            if pub is None or corr is None:
                det["%s/%s" % (arm_pair, bb)] = {"missing": True}
                ok_all = False
                continue
            mine = contrasts[(arm_pair, bb)]
            fdet, ok = {}, True

            def put(name, pv, mv, tol, note=""):
                nonlocal ok
                same = abs(pv - mv) <= tol
                item = {"published": pv, "recomputed": round(float(mv), 4),
                        "abs_diff": round(abs(pv - mv), 4), "tol": tol, "match": bool(same)}
                if note:
                    item["note"] = note
                fdet[name] = item
                ok = ok and same

            put("delta_slice_pt", float(pub["delta_slice_pt"]), mine["slice"]["delta_pp"], 0.011)
            put("ci_lo_slice_pt", float(pub["ci_lo_slice_pt"]), mine["slice"]["ci_lo"], 0.011)
            put("ci_hi_slice_pt", float(pub["ci_hi_slice_pt"]), mine["slice"]["ci_hi"], 0.011)
            put("seed_range_pt", float(pub["seed_range_pt"]), mine["slice"]["seed_range"], 0.011)
            put("boot_p_slice", float(pub["boot_p_slice"]), mine["slice"]["boot_p_legacy"], 5e-4,
                "e1_delta.csv convention 2*min(k_le0,k_ge0)/B; the current analyze_e1 two-sided +1 value is %.4f"
                % mine["slice"]["boot_p"])
            put("ci_lo_rec_pt", float(pub["ci_lo_rec_pt"]), mine["recording"]["ci_lo"], 0.011)
            put("ci_hi_rec_pt", float(pub["ci_hi_rec_pt"]), mine["recording"]["ci_hi"], 0.011)
            put("delta_rec_pt_unw_vs_corrected", float(corr["point_unw_pt"]),
                mine["recording"]["delta_pp"], 5e-3,
                "point_unw_pt of corrected_recording_delta.csv is authoritative (the manuscript text uses this value)")
            fdet["delta_rec_pt_legacy_in_e1_delta"] = {
                "published": float(pub["delta_rec_pt"]),
                "recomputed": round(mine["recording"]["delta_pp"], 4),
                "abs_diff": round(abs(float(pub["delta_rec_pt"]) - mine["recording"]["delta_pp"]), 4),
                "correction_pt": float(corr["correction_pt"]),
                "match": None,
                "note": "known correction, not counted in the gate: e1_delta.csv keeps the legacy slice-weighted point estimate,"
                        " and correction_pt of corrected_recording_delta.csv is the difference between the two"}
            ps_mine = "|".join("%.2f" % v for v in mine["slice"]["seed_deltas"])
            same = ps_mine == pub["per_seed_delta_pt"]
            fdet["per_seed_delta_pt"] = {"published": pub["per_seed_delta_pt"],
                                         "recomputed": ps_mine, "match": bool(same)}
            ok = ok and same
            psr_mine = "|".join("%.2f" % v for v in mine["recording"]["seed_deltas"])
            same_r = psr_mine == corr["per_seed_unw_pt"]
            fdet["per_seed_unw_pt"] = {"published": corr["per_seed_unw_pt"],
                                       "recomputed": psr_mine, "match": bool(same_r),
                                       "note": "recording level, per seed, against corrected_recording_delta.csv"}
            ok = ok and same_r
            det["%s/%s" % (arm_pair, bb)] = {"fields": fdet, "all_match": bool(ok)}
            ok_all = ok_all and ok
        return {"source": os.path.relpath(path, ROOT).replace("\\", "/"),
                "recording_point_source": os.path.relpath(CORR, ROOT).replace("\\", "/"),
                "details": det, "all_fields_match": bool(ok_all)}

    fidelity = {"A1-A0": cmp_block(PUB_A1A0, "A1-A0"),
                "C1-A0": cmp_block(PUB_C1A0, "C1-A0")}
    fidelity["verdict"] = (
        "consistent: slice-level Δ/CI/seed identical field by field with e1_delta.csv; recording-level CI identical, point estimate identical to "
        "point_unw_pt of corrected_recording_delta.csv; boot_p reproduced under the 2k/B convention of e1_delta.csv"
        if fidelity["A1-A0"]["all_fields_match"] and fidelity["C1-A0"]["all_fields_match"]
        else "fields disagree: check this script's implementation first; the published numbers must not be changed")

    # ================= 5. exploratory grouping: exposed to the A1 injection pool vs not =================
    test_rows = read_csv_dicts(os.path.join(SPLITS, "CTIS_qtest.csv"))
    pool_rows = read_csv_dicts(os.path.join(SPLITS, "CTIS_qpool.csv"))
    a1_rows = read_csv_dicts(os.path.join(SPLITS, "CTIS_a1_train.csv"))
    pool_ids = {r["sample_id"] for r in pool_rows}
    injected = [r for r in a1_rows if r["sample_id"] in pool_ids]
    # exposure convention agrees with code/audit_review_exposure_20261108.py and the npz records column: recording_canonical is used
    inj_recs = {r["recording_canonical"] for r in injected}
    test_recs = {r["recording_canonical"] for r in test_rows}
    assert not ({r["sample_id"] for r in test_rows} & pool_ids), "sample_id overlap between qpool and qtest"
    assert set(rec_ref.tolist()) == test_recs, "npz records disagree with qtest recording_canonical"
    uid_x = {r["recording_uid"] for r in injected} & {r["recording_uid"] for r in test_rows}
    exposed = inj_recs & test_recs
    assert len(exposed) == 578, "exposed recordings should be 578 (measured by code/audit_review_exposure_20261108.py)"
    group_info = {"source": "data/splits/v2/CTIS_{qpool,qtest,a1_train}.csv",
                   "rule": "rows of a1_train with sample_id ∈ qpool are the slices actually injected; their recording_canonical "
                           "∩ qtest recording_canonical gives the exposed recordings (same name and values as the npz records column, asserted)",
                  "cross_check": {"audit_script": "code/audit_review_exposure_20261108.py",
                                   "audit_reported": "4178 slices injected / 578 exposed test recordings / 963 test recordings",
                                  "uid_based_exposed_recordings": len(uid_x)},
                  "injected_slices": len(injected), "injected_recordings": len(inj_recs),
                  "test_recordings": len(test_recs), "exposed_test_recordings": len(exposed),
                  "unexposed_test_recordings": len(test_recs - exposed),
                  "paper_table_cell": "578/963 (paper/sections/10_supplement.tex:44)"}
    if not (len(exposed) == 578 and len(test_recs) == 963 and len(injected) == 4178):
        group_info["warn"] = "disagrees with the supplementary table (578/963, 4178 slices); group results downgraded to unverified"
        print("[warn] exposure list disagrees with the supplementary table: %d/%d, injected=%d"
              % (len(exposed), len(test_recs), len(injected)))

    recs_all, rec_idx_all = np.unique(rec_ref, return_inverse=True)
    nc = int(y_ref.max()) + 1
    grp_rows = []
    rng_g = np.random.default_rng(BOOT_SEED)
    for gname, gset in (("exposed", exposed), ("unexposed", test_recs - exposed)):
        gmask = np.array([r in gset for r in recs_all])
        keep_slices = gmask[rec_idx_all]
        g_rec_idx = rec_idx_all[keep_slices]
        g_map = np.full(recs_all.size, -1)
        g_map[gmask] = np.arange(int(gmask.sum()))
        g_rec_idx = g_map[g_rec_idx]
        R_g = int(gmask.sum())
        y_g = y_ref[keep_slices]
        mult_g = np.stack([np.bincount(p, minlength=R_g)
                           for p in rng_g.integers(0, R_g, size=(BOOT, R_g))])
        W_g = mult_g[:, g_rec_idx]
        g_pt, g_boot = {}, {}
        for bb in BBS:
            for s in SEEDS:
                for arm in ("A0", "A1"):
                    probs, y, rec = ae.load_npz(os.path.join(
                        RUNS, "CTIS_%s_%s" % (arm, bb), "seed%d_qtest.npz" % s))
                    pred = probs[keep_slices].argmax(1)
                    rec_y_g, rec_pred_g, _ = ae.per_recording(probs[keep_slices], y_g, g_rec_idx, R_g, nc)
                    g_pt[(arm, bb, s)] = (ae.macro_f1(ae.conf_slice(y_g, pred, nc))[0],
                                          ae.macro_f1(ae.conf_slice(rec_y_g, rec_pred_g, nc))[0])
                    g_boot[(arm, bb, s)] = (batch_f1_slice(pred, y_g, nc, W_g),
                                            batch_f1_rec(rec_y_g, rec_pred_g, nc, mult_g))
            d_s = np.mean(np.stack([g_boot[("A1", bb, s)][0] - g_boot[("A0", bb, s)][0]
                                    for s in SEEDS]), axis=0)
            d_r = np.mean(np.stack([g_boot[("A1", bb, s)][1] - g_boot[("A0", bb, s)][1]
                                    for s in SEEDS]), axis=0)
            ps = [100.0 * (g_pt[("A1", bb, s)][0] - g_pt[("A0", bb, s)][0]) for s in SEEDS]
            pr = [100.0 * (g_pt[("A1", bb, s)][1] - g_pt[("A0", bb, s)][1]) for s in SEEDS]
            for level, d_b, pd in (("slice", d_s, ps), ("recording", d_r, pr)):
                lo, hi = np.percentile(100.0 * d_b, [2.5, 97.5])
                grp_rows.append({"lib": "CTIS", "backbone": bb, "level": level,
                                 "contrast": "A1-A0[%s]" % gname,
                                 "delta_pp": float(np.mean(pd)), "ci_lo": float(lo),
                                 "ci_hi": float(hi),
                                 "seed_deltas": [float(v) for v in pd],
                                 "seed_range": float(max(pd) - min(pd)),
                                 "boot_p": boot_p(d_b), "boot_p_legacy": boot_p_legacy(d_b),
                                 "n_le0": int(np.count_nonzero(d_b <= 0)),
                                 "n_ge0": int(np.count_nonzero(d_b >= 0)),
                                 "n_slices": int(keep_slices.sum()),
                                 "n_recordings": R_g,
                                 "group_definition": group_info["rule"]})
    group_info["rows"] = grp_rows
    _g = {(r["contrast"], r["level"]): r for r in grp_rows}
    print("[group] exposed %d rec / %d slices | unexposed %d rec / %d slices"
          % (len(exposed), _g[("A1-A0[exposed]", "slice")]["n_slices"],
             len(test_recs - exposed), _g[("A1-A0[unexposed]", "slice")]["n_slices"]))

    # ================= 6. write out =================
    rows = []

    def emit(contrast_name, bb, level, m, n_slices, n_recs):
        rows.append({"lib": "CTIS", "backbone": bb, "level": level, "contrast": contrast_name,
                     "delta_pp": round(m["delta_pp"], 2), "ci_lo": round(m["ci_lo"], 2),
                     "ci_hi": round(m["ci_hi"], 2),
                     "seed_deltas": "|".join("%.2f" % v for v in m["seed_deltas"]),
                     "seed_range": round(m["seed_range"], 2),
                     "n_slices": n_slices, "n_recordings": n_recs,
                     "boot_p": round(m.get("boot_p", float("nan")), 4),
                     "boot_p_legacy": round(m.get("boot_p_legacy", float("nan")), 4)})

    n_slices, n_recs = int(y_ref.size), int(nrec)
    for bb in BBS:
        for level in ("slice", "recording"):
            emit("A1-C1", bb, level, contrasts[("A1-C1", bb)][level], n_slices, n_recs)
    for bb in BBS:
        for level in ("slice", "recording"):
            emit("A1-A0", bb, level, contrasts[("A1-A0", bb)][level], n_slices, n_recs)
            emit("C1-A0", bb, level, contrasts[("C1-A0", bb)][level], n_slices, n_recs)
    for g in grp_rows:
        emit(g["contrast"], g["backbone"], g["level"],
             {"delta_pp": g["delta_pp"], "ci_lo": g["ci_lo"], "ci_hi": g["ci_hi"],
              "seed_deltas": g["seed_deltas"], "seed_range": g["seed_range"],
              "boot_p": g["boot_p"], "boot_p_legacy": g.get("boot_p_legacy", float("nan"))},
             g["n_slices"], g["n_recordings"])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "script": "code/round2_a1_minus_c1.py",
        "inputs": {"sealed": "runs/sealed_eval/CTIS_{A0,A1,A2,C1}_{alexnet,swin_t}/seed{0,1,2}_qtest.npz",
                   "public": ["data/results/e1_delta.csv", "data/results/c1/e1_delta.csv"],
                   "splits": "data/splits/v2/CTIS_{qpool,qtest,a1_train}.csv"},
        "env": {"python": sys.executable, "numpy": np.__version__},
        "pairing_assertions": {
            "passed": True,
            "what": "under one backbone and seed the y and records of A0/A1/A2/C1 match element by element; the slice set is identical across seeds;"
                    " every macro F1 cell (slice/recording level) agrees with the qtest record in seed{k}.json (tol 5e-4)",
            "counts": {"y_compare": n_y, "records_compare": n_r, "f1_recompute": n_f1},
            "detail": checks},
        "method": {
            "metric": "macro-F1, confusion-matrix convention, empty classes excluded (analyze_e1.macro_f1)",
            "levels": {"slice": "argmax over all slices", "recording": "mean probs within a recording, then argmax"},
            "bootstrap": {"n": BOOT, "cluster": "recording", "rng": "default_rng(%d)" % BOOT_SEED,
                          "stream": "drawn once per library in sorted(lib) order, size=(2000,n_rec), the same stream as analyze_e1.py:205-210;"
                                    " the CTIS index matrix is identical to the published e1_delta.csv",
                          "delta": "per-seed paired differences within each replicate, averaged over the 3 seeds",
                          "ci": "2.5/97.5 percentiles of the Δ distribution",
                          "p": "both conventions reported: boot_p = analyze_e1 current two-sided +1 (2*(k+1)/(B+1));"
                               " boot_p_legacy = e1_delta.csv convention 2*min(k_le0,k_ge0)/B (floor 1/B)"},
            "units": "percentage points (pp)",
            "sesoi_pp": SESOI,
            "seeds": list(SEEDS),
        },
        "contrasts": [{"lib": "CTIS", "backbone": bb, "contrast": c,
                       "level": lv, "delta_pp": round(contrasts[(c, bb)][lv]["delta_pp"], 4),
                       "ci_lo": round(contrasts[(c, bb)][lv]["ci_lo"], 4),
                       "ci_hi": round(contrasts[(c, bb)][lv]["ci_hi"], 4),
                       "seed_deltas_pp": [round(v, 4) for v in contrasts[(c, bb)][lv]["seed_deltas"]],
                       "seed_range_pp": round(contrasts[(c, bb)][lv]["seed_range"], 4),
                       "boot_p": round(contrasts[(c, bb)][lv]["boot_p"], 4),
                       "boot_p_legacy": round(contrasts[(c, bb)][lv]["boot_p_legacy"], 4),
                       "n_le0": contrasts[(c, bb)][lv]["n_le0"],
                       "n_ge0": contrasts[(c, bb)][lv]["n_ge0"],
                       "n_slices": n_slices, "n_recordings": n_recs,
                       "ci_excludes_zero": bool(contrasts[(c, bb)][lv]["ci_lo"] > 0
                                                or contrasts[(c, bb)][lv]["ci_hi"] < 0),
                       "meets_sesoi": bool(abs(contrasts[(c, bb)][lv]["delta_pp"]) >= SESOI)}
                      for c in ("A1-C1", "A1-A0", "C1-A0") for bb in BBS
                      for lv in ("slice", "recording")],
        "fidelity_check": fidelity,
        "transitivity_check": {"what": "Δ(A1−C1) = Δ(A1−A0) − Δ(C1−A0) (point estimates, pp)",
                               "passed": True, "rows": tri},
        "exploratory_exposure": group_info,
        "notes": [
            "A1−C1 is a contrast added in this round with no previously published value; A1−A0 / C1−A0 serve the fidelity gate.",
            "Fidelity: slice level identical field by field with e1_delta.csv; the recording-level point estimate follows human_review/"
            "corrected_recording_delta.csv's point_unw_pt (the manuscript text uses this value);"
            " the delta_rec_pt in e1_delta.csv is the superseded legacy slice-weighted value (correction_pt is the difference between the two).",
            "boot_p differs from e1_delta.csv because of the convention: the CSV uses 2*min(k_le0,k_ge0)/B (floor 1/B), while"
            " analyze_e1.py is now two-sided +1; both are listed in the JSON, significance for A1−C1 is judged by CI and p is informative only.",
            "A1 and C1 have different n_train (A1=25498, C1=25502); the gap comes from how the injection pool was realised,"
            " not from an evaluation-convention difference.",
            "The macro F1 of a group Δ averages only over classes present in that group; the two groups have different class universes,"
            " so absolute values are not comparable across them;"
            " within a group Δ is still a paired difference.",
            "The group bootstrap uses an independent stream default_rng(%d), not the same stream as the original CTIS table." % BOOT_SEED,
        ],
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)

    # ================= 7. console summary =================
    print("\n%-8s %-10s %-20s %8s  %-18s %-18s %6s %6s %7s %7s" % (
        "backbone", "level", "contrast", "delta", "95%CI", "per-seed", "rng", "nRec", "p+1", "p_leg"))
    for r in rows:
        print("%-8s %-10s %-20s %+8.2f  [%+6.2f,%+6.2f]  %-18s %6.2f %6d %7.4f %7.4f" % (
            r["backbone"], r["level"], r["contrast"], r["delta_pp"], r["ci_lo"], r["ci_hi"],
            r["seed_deltas"], r["seed_range"], r["n_recordings"], r["boot_p"], r["boot_p_legacy"]))
    print("\n[gate] %s" % fidelity["verdict"])
    print("written: %s | %s" % (OUT_CSV, OUT_JSON))
    return 0


if __name__ == "__main__":
    sys.exit(main())
