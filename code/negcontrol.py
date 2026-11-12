"""negcontrol.py -- zero-effect negative controls for the four libraries with K̄≈1 (design note v2.8 §3.7 :296-310).

Caliber:
1) K̄_c=1 ⇒ Q_pool_c = floor(K̄_c/2) = 0 and α_c = Q_pool_c/m_c ≡ 0; A1 and A0 are then mathematically equivalent,
   so the negative controls run A0 vs A2 only.
2) The 46 long recordings of GZ_IsoTech (75 extra segments, train 34->61 / test 12->14) do not enter Q_pool: per-class K̄_c stays 1 (:309),
   and the extra segments are reported separately through the Monte Carlo expected pool (mc_pool_mean) without changing the α caliber.
3) Every number is taken from disk: scan slices -> aggregate recordings by LIB_RULES -> measured K distribution -> 400 MC draws.
4) 26 assertions; any mismatch prints the failing items and exits 1.
"""
import os, re, json, random, collections, sys

ROOT = r"D:/VSCode/Datasets"
OUT = "data/splits/v2/negcontrol"
SEED, DRAWS, Q_SHARE, M_SHARE = 20261105, 400, 0.20, 0.60
DESIGN = {  # §3.7 table :298-303
    "chest_falsetto":   {"slices": 1280, "recordings": 1280, "k": 1.0000},
    "erhu_playing_tech": {"slices": 1253, "recordings": 1253, "k": 1.0000},
    "Guzheng_Tech99":   {"slices": 99,   "recordings": 99,   "k": 1.0000},
    "GZ_IsoTech":       {"slices": 2899, "recordings": 2824, "k": 1.0266},
}


def _p(*a):
    return os.path.join(ROOT, *a)


def _walk_jpg(base):
    for dp, _, fs in os.walk(base):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield dp, f


def scan(lib):
    """Returns [(path, recording_key, cls, split)]; one rule per library, split decided by directory name."""
    rows = []
    if lib == "chest_falsetto":
        base = _p(lib, "data", "eval", "mel")
        for dp, f in _walk_jpg(base):
            rows.append((os.path.join(dp, f), os.path.splitext(f)[0], "all", "eval"))
    elif lib == "erhu_playing_tech":
        base = _p(lib, "data", "eval")
        for dp, f in _walk_jpg(base):
            if not f.lower().endswith("_mel.jpg"):
                continue
            stem = os.path.splitext(f)[0][:-4]
            rows.append((os.path.join(dp, f), stem, stem.split("__")[0], "eval"))
    elif lib == "Guzheng_Tech99":
        base = _p(lib, "data", "mel")
        for dp, f in _walk_jpg(base):
            split = os.path.basename(dp) or "all"
            stem = os.path.splitext(f)[0]
            rows.append((os.path.join(dp, f), split + "/" + stem, split, split))
    elif lib == "GZ_IsoTech":
        base = _p(lib, "data", "eval", "mel")
        for dp, f in _walk_jpg(base):
            rel = os.path.relpath(dp, base)
            split = "eval" if rel == "." else rel.split(os.sep)[0]
            stem = os.path.splitext(f)[0]
            cls = os.path.basename(dp)
            rows.append((os.path.join(dp, f), re.sub(r"\d+$", "", stem), cls, split))
    else:
        raise SystemExit("unknown lib: " + lib)
    return rows


def capacity(rows):
    per = collections.defaultdict(list)
    for _, rec, cls, split in rows:
        per[cls].append(rec)
    qpanel = []
    for cls in sorted(per):
        recs = collections.Counter(per[cls])
        n_s, n_r = sum(recs.values()), len(recs)
        kbar = n_s / n_r
        m_c = int(n_s * M_SHARE)
        qpool = int(kbar // 2)          # K̄_c=1 → 0
        qpanel.append({"cls": cls, "slices": n_s, "recordings": n_r, "kbar": round(kbar, 4),
                       "m_c": m_c, "qpool_c": qpool, "alpha_c": (qpool / m_c) if m_c else 0.0})
    return qpanel


def mc_pool(rec_counter, q):
    uniq = sorted(rec_counter)
    rng = random.Random("%s|negcontrol|%d" % (q, SEED))
    vals = []
    for _ in range(DRAWS):
        pick = rng.sample(uniq, min(q, len(uniq)))
        vals.append(sum(rec_counter[r] // 2 for r in pick))
    vals.sort()
    return sum(vals) / len(vals), vals[int(0.95 * (len(vals) - 1))]


def official_overlap(lib):
    """Whether the official splits are disjoint (by recording key)."""
    if lib == "GZ_IsoTech":
        base = _p(lib, "data", "audio")
        sets = {s: {re.sub(r"\d+$", "", os.path.splitext(f)[0]) for _, _, fs in os.walk(os.path.join(base, s)) for f in fs if f.endswith(".wav")} for s in ("train", "test") if os.path.isdir(os.path.join(base, s))}
    elif lib == "Guzheng_Tech99":
        base = _p(lib, "data", "audio")
        sets = {s: {os.path.splitext(f)[0] for _, _, fs in os.walk(os.path.join(base, s)) for f in fs} for s in ("train", "validation", "test") if os.path.isdir(os.path.join(base, s))}
    else:
        return None
    keys = sorted(sets)
    return {a + "&" + b: len(sets[a] & sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}


def main():
    out = {"generated_by": "code/negcontrol.py",
           "design_ref": "design note v2.8 §3.7 :296-310",
           "seed": SEED, "draws": DRAWS, "q_share": Q_SHARE, "m_share": M_SHARE,
           "libraries": {}, "assertions": []}
    A = out["assertions"]

    def add(aid, desc, expected, actual, ok):
        A.append({"id": aid, "desc": desc, "expected": expected, "actual": actual, "ok": bool(ok)})

    gz_rows = None
    for lib, d in DESIGN.items():
        rows = scan(lib)
        recs = collections.Counter(r[1] for r in rows)
        slices, n_rec = len(rows), len(recs)
        kd = collections.Counter(recs.values())
        kbar = slices / n_rec
        qpanel = capacity(rows)
        qpool_total = sum(c["qpool_c"] for c in qpanel)
        alpha_total = sum(c["alpha_c"] for c in qpanel)
        q = max(1, round(slices * Q_SHARE))
        mc_mean, mc_p95 = mc_pool(recs, q)
        m_c_lib = int(slices * M_SHARE)
        alpha_mc_mean = mc_mean / m_c_lib
        alpha_mc_p95 = mc_p95 / m_c_lib
        ov = official_overlap(lib)
        if lib == "GZ_IsoTech":
            long_recs = {r: c for r, c in recs.items() if c > 1}
            by_split = collections.defaultdict(lambda: {"rows": 0, "segments": 0})
            for r, c in long_recs.items():
                sp = {x[3] for x in rows if x[1] == r}.pop()
                by_split[sp]["rows"] += 1
                by_split[sp]["segments"] += c - 1   # count the extra segments (design caliber: train 61 / test 14)
            gz_rows = {"rows": len(long_recs), "extra_segments": sum(c - 1 for c in long_recs.values()),
                       "by_split": {k: dict(v) for k, v in sorted(by_split.items())}}
        out["libraries"][lib] = {"slices": slices, "recordings": n_rec, "k": round(kbar, 4),
                                 "k_dist": {str(k): v for k, v in sorted(kd.items())},
                                 "q_panel_q": q, "m_c_total": int(slices * M_SHARE),   # library-level m_c = 0.60*S (design note §3.2)
                                 "qpool_total": qpool_total, "alpha_total": round(alpha_total, 6),
                                 "mc_pool_mean": round(mc_mean, 4), "mc_pool_p95": mc_p95,
                                 # design caliber (K̄_c//2, the K̄=1 idealization of :306); the expectation of one real draw is recorded separately
                                 "alpha_mc_mean": round(alpha_mc_mean, 6),
                                 "alpha_mc_p95": round(alpha_mc_p95, 6),
                                 "official_overlap": ov, "qpanel": qpanel}

        add(lib + "_slices", "slice count matches design note §3.7", d["slices"], slices, slices == d["slices"])
        add(lib + "_recordings", "recording count matches design note §3.7", d["recordings"], n_rec, n_rec == d["recordings"])
        add(lib + "_k", "K̄ matches the design note (+-0.0005)", d["k"], round(kbar, 4), abs(kbar - d["k"]) <= 0.0005)
        add(lib + "_kdist_sum", "Σ k_dist == recordings", n_rec, sum(kd.values()), sum(kd.values()) == n_rec)
        add(lib + "_qpool_zero", "Q_pool total = 0 (K̄=1 -> floor(1/2))", 0, qpool_total, qpool_total == 0)
        add(lib + "_alpha_zero", "α total = 0 (A1≡A0, A1 is not run)", 0.0, round(alpha_total, 6), alpha_total == 0.0)
        add(lib + "_alpha_mc_under_1pct", "expected α of one real draw <= 1% (multi-segment recordings add negligible exposure)",
            0.01, round(alpha_mc_mean, 6), alpha_mc_mean <= 0.01)

    gz = out["libraries"]["GZ_IsoTech"]
    add("GZ_kdist_exact", "GZ k_dist exact = {1:2778,2:26,3:11,4:9}",
        {"1": 2778, "2": 26, "3": 11, "4": 9}, gz["k_dist"], gz["k_dist"] == {"1": 2778, "2": 26, "3": 11, "4": 9})
    exp_long = {"rows": 46, "extra_segments": 75, "train": 61, "test": 14}
    act_long = {"rows": gz_rows["rows"], "extra_segments": gz_rows["extra_segments"],
                "train": gz_rows["by_split"].get("train", {}).get("segments"),
                "test": gz_rows["by_split"].get("test", {}).get("segments")}
    add("GZ_long_records", "GZ long recordings: 46 rows / 75 extra segments (train 61, test 14)", exp_long, act_long, act_long == exp_long)

    out["gz_long_recordings"] = gz_rows
    out["domain"] = {lib: ("synth" if lib == "chest_falsetto" else "real") for lib in DESIGN}
    out["all_pass"] = all(a["ok"] for a in A)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "negcontrol.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "GZ_IsoTech_long_recordings.csv"), "w", encoding="utf-8") as f:
        f.write("recording_key,split,segments,extra_segments\n")
        rows = scan("GZ_IsoTech")
        recs = collections.Counter(r[1] for r in rows)
        for r in sorted(recs):
 
            if recs[r] > 1:
                sp = {x[3] for x in rows if x[1] == r}.pop()
                f.write("%s,%s,%d,%d\n" % (r, sp, recs[r], recs[r] - 1))
    bad = [a for a in A if not a["ok"]]
    for a in bad:
        print("!! FAIL %s | expected=%s actual=%s" % (a["id"], a["expected"], a["actual"]))
    print("%d assertions, %d failed, all_pass=%s" % (len(A), len(bad), out["all_pass"]))
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
