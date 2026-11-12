"""Add-one rescaling of the slice-level p-values (review comment B10, follow-up item).

Two things only, with **no re-run and no replacement of the existing slice-level CIs**:
  1. Replay the original random stream of `code/analyze_e1.py` (`default_rng(20261105)`, one draw per library in
     alphabetical order, `boot=2000`) to recompute the bootstrap distribution of the slice-level delta, and **verify**
     that the 2.5/97.5 percentiles match `data/results/e1_delta.csv` (a match proves these are the same resamples,
     not a fresh set);
  2. Report add-one p-values on that set of resamples: p = 2*(1+min(#{delta*<=0}, #{delta*>=0})) / (B+1),
     with a Holm correction over the confirmatory family (CTIS x two backbones x A1-A0).

Writes data/results/human_review/slice_p_addone.{csv,json}; never writes back to e1_delta.csv / e1_holm.json.
"""
import argparse
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs", "sealed_eval")
OUT = os.path.join(ROOT, "data", "results", "human_review")
RES = os.path.join(ROOT, "data", "results")
LIBS = ["CTIS", "music_genre", "pianos", "CNPM", "bel_canto",
        "chest_falsetto", "erhu_playing_tech", "GZ_IsoTech", "Guzheng_Tech99"]
LIBS = sorted(LIBS, key=len, reverse=True)


def load(arm, bb, seed, lib):
    p = os.path.join(RUNS, "%s_%s_%s" % (lib, arm, bb), "seed%d_qtest.npz" % seed)
    z = np.load(p, allow_pickle=True)
    return z["probs"].astype(np.float32), z["y"].astype(np.int64), z["records"].astype(str)


def macro_f1(conf):
    tp = np.diag(conf).astype(float)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / den, np.nan)
    return float(np.nanmean(f1))


def conf_slice(y, pred, nc):
    conf = np.zeros((nc, nc), dtype=np.float64)
    np.add.at(conf, (y, pred), 1.0)
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--arms", default="A1,A2")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    os.makedirs(OUT, exist_ok=True)

    # ---- build the data mapping in the directory order analyze_e1 resolves
    data = {}
    for name in os.listdir(RUNS):
        path = os.path.join(RUNS, name)
        if not os.path.isdir(path):
            continue
        lib = next((L for L in LIBS if name.startswith(L + "_")), None)
        if lib is None:
            continue
        rest = name[len(lib) + 1:].split("_")
        if len(rest) != 2:
            continue
        arm, bb = rest
        seeds = sorted(int(f.split("seed")[1].split("_")[0])
                       for f in os.listdir(path) if f.startswith("seed") and "_qtest.npz" in f)
        data.setdefault(lib, {}).setdefault((arm, bb), {})[bb] = seeds
        data[lib].setdefault((arm, bb), {})[bb] = seeds

    # CI from the old CSV, used to verify "same set of resamples"
    old = {}
    with open(os.path.join(RES, "e1_delta.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            old[(r["library"], r["bb"], r["contrast"])] = r

    rng = np.random.default_rng(a.seed)
    rows = []
    for lib in sorted(data):
        per_arm = {}
        for (arm, bb), seeds in data[lib].items():
            for bb2, ss in seeds.items():
                for s in ss:
                    probs, y, records = load(arm, bb2, s, lib)
                    per_arm.setdefault((arm, bb2), {})[s] = (probs.argmax(1), y)
        if "A0" not in {k[0] for k in per_arm}:
            continue
        ref = next(iter(per_arm.values()))[next(iter(next(iter(per_arm.values())).keys()))]
        y_ref = ref[1]
        n = len(y_ref)
        nrec = int(len(np.unique(next(iter(per_arm.values()))[next(iter(next(iter(per_arm.values())).keys()))][0]))) if False else None
        records_ref = load("A0", "alexnet", 0, lib)[2]
        nrec = len(dict.fromkeys(records_ref.tolist()))
        nc = int(max(y_ref.max(), ref[0].max())) + 1
        recs = list(dict.fromkeys(records_ref.tolist()))
        rmap = {r: i for i, r in enumerate(recs)}
        rec_idx = np.array([rmap[r] for r in records_ref], dtype=np.int64)

        pick = rng.integers(0, nrec, size=(a.boot, nrec))
        mult = np.zeros((a.boot, nrec), dtype=np.float64)
        for b in range(a.boot):
            mult[b] = np.bincount(pick[b], minlength=nrec)
        w_slice = mult[:, rec_idx]

        def f1_boot(pred):
            flat = y_ref * nc + pred
            wsum = np.zeros((a.boot, nc * nc))
            for b in range(a.boot):
                wsum[b] = np.bincount(flat, weights=w_slice[b], minlength=nc * nc)
            confs = wsum.reshape(a.boot, nc, nc)
            return np.array([macro_f1(confs[b]) for b in range(a.boot)])

        cache = {}
        for (arm, bb), seeds in per_arm.items():
            for s, (pred, _y) in seeds.items():
                cache[(arm, bb, s)] = f1_boot(pred)

        bbs = sorted({k[1] for k in per_arm})
        for bb in bbs:
            base_seeds = sorted(s for (arm2, bb2), ss in per_arm.items()
                                if arm2 == "A0" and bb2 == bb for s in ss)
            for arm in arms:
                seeds = sorted(set(base_seeds) & set(s for (a2, b2), ss in per_arm.items()
                                                     if a2 == arm and b2 == bb for s in ss))
                if not seeds:
                    continue
                d = np.mean([cache[(arm, bb, s)] - cache[("A0", bb, s)] for s in seeds], axis=0)
                pt_pt = 100 * float(np.mean([
                    macro_f1(conf_slice(per_arm[(arm, bb)][s][1], per_arm[(arm, bb)][s][0], nc))
                    - macro_f1(conf_slice(per_arm[("A0", bb)][s][1], per_arm[("A0", bb)][s][0], nc))
                    for s in seeds]))
                pt = pt_pt
                lo, hi = np.percentile(100 * d, [2.5, 97.5])
                n_le = int((d <= 0).sum())
                n_ge = int((d >= 0).sum())
                p_add = 2.0 * (1 + min(n_le, n_ge)) / (a.boot + 1.0)
                p_add = min(p_add, 1.0)
                key = (lib, bb, "%s-A0" % arm)
                o = old.get(key)
                match = None
                if o is not None:
                    match = (abs(round(pt, 2) - float(o["delta_slice_pt"])) < 0.011
                             and abs(round(lo, 2) - float(o["ci_lo_slice_pt"])) < 0.011
                             and abs(round(hi, 2) - float(o["ci_hi_slice_pt"])) < 0.011)
                rows.append({
                    "library": lib, "bb": bb, "arm": arm, "contrast": "%s-A0" % arm,
                    "point_pt": round(pt, 4),
                    "ci_lo_pt_reproduced": round(float(lo), 4),
                    "ci_hi_pt_reproduced": round(float(hi), 4),
                    "ci_matches_old_csv": match,
                    "n_draws_le0": n_le, "n_draws_ge0": n_ge,
                    "p_addone_raw": round(p_add, 6),
                    "p_legacy_clamped": (round(min(max(2 * min(n_le, n_ge) / a.boot, 1.0 / a.boot), 1.0), 6)
                                         if n_le and n_ge else 1.0 / a.boot),
                    "boot": a.boot,
                })
                print("%-11s %-8s %-3s pt %+6.2f ci [%+6.2f,%+6.2f] old_match=%s  add-one p=%.4f (legacy %.4f)"
                      % (lib, bb, arm, pt, lo, hi, match, p_add, rows[-1]["p_legacy_clamped"]))

    # ---- Holm: confirmatory family = CTIS x {alexnet, swin_t} x A1-A0
    fam = [r for r in rows if r["library"] == "CTIS" and r["arm"] == "A1"]
    fam_sorted = sorted(fam, key=lambda r: r["p_addone_raw"])
    m = len(fam_sorted)
    for i, r in enumerate(fam_sorted):
        r["holm_addone"] = round(min(1.0, r["p_addone_raw"] * (m - i)), 6)

    cols = ["library", "bb", "arm", "contrast", "point_pt", "ci_lo_pt_reproduced",
            "ci_hi_pt_reproduced", "ci_matches_old_csv", "n_draws_le0", "n_draws_ge0",
            "p_addone_raw", "p_legacy_clamped", "holm_addone", "boot"]
    csv_p = os.path.join(OUT, "slice_p_addone.csv")
    with open(csv_p, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    json_p = os.path.join(OUT, "slice_p_addone.json")
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "boot": a.boot, "seed": a.seed,
                   "family_definition": "CTIS × {alexnet, swin_t} × A1−A0",
                   "note": "the CI only verifies that the random stream matches e1_delta.csv; existing CIs are not replaced"},
                  f, ensure_ascii=False, indent=1, default=str)
    print("\nwrote", csv_p)
    print("wrote", json_p)


if __name__ == "__main__":
    main()
