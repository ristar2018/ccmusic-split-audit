# -*- coding: utf-8 -*-
"""Design-review arithmetic: GPU-hour accounting + minimum detectable effect (MDE) + class-frequency statistics.

No model is trained; this is arithmetic over the measured throughput, the slice counts and the leakage-audit results.
Writes results/design/design_audit.json plus a few CSVs.

Usage:
    python code/design_audit.py            # everything
    python code/design_audit.py budget     # GPU hours only
    python code/design_audit.py mde        # MDE only
"""
import argparse
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = r"D:\VSCode\Datasets"
OUT = os.path.join(ROOT, "results", "design")

# measured throughput (img/s, 224², AMP fp16, bs 32/64) — from the fapn measurement, see design note §6.1
THROUGHPUT = {"alexnet": 339.0, "vgg19_bn": 167.0, "swin_t": 341.0}

# mel slice count per library (= number of images under data/eval/<feature>; pianos has a single feature)
SLICES = {
    "CTIS": 43054, "music_genre": 36375, "pianos": 18352, "bel_canto": 9910,
    "GZ_IsoTech": 2899, "CNPM": 2724, "chest_falsetto": 1280,
    "erhu_playing_tech": 1253, "Guzheng_Tech99": 99,
}
# libraries that have all three feature sets: mel, cqt, chroma
MULTI_FEATURE = ["CTIS", "music_genre", "CNPM", "bel_canto", "chest_falsetto", "GZ_IsoTech"]

EPOCHS = 10


def per_config_hours(datasets, backbones, epochs=EPOCHS, repeats=1):
    """GPU·h needed to run every given library × every given backbone once (repeats configurations, e.g. 3 seeds × 2 protocols)."""
    imgs = sum(SLICES[d] for d in datasets) * epochs
    hours = {}
    for b in backbones:
        hours[b] = imgs / THROUGHPUT[b] / 3600.0 * repeats
    return hours


def budget():
    rows = []
    # ---- v1.0 grid: 9 libraries × 3 backbones × (3 seeds × 2 protocols) = 18 configurations ----
    mel_all = list(SLICES)
    h = per_config_hours(mel_all, list(THROUGHPUT), repeats=18)
    rows.append(("v1.0 E1 main grid (mel, 9 libraries × 3 backbones × 6 configurations)", sum(h.values())))
    # CQT add-on (CNPM, CTIS)
    h = per_config_hours(["CTIS", "CNPM"], list(THROUGHPUT), repeats=18)
    rows.append(("v1.0 E1 CQT add-on (same 18 configurations)", sum(h.values())))
    # equal-budget control (P-orig subsampled to the P-source scale, 3 backbones × 3 seeds = 9 configurations)
    h = per_config_hours(mel_all, list(THROUGHPUT), repeats=9)
    rows.append(("v1.0 E1 equal-budget control (9 configurations)", sum(h.values())))
    # P-proxy (4 libraries with K≈1 × 3 backbones × 3 seeds = 9 configurations)
    proxy = ["GZ_IsoTech", "chest_falsetto", "erhu_playing_tech", "Guzheng_Tech99"]
    h = per_config_hours(proxy, list(THROUGHPUT), repeats=9)
    rows.append(("v1.0 E1 P-proxy (4 libraries × 9 configurations)", sum(h.values())))
    v10 = sum(v for _, v in rows)

    # ---- v1.1 tiered grid ----
    tier1 = ["CTIS", "music_genre", "pianos", "bel_canto", "CNPM"]
    tier2 = ["GZ_IsoTech", "chest_falsetto", "erhu_playing_tech"]
    v11 = []
    # T1: 3 backbones × {orig,source} × 2 seeds = 12 configurations
    h = per_config_hours(tier1, list(THROUGHPUT), repeats=12)
    v11.append(("T1 five main libraries: 3 backbones × 2 protocols × 2 seeds (12 configurations)", sum(h.values())))
    # T2: 2 backbones (alexnet, swin_t) × 2 protocols × 2 seeds = 8 configurations
    h = per_config_hours(tier2, ["alexnet", "swin_t"], repeats=8)
    v11.append(("T2 three thin libraries: 2 backbones × 2 protocols × 2 seeds (8 configurations)", sum(h.values())))
    # feature control: 2 backbones × 2 protocols × 1 seed = 4 configurations, one round each for CQT and chroma
    h = per_config_hours(["CTIS", "music_genre", "CNPM"], ["alexnet", "swin_t"], repeats=4)
    v11.append(("T3 feature control CQT (CTIS/music_genre/CNPM, 4 configurations)", 2 * sum(h.values())))
    # P-proxy: 2 backbones × 3 seeds = 6 configurations
    h = per_config_hours(proxy, ["alexnet", "swin_t"], repeats=6)
    v11.append(("T4 P-proxy (4 libraries × 6 configurations)", sum(h.values())))
    # equal-budget control: 2 backbones × 2 protocols × 2 seeds = 8 configurations (only the 3 libraries with the largest Δ)
    h = per_config_hours(["CTIS", "music_genre", "bel_canto"], ["alexnet", "swin_t"], repeats=8)
    v11.append(("T5 equal-budget control (3 libraries, 8 configurations)", sum(h.values())))
    v11_total = sum(v for _, v in v11)

    out = {
        "throughput": THROUGHPUT,
        "slices": SLICES,
        "v1_0": {"items": [{"name": n, "gpu_h": round(v, 1)} for n, v in rows], "total": round(v10, 1)},
        "v1_1": {"items": [{"name": n, "gpu_h": round(v, 1)} for n, v in v11], "total": round(v11_total, 1)},
    }
    return out


def _mde_points(p, k, n_sources, rho=0.5, seed_pt=0.4, n_seeds=2,
                alpha=0.05, power=0.80):
    """Minimum detectable effect of a paired comparison (percentage points), plus the shrinkage factor of the "pseudo-independent" CI.

    Three error sources:
      1) test-set sampling (clustered by source): n_sources sources with k slices each, within-source accuracy correlation ρ;
      2) training randomness: the metric standard deviation across seeds, seed_pt (percentage points, 0.4 empirically);
      3) the paired comparison itself cancels the share of variance common to the arms.
    Returns (MDE_paired, CI_source-cluster half-width, CI_pseudo-independent half-width).
    """
    z = 1.959964 + 0.8416212
    var_p = p * (1 - p)
    # accuracy CI: clustered by source (ρ = within-source correlation) vs pretending every slice is independent
    ci_src = 1.959964 * math.sqrt(2 * var_p * (1 + (k - 1) * rho) / n_sources / k)
    ci_naive = 1.959964 * math.sqrt(2 * var_p / (n_sources * k))
    # paired Δ: the within-source shared part cancels, leaving per-source binomial noise + seed noise
    se_pair = math.sqrt(2 * var_p / k / n_sources + (seed_pt / 100.0) ** 2 / n_seeds)
    return z * se_pair * 100.0, ci_src * 100.0, ci_naive * 100.0


def _mcnemar_mde(n, p_disc=0.20, alpha=0.05, power=0.80):
    """Paired binary comparison (McNemar): the smallest difference (percentage points) detectable for a given discordance rate and sample size."""
    se = math.sqrt(p_disc / n)
    return (1.959964 + 0.8416212) * se * 100.0


def mde():
    csv_path = os.path.join(ROOT, "data", "splits", "_audit_summary.csv")
    base_acc = {
        "chest_falsetto": 0.994, "erhu_playing_tech": 0.994, "CTIS": 0.980,
        "bel_canto": 0.948, "CNPM": 0.859, "Guzheng_Tech99": 0.862, "GZ_IsoTech": 0.855,
        "pianos": 0.90, "music_genre": 0.80,
    }
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ds = r["dataset"]
            try:
                n_src = int(r["sources"])
                slices = int(r["slices"])
                k = float(r["K"]) if r.get("K") else slices / max(n_src, 1)
            except (KeyError, ValueError):
                continue
            p = base_acc.get(ds, 0.90)
            mde_pair, ci_src, ci_naive = _mde_points(p, k, n_src)
            rows.append({
                "dataset": ds, "slices": slices, "sources": n_src, "K": round(k, 2),
                "baseline_acc": p,
                "mde_paired_pt": round(mde_pair, 2),
                "ci_src_pt": round(ci_src, 2),
                "ci_naive_pt": round(ci_naive, 2),
                "ci_inflation": round(ci_src / ci_naive, 2) if ci_naive else None,
            })
    paired = {
        "E2_test_496_real": round(_mcnemar_mde(496), 2),
        "E2_cross_session_5cls_n160": round(_mcnemar_mde(160), 2),
        "E3_287_songs_5fold": round(_mcnemar_mde(287), 2),
        "E4_few_group_230_slices": round(_mcnemar_mde(230), 2),
        "E1_chest_1280": round(_mcnemar_mde(1280), 2),
    }
    return {"per_dataset": rows, "paired_mcnemar": paired}


def class_histograms():
    """CTIS class frequencies + long-tail grouping (used to test the E4 few-group hypothesis)."""
    base = os.path.join(DATA, "CTIS", "data", "audio")
    counts = {}
    if os.path.isdir(base):
        for d in os.listdir(base):
            p = os.path.join(base, d)
            if os.path.isdir(p):
                counts[d] = sum(1 for f in os.listdir(p) if not f.startswith("."))
    vals = sorted(counts.values())
    groups = {
        "many_ge50": sum(1 for v in vals if v >= 50),
        "medium_10_49": sum(1 for v in vals if 10 <= v < 50),
        "few_lt10": sum(1 for v in vals if v < 10),
        "lt5": sum(1 for v in vals if v < 5),
        "eq1": sum(1 for v in vals if v == 1),
    }
    return {
        "n_classes": len(counts), "n_files": sum(vals),
        "min": vals[0] if vals else None, "median": vals[len(vals) // 2] if vals else None,
        "max": vals[-1] if vals else None, "groups": groups,
        "few_share_of_classes_pct": round(100 * groups["few_lt10"] / max(len(counts), 1), 1),
        "few_share_of_files_pct": round(100 * sum(v for v in vals if v < 10) / max(sum(vals), 1), 1),
        "counts": counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all", choices=["all", "budget", "mde", "hist"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    out = {}
    if args.what in ("all", "budget"):
        out["budget"] = budget()
        print("== GPU hours ==")
        for k in ("v1_0", "v1_1"):
            print(f"[{k}] total {out['budget'][k]['total']} GPU·h")
            for it in out["budget"][k]["items"]:
                print(f"   {it['gpu_h']:6.1f}  {it['name']}")
    if args.what in ("all", "mde"):
        out["mde"] = mde()
        print("\n== minimum detectable effect (percentage points) ==")
        for r in out["mde"]["per_dataset"]:
            print(f"   {r['dataset']:18s} n_src={r['sources']:5d} K={r['K']:6.2f} "
                  f"paired MDE={r['mde_paired_pt']:5.2f}  CI: source cluster±{r['ci_src_pt']:4.2f} vs pseudo-independent±{r['ci_naive_pt']:4.2f} "
                  f"({r['ci_inflation']}×)")
        for k, v in out["mde"]["paired_mcnemar"].items():
            print(f"   {k:34s} McNemar MDE={v} pt")
    if args.what in ("all", "hist"):
        out["ctis_hist"] = class_histograms()
        h = out["ctis_hist"]
        print(f"\n== CTIS class frequencies == {h['n_classes']} classes / {h['n_files']} files,"
              f"median={h['median']} min={h['min']} max={h['max']}, groups {h['groups']}")
    with open(os.path.join(OUT, "design_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    if "ctis_hist" in out:
        with open(os.path.join(OUT, "ctis_class_counts.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_dir", "n_files"])
            for k, v in sorted(out["ctis_hist"]["counts"].items(), key=lambda x: -x[1]):
                w.writerow([k, v])
    if "mde" in out:
        with open(os.path.join(OUT, "mde.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out["mde"]["per_dataset"][0]))
            w.writeheader()
            w.writerows(out["mde"]["per_dataset"])
    print(f"\nwrote {os.path.join(OUT, 'design_audit.json')}")


if __name__ == "__main__":
    main()
