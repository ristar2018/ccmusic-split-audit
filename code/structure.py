"""Is the 12-tag editorial taxonomy organised like an affective circumplex?

Tests, from label co-occurrence only:
  1. average-linkage clustering of the Jaccard distance;
  2. within- vs between-cluster Jaccard for a hypothesised quadrant grouping;
  3. correlation between observed similarity and circumplex (valence, arousal) similarity.
"""
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")
NAMES = ["Nostalgic", "Fresh", "Romantic", "Melancholic", "Healing", "Relaxing",
         "Lonely", "Touching", "Exciting", "Joyful", "Peaceful", "Missing"]

# hypothesised circumplex coordinates: (valence, arousal) on [-2, 2]
CIRCUMPLEX = {
    "Joyful": (2.0, 0.8), "Exciting": (1.2, 2.0), "Fresh": (1.5, 1.2),
    "Touching": (1.4, -0.6), "Romantic": (1.0, -0.8), "Healing": (1.0, -1.6),
    "Relaxing": (1.2, -1.6), "Peaceful": (0.8, -2.0), "Nostalgic": (0.2, -0.6),
    "Missing": (-1.4, -1.0), "Lonely": (-1.8, -1.2), "Melancholic": (-2.0, -1.0),
}
QUADRANTS = {
    "high_arousal_positive": ["Joyful", "Exciting", "Fresh"],
    "low_arousal_positive": ["Healing", "Relaxing", "Peaceful", "Romantic", "Touching", "Nostalgic"],
    "low_arousal_negative": ["Melancholic", "Lonely", "Missing"],
}


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    rows = load(os.path.join(REPO, "data-processor", "songs.jsonl"))
    sets = {n: set() for n in NAMES}
    for r in rows:
        sets[r["label"]].add(r["id"])
    inter = np.array([[len(sets[a] & sets[b]) for b in NAMES] for a in NAMES], dtype=float)
    sizes = np.array([len(sets[a]) for a in NAMES], dtype=float)
    union = sizes[:, None] + sizes[None, :] - inter
    jac = inter / union
    np.fill_diagonal(jac, np.nan)

    # ---- PMI ranking -------------------------------------------------------
    n = len({r["id"] for r in rows})
    pmi = np.zeros_like(inter)
    for i in range(12):
        for j in range(12):
            if i != j:
                pmi[i, j] = np.log2((inter[i, j] / n) / ((sizes[i] / n) * (sizes[j] / n)))
    pairs = [(NAMES[i], NAMES[j], float(pmi[i, j]), float(jac[i, j]))
             for i, j in itertools.combinations(range(12), 2)]
    pairs.sort(key=lambda t: -t[2])

    # ---- average-linkage clustering (pure numpy) ---------------------------
    d = 1 - np.nan_to_num(jac, nan=1.0)
    np.fill_diagonal(d, 0.0)
    clusters = {i: [NAMES[i]] for i in range(12)}
    dist = d.copy()
    order = []
    active = list(range(12))
    while len(active) > 1:
        best = None
        for a, b in itertools.combinations(active, 2):
            if best is None or dist[a, b] < best[0]:
                best = (dist[a, b], a, b)
        dd, a, b = best
        order.append({"merge_distance": round(float(dd), 4),
                      "left": clusters[a], "right": clusters[b]})
        merged = clusters[a] + clusters[b]
        keep, drop = min(a, b), max(a, b)
        clusters[keep] = merged
        # average linkage update
        for c in active:
            if c in (a, b):
                continue
            new = (dist[a, c] * len(clusters[a]) + dist[b, c] * len(clusters[b])) / (
                len(clusters[a]) + len(clusters[b]))
            dist[keep, c] = dist[c, keep] = new
        active.remove(drop)
        del clusters[drop]

    # ---- within / between ------------------------------------------------
    def mean_jac(group_a, group_b):
        vals = [jac[NAMES.index(a), NAMES.index(b)] for a in group_a for b in group_b if a != b]
        return float(np.nanmean(vals)) if vals else float("nan")

    within, between = {}, {}
    for name, grp in QUADRANTS.items():
        within[name] = round(mean_jac(grp, grp), 4)
    for a, b in itertools.combinations(QUADRANTS, 2):
        between[f"{a}|{b}"] = round(mean_jac(QUADRANTS[a], QUADRANTS[b]), 4)
    multi = [g for g in QUADRANTS.values() if len(g) > 3][0]
    # split the broad low-arousal-positive group in two for a fairer comparison
    SUB = {
        "warm_low_arousal": ["Romantic", "Touching"],
        "calm_low_arousal": ["Healing", "Relaxing", "Peaceful", "Nostalgic"],
    }
    within_sub = {k: round(mean_jac(v, v), 4) for k, v in SUB.items()}
    between_sub = round(mean_jac(SUB["warm_low_arousal"], SUB["calm_low_arousal"]), 4)

    # ---- circumplex correlation ------------------------------------------
    xs, ys = [], []
    for i, j in itertools.combinations(range(12), 2):
        vi = np.array(CIRCUMPLEX[NAMES[i]])
        vj = np.array(CIRCUMPLEX[NAMES[j]])
        xs.append(-float(np.linalg.norm(vi - vj)))  # negative distance = similarity
        ys.append(float(jac[i, j]))
    xs, ys = np.array(xs), np.array(ys)
    r = float(np.corrcoef(xs, ys)[0, 1])
    rho = float(np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1])

    rep = {
        "top_pmi_pairs": pairs[:6],
        "bottom_pmi_pairs": pairs[-6:],
        "cluster_merges": order[:6],
        "quadrant_within_jaccard": within,
        "quadrant_between_jaccard": between,
        "refined_within_jaccard": within_sub,
        "refined_between_jaccard": between_sub,
        "circumplex_pearson_r": round(r, 4),
        "circumplex_spearman_rho": round(rho, 4),
    }
    with open(os.path.join(OUT, "structure.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
