"""Label co-occurrence structure: Jaccard, PMI, and a 2-D classical-MDS embedding.

Shows how the 12 editorial mood tags overlap at the song level, which is direct
evidence about the construct structure of the label space.
"""
import collections
import itertools
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")
NAMES = ["Nostalgic", "Fresh", "Romantic", "Melancholic", "Healing", "Relaxing",
         "Lonely", "Touching", "Exciting", "Joyful", "Peaceful", "Missing"]


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load(os.path.join(REPO, "data-processor", "songs.jsonl"))
    sets = collections.defaultdict(set)
    for r in rows:
        sets[r["label"]].add(r["id"])
    n_songs = len({r["id"] for r in rows})

    inter = np.zeros((12, 12), dtype=int)
    for i, a in enumerate(NAMES):
        for j, b in enumerate(NAMES):
            inter[i, j] = 0 if i == j else len(sets[a] & sets[b])
    sizes = np.array([len(sets[a]) for a in NAMES], dtype=float)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / union, 0.0)
    np.fill_diagonal(jac, np.nan)

    # pointwise mutual information between labels (song-level coverage)
    pmi = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            if i == j:
                continue
            pij = inter[i, j] / n_songs
            pmi[i, j] = math.log2(pij / ((sizes[i] / n_songs) * (sizes[j] / n_songs)))

    # classical MDS from Jaccard distance
    d = 1.0 - np.nan_to_num(jac, nan=1.0)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2
    n = 12
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (d ** 2) @ J
    w, v = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    coords = v[:, order[:2]] * np.sqrt(np.maximum(w[order[:2]], 0))

    rep = {
        "n_songs": n_songs,
        "label_sizes": dict(zip(NAMES, sizes.astype(int).tolist())),
        "cooccurrence": {NAMES[i]: {NAMES[j]: int(inter[i, j]) for j in range(12)} for i in range(12)},
        "jaccard": {NAMES[i]: {NAMES[j]: (None if i == j else round(float(jac[i, j]), 4))
                               for j in range(12)} for i in range(12)},
        "pmi_bits": {NAMES[i]: {NAMES[j]: (None if i == j else round(float(pmi[i, j]), 3))
                                for j in range(12)} for i in range(12)},
        "mds_2d": {NAMES[i]: [round(float(coords[i, 0]), 4), round(float(coords[i, 1]), 4)]
                   for i in range(12)},
        "mds_eig_share": [round(float(x), 4) for x in (w[order[:2]] / w.sum())],
    }

    # strongest / weakest pairs
    pairs = []
    for i, j in itertools.combinations(range(12), 2):
        pairs.append((NAMES[i], NAMES[j], round(float(jac[i, j]), 4), round(float(pmi[i, j]), 3),
                      int(inter[i, j])))
    pairs.sort(key=lambda t: -t[2])
    rep["top_jaccard_pairs"] = pairs[:8]
    rep["bottom_jaccard_pairs"] = pairs[-8:]
    rep["mean_offdiag_jaccard"] = round(float(np.nanmean(jac)), 4)

    with open(os.path.join(OUT, "cooccurrence.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)

    print("mean off-diagonal Jaccard:", rep["mean_offdiag_jaccard"])
    print("\ntop pairs (Jaccard, PMI, shared songs):")
    for p in pairs[:8]:
        print("  ", p)
    print("\nweakest pairs:")
    for p in pairs[-6:]:
        print("  ", p)
    print("\nMDS coords:")
    for k, v in rep["mds_2d"].items():
        print(f"   {k:12s} {v}")
    print("eig share:", rep["mds_eig_share"])


if __name__ == "__main__":
    main()
