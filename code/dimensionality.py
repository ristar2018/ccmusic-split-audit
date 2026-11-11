"""Effective dimensionality of the 12-tag label space and cluster-level ambiguity.

1. SVD of the song x label indicator matrix: how many independent dimensions does
   the taxonomy actually span?
2. Collapsing tags into k affective clusters: how often does one song receive
   tags from *different* clusters (which bounds any coarse-grained task)?
"""
import collections
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")
NAMES = ["Nostalgic", "Fresh", "Romantic", "Melancholic", "Healing", "Relaxing",
         "Lonely", "Touching", "Exciting", "Joyful", "Peaceful", "Missing"]
IDX = {n: i for i, n in enumerate(NAMES)}

CLUSTERINGS = {
    "k3": {
        "negative_low_arousal": ["Melancholic", "Lonely", "Missing", "Nostalgic"],
        "calm_positive": ["Healing", "Relaxing", "Peaceful", "Romantic", "Touching"],
        "high_arousal_positive": ["Exciting", "Joyful", "Fresh"],
    },
    "k4": {
        "negative": ["Melancholic", "Lonely", "Missing"],
        "bittersweet": ["Nostalgic", "Touching"],
        "calm_positive": ["Healing", "Relaxing", "Peaceful", "Romantic"],
        "high_arousal_positive": ["Exciting", "Joyful", "Fresh"],
    },
    "k2": {
        "negative": ["Melancholic", "Lonely", "Missing", "Nostalgic"],
        "positive": ["Healing", "Relaxing", "Peaceful", "Romantic", "Touching",
                     "Exciting", "Joyful", "Fresh"],
    },
}


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    rows = load(os.path.join(REPO, "data-processor", "songs.jsonl"))
    songs = sorted({r["id"] for r in rows})
    sid = {s: i for i, s in enumerate(songs)}
    X = np.zeros((len(songs), 12), dtype=np.float32)
    for r in rows:
        X[sid[r["id"]], IDX[r["label"]]] = 1.0

    cov = (X.T @ X) / len(songs)
    w = np.linalg.eigvalsh(cov)[::-1]
    share = w / w.sum()
    rep = {
        "n_songs": len(songs),
        "eigenvalues": [round(float(x), 4) for x in w],
        "variance_share": [round(float(x), 4) for x in share],
        "cum_share": [round(float(x), 4) for x in np.cumsum(share)],
        "n_components_80pct": int(np.searchsorted(np.cumsum(share), 0.80) + 1),
        "n_components_90pct": int(np.searchsorted(np.cumsum(share), 0.90) + 1),
        "participation_ratio": round(float(w.sum() ** 2 / (w ** 2).sum()), 3),
        "mean_labels_per_song": round(float(X.sum(1).mean()), 3),
    }

    # cluster-level analysis
    song_labels = collections.defaultdict(set)
    for r in rows:
        song_labels[r["id"]].add(r["label"])
    rep["clusterings"] = {}
    for name, groups in CLUSTERINGS.items():
        lab2grp = {lab: g for g, labs in groups.items() for lab in labs}
        spans = []
        for s, labs in song_labels.items():
            gs = {lab2grp[l] for l in labs}
            if len(gs) > 1:
                spans.append(s)
        # residual ambiguity inside a single cluster: songs with >1 tag in the same group
        intra = sum(1 for s, labs in song_labels.items()
                    if len(labs) > 1 and len({lab2grp[l] for l in labs}) == 1)
        rep["clusterings"][name] = {
            "groups": groups,
            "songs_spanning_clusters": len(spans),
            "share_spanning_clusters": round(len(spans) / len(songs), 4),
            "songs_multi_tag_within_one_cluster": intra,
            "share_multi_tag_within_one_cluster": round(intra / len(songs), 4),
            "clean_single_cluster_songs": len(songs) - len(spans) - 0,
            "share_clean_single_cluster": round((len(songs) - len(spans)) / len(songs), 4),
        }
        # majority-vote accuracy if the true target were the song's dominant cluster
        grp_count = collections.Counter()
        for s, labs in song_labels.items():
            c = collections.Counter(lab2grp[l] for l in labs)
            pass
        row_grp = collections.Counter()
        correct = 0
        for r in rows:
            g = lab2grp[r["label"]]
            row_grp[g] += 1
        rep["clusterings"][name]["rows_per_cluster"] = dict(row_grp)
        rep["clusterings"][name]["majority_cluster_share"] = round(
            max(row_grp.values()) / len(rows), 4)

    with open(os.path.join(OUT, "dimensionality.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
