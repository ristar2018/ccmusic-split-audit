"""Pilot 3b: mechanism reproduction with a *label-correlated* synthetic corpus.

True process: songs belong to 1-2 label communities, playlists are curated mostly
from the matching community. Release process: keep one row per (song, label).

We measure, on the true graph and on the released graph: edge count, label
homophily, and 1-hop label-propagation accuracy for held-out playlists.
"""
import collections
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")


def simulate(seed, n_playlists=2000, n_songs=20000, n_labels=12,
             p_two_labels=0.40, within_label_share=0.85, mean_songs=60):
    rng = random.Random(seed)

    # songs belong to one or two label communities, Zipf-popular
    song_labels = {}
    for i in range(n_songs):
        if rng.random() < p_two_labels:
            song_labels[i] = {rng.randrange(n_labels), rng.randrange(n_labels)}
        else:
            song_labels[i] = {rng.randrange(n_labels)}
    by_label = collections.defaultdict(list)
    for s, ls in song_labels.items():
        for l in ls:
            by_label[l].append(s)

    # playlist curation
    playlist_label = {}
    song_pl = collections.defaultdict(list)
    for p in range(n_playlists):
        L = rng.randrange(n_labels)
        playlist_label[p] = L
        k = max(5, int(rng.gauss(mean_songs, mean_songs * 0.4)))
        for _ in range(k):
            if rng.random() < within_label_share and by_label[L]:
                s = rng.choice(by_label[L])
            else:
                s = rng.randrange(n_songs)
            song_pl[s].append((p, L))

    def edges_from(pairs):
        return collections.defaultdict(int, pairs)

    # --- true graph: every (song, playlist) incidence is kept
    true_pairs = collections.defaultdict(int)
    for s, v in song_pl.items():
        ps = sorted({p for p, _ in v})
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                true_pairs[(ps[i], ps[j])] += 1

    # --- released graph: one playlist per (song, label)
    kept = collections.defaultdict(dict)
    for s, v in song_pl.items():
        for p, L in v:
            kept[s][L] = p          # last writer wins, as in the published pipeline
    rel_pairs = collections.defaultdict(int)
    for s, d in kept.items():
        ps = sorted(d.values())
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                rel_pairs[(ps[i], ps[j])] += 1

    def homophily(pairs):
        if not pairs:
            return 0.0
        return sum(1 for (a, b) in pairs if playlist_label[a] == playlist_label[b]) / len(pairs)

    # --- 1-hop label propagation, 5-fold by playlist
    nbrs_true = collections.defaultdict(list)
    for (a, b), w in true_pairs.items():
        nbrs_true[a].append((b, w))
        nbrs_true[b].append((a, w))
    nbrs_rel = collections.defaultdict(list)
    for (a, b), w in rel_pairs.items():
        nbrs_rel[a].append((b, w))
        nbrs_rel[b].append((a, w))

    def propagate(nbrs, folds=5):
        correct = total = 0
        for f in range(folds):
            test = [p for p in playlist_label if p % folds == f]
            train = {p for p in playlist_label if p % folds != f}
            for p in test:
                score = collections.Counter()
                for q, w in nbrs.get(p, []):
                    if q in train:
                        score[playlist_label[q]] += w
                if not score:
                    pred = max(collections.Counter(playlist_label[q] for q in train).items(),
                               key=lambda t: t[1])[0]
                else:
                    pred = score.most_common(1)[0][0]
                total += 1
                correct += (pred == playlist_label[p])
        return 100.0 * correct / total

    lab_ct = collections.Counter(playlist_label.values())
    chance = sum((c / n_playlists) ** 2 for c in lab_ct.values())

    return {
        "seed": seed,
        "true_edges": len(true_pairs),
        "released_edges": len(rel_pairs),
        "edge_retention": round(len(rel_pairs) / max(1, len(true_pairs)), 4),
        "true_homophily": round(homophily(true_pairs), 4),
        "released_homophily": round(homophily(rel_pairs), 4),
        "chance_agreement": round(chance, 4),
        "propagation_acc_true_graph": round(propagate(nbrs_true), 2),
        "propagation_acc_released_graph": round(propagate(nbrs_rel), 2),
    }


def main():
    rep = {"simulations": [simulate(s) for s in (1, 2, 3)]}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sim_graphdestruction.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
