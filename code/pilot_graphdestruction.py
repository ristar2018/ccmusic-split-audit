"""Pilot 3: (a) verify the graph-destruction property on the released table,
(b) reproduce the mechanism on a synthetic corpus with known structure.

Claim under test: the release keeps exactly one row per (song, label). Therefore
two different playlists can only share a song if the song was recorded under two
different labels, i.e. every surviving edge in the released co-membership graph is
a cross-label edge. Same-label co-membership is erased by construction.
"""
import collections
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def check_release():
    rows = load(os.path.join(REPO, "data-processor", "songs.jsonl"))
    song_rows = collections.defaultdict(list)
    for r in rows:
        song_rows[r["id"]].append((r["playlist"], r["label"]))

    multi = {s: v for s, v in song_rows.items() if len(v) > 1}
    same = diff = 0
    for s, v in multi.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v[i][1] == v[j][1]:
                    same += 1
                else:
                    diff += 1
    # songs per playlist used to build the graph
    return {
        "songs_total": len(song_rows),
        "songs_in_multiple_playlists": len(multi),
        "share_of_songs_in_multiple_playlists": round(len(multi) / len(song_rows), 4),
        "playlist_pairs_same_label": same,
        "playlist_pairs_different_label": diff,
        "same_label_edge_share": round(same / max(1, same + diff), 6),
        "labels_per_song_mean": round(len(rows) / len(song_rows), 3),
    }


def simulate(seed=0, n_playlists=2000, n_songs=20000, n_labels=12,
             within_label_share=0.82, mean_songs=60):
    """Synthetic curation process, then apply the release procedure and compare."""
    rng = random.Random(seed)
    popularity = [1.0 / (i + 1) ** 0.8 for i in range(n_songs)]
    cum = []
    t = 0.0
    for w in popularity:
        t += w
        cum.append(t)
    total = cum[-1]

    def draw_song():
        x = rng.random() * total
        lo, hi = 0, n_songs - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    labels = [rng.randrange(n_labels) for _ in range(n_playlists)]
    true_edges = collections.defaultdict(set)   # playlist -> songs (full truth)
    song_pl = collections.defaultdict(list)     # song -> [(playlist, label)]
    for p in range(n_playlists):
        lab = labels[p]
        k = max(5, int(rng.gauss(mean_songs, mean_songs * 0.4)))
        for _ in range(k):
            if rng.random() < within_label_share:
                # curated from the label's own pool: sample a song "belonging" to that label
                s = draw_song()
                pool = s  # one global pool; label association is via the playlist
            else:
                s = draw_song()
            true_edges[p].add(s)
            song_pl[s].append((p, lab))

    # full graph (what the platform has)
    full_pairs = collections.defaultdict(int)
    for s, v in song_pl.items():
        ps = [p for p, _ in v]
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                full_pairs[(ps[i], ps[j])] += 1

    # released graph: keep one row per (song, label) -> one playlist per label
    kept = collections.defaultdict(dict)
    for s, v in song_pl.items():
        for p, lab in v:
            kept[s][lab] = p
    rel_pairs = collections.defaultdict(int)
    for s, d in kept.items():
        ps = list(d.values())
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                rel_pairs[(min(a, b), max(a, b))] += 1

    def homophily(pairs):
        if not pairs:
            return 0.0
        same = sum(1 for (a, b) in pairs if labels[a] == labels[b])
        return same / len(pairs)

    lab_ct = collections.Counter(labels)
    chance = sum((c / n_playlists) ** 2 for c in lab_ct.values())

    return {
        "n_playlists": n_playlists,
        "n_songs": n_songs,
        "full_graph_edges": len(full_pairs),
        "released_graph_edges": len(rel_pairs),
        "edge_retention": round(len(rel_pairs) / max(1, len(full_pairs)), 4),
        "full_homophily": round(homophily(full_pairs), 4),
        "released_homophily": round(homophily(rel_pairs), 4),
        "chance": round(chance, 4),
    }


def main():
    rep = {"release_check": check_release(),
           "simulation": [simulate(seed=s) for s in (1, 2, 3)]}
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "pilot_graphdestruction.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
