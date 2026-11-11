"""Pilot 2: is there *any* honest signal? Playlist co-membership graph + label propagation.

The playlist-playlist graph connects two playlists when they share songs. Under a
playlist-grouped split this is the only information a model can exploit, so its
accuracy is the realistic ceiling of a weak-supervision approach on emo163.

Reported:
  * homophily  : P(two connected playlists share a label) vs the chance baseline
  * top-1/top-3 row accuracy of weighted neighbour voting, playlist-grouped 5-fold
"""
import collections
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def stable_hash(s, mod):
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 1000003
    return h % mod


def main():
    rows = load(os.path.join(REPO, "data-processor", "songs.jsonl"))
    pl_rows = collections.defaultdict(list)
    pl_songs = collections.defaultdict(set)
    song_pls = collections.defaultdict(set)
    for r in rows:
        pl_rows[r["playlist"]].append(r)
        pl_songs[r["playlist"]].add(r["id"])
        song_pls[r["id"]].add(r["playlist"])

    # dominant label per playlist
    pl_label = {}
    for p, rs in pl_rows.items():
        pl_label[p] = collections.Counter(r["label"] for r in rs).most_common(1)[0][0]

    # weighted playlist graph
    edges = collections.defaultdict(float)
    for sid, pls in song_pls.items():
        if len(pls) < 2:
            continue
        pls = sorted(pls)
        for i in range(len(pls)):
            for j in range(i + 1, len(pls)):
                edges[(pls[i], pls[j])] += 1.0

    # homophily
    same = sum(1 for (a, b) in edges if pl_label[a] == pl_label[b])
    total = len(edges)
    lab_ct = collections.Counter(pl_label.values())
    n_pl = len(pl_label)
    chance = sum((c / n_pl) ** 2 for c in lab_ct.values())

    nbrs = collections.defaultdict(list)
    for (a, b), w in edges.items():
        norm = w / math.sqrt(len(pl_songs[a]) * len(pl_songs[b]))
        nbrs[a].append((b, norm))
        nbrs[b].append((a, norm))

    deg = [len(v) for v in nbrs.values()]
    deg.sort()

    rep = {
        "n_playlists": n_pl,
        "n_edges": total,
        "edge_density": round(total / (n_pl * (n_pl - 1) / 2), 6),
        "degree_median": deg[len(deg) // 2],
        "degree_p90": deg[int(len(deg) * 0.9)],
        "degree_max": deg[-1],
        "homophily": round(same / total, 4),
        "chance_agreement": round(chance, 4),
        "homophily_lift": round((same / total) / chance, 2),
    }

    global_major = collections.Counter(pl_label.values()).most_common(1)[0][0]

    acc1 = acc3 = n_rows = 0
    cover_edges = 0
    for f in range(5):
        test_p = [p for p in pl_label if stable_hash(p, 5) == f]
        train_p = {p for p in pl_label if stable_hash(p, 5) != f}
        for p in test_p:
            score = collections.Counter()
            wsum = 0.0
            for q, w in nbrs.get(p, []):
                if q in train_p:
                    score[pl_label[q]] += w
                    wsum += w
            if not score:
                pred = global_major
                top3 = {global_major}
            else:
                ranked = [lab for lab, _ in score.most_common(3)]
                pred = ranked[0]
                top3 = set(ranked)
                cover_edges += 1
            for r in pl_rows[p]:
                n_rows += 1
                if r["label"] == pred:
                    acc1 += 1
                if r["label"] in top3:
                    acc3 += 1

    rep["label_propagation"] = {
        "rows": n_rows,
        "top1_acc": round(100 * acc1 / n_rows, 2),
        "top3_acc": round(100 * acc3 / n_rows, 2),
        "playlists_with_train_neighbours": round(cover_edges / n_pl, 4),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "pilot_graph.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
