"""Statistical audit of the emo163 label tables.

Produces results/audit.json with everything the report needs: sizes, label
distributions, song/playlist multiplicity, and the amount of *irreducible*
label ambiguity plus split leakage.
"""
import collections
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")
NAMES = [
    "Nostalgic", "Fresh", "Romantic", "Melancholic", "Healing", "Relaxing",
    "Lonely", "Touching", "Exciting", "Joyful", "Peaceful", "Missing",
]
SPLITS = ["train", "valid", "test"]


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def entropy(counter):
    total = sum(counter.values())
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {s: load(os.path.join(REPO, "data", f"songs-{s}.jsonl")) for s in SPLITS}
    master = load(os.path.join(REPO, "data-processor", "songs.jsonl"))

    rep = {"file_rows": {s: len(data[s]) for s in SPLITS}, "master_rows": len(master)}
    rep["row_total"] = sum(rep["file_rows"].values())

    # --- schema / field check -------------------------------------------------
    keysets = collections.Counter(tuple(sorted(r.keys())) for s in SPLITS for r in data[s])
    rep["field_sets"] = {" | ".join(k): v for k, v in keysets.items()}
    rep["id_dtype"] = collections.Counter(type(r["id"]).__name__ for r in master).most_common()
    rep["label_dtype"] = collections.Counter(type(r["label"]).__name__ for r in master).most_common()

    # --- label distribution ---------------------------------------------------
    lab_all = collections.Counter(r["label"] for s in SPLITS for r in data[s])
    rep["label_counts"] = {k: lab_all.get(k, 0) for k in NAMES}
    rep["label_counts_other"] = {k: v for k, v in lab_all.items() if k not in NAMES}
    rep["label_share"] = {k: round(v / rep["row_total"], 5) for k, v in rep["label_counts"].items()}
    rep["label_entropy_bits"] = round(entropy(lab_all), 4)
    rep["label_imbalance_ratio"] = round(max(lab_all.values()) / min(lab_all.values()), 3)
    rep["per_split_label_counts"] = {
        s: {k: collections.Counter(r["label"] for r in data[s]).get(k, 0) for k in NAMES}
        for s in SPLITS
    }

    # --- entity cardinality ---------------------------------------------------
    songs = {r["id"] for r in master}
    playlists = {r["playlist"] for r in master}
    rep["unique_songs_master"] = len(songs)
    rep["unique_playlists_master"] = len(playlists)
    rep["unique_songs_splits"] = {s: len({r["id"] for r in data[s]}) for s in SPLITS}
    rep["unique_playlists_splits"] = {s: len({r["playlist"] for r in data[s]}) for s in SPLITS}
    rep["rows_per_song_master"] = round(len(master) / len(songs), 3)
    rep["songs_per_playlist"] = round(len(master) / len(playlists), 3)

    # rows per playlist distribution
    pp = collections.Counter(r["playlist"] for r in master)
    vals = sorted(pp.values())
    rep["playlist_size_stats"] = {
        "min": vals[0], "p25": vals[len(vals) // 4], "median": vals[len(vals) // 2],
        "p75": vals[3 * len(vals) // 4], "p95": vals[int(len(vals) * 0.95)],
        "max": vals[-1], "mean": round(sum(vals) / len(vals), 2),
    }

    # --- label multiplicity per song (ambiguity) ------------------------------
    song_labels = collections.defaultdict(collections.Counter)
    for r in master:
        song_labels[r["id"]][r["label"]] += 1
    multi = sum(1 for c in song_labels.values() if len(c) > 1)
    rep["songs_multi_label"] = multi
    rep["songs_multi_label_share"] = round(multi / len(songs), 4)
    hist = collections.Counter(len(c) for c in song_labels.values())
    rep["labels_per_song_hist"] = dict(sorted(hist.items()))
    # majority-label conditional entropy: irreducible ambiguity of a per-song task
    cond = collections.Counter()
    for c in song_labels.values():
        n = sum(c.values())
        cond.update(c)
        # accumulate weighted entropy later
    cond_h = 0.0
    for c in song_labels.values():
        n = sum(c.values())
        cond_h += (n / len(master)) * entropy(c)
    rep["conditional_entropy_label_given_song"] = round(cond_h, 4)
    rep["majority_baseline_acc"] = round(
        sum(max(c.values()) for c in song_labels.values()) / len(master), 4
    )

    # --- split leakage --------------------------------------------------------
    S = {s: {r["id"] for r in data[s]} for s in SPLITS}
    P = {s: {r["playlist"] for r in data[s]} for s in SPLITS}
    rep["song_overlap"] = {
        f"{a}_{b}": len(S[a] & S[b]) for a in SPLITS for b in SPLITS if a < b
    }
    rep["song_overlap_share_of_test"] = {
        f"{a}_{b}": round(len(S[a] & S[b]) / len(S["test"]), 4)
        for a in SPLITS for b in SPLITS if a < b
    }
    rep["playlist_overlap"] = {
        f"{a}_{b}": len(P[a] & P[b]) for a in SPLITS for b in SPLITS if a < b
    }
    rep["playlist_overlap_share_of_test"] = {
        f"{a}_{b}": round(len(P[a] & P[b]) / len(P["test"]), 4)
        for a in SPLITS for b in SPLITS if a < b
    }
    # test rows whose song also appears in train
    train_songs = S["train"]
    rep["test_rows_song_in_train"] = sum(1 for r in data["test"] if r["id"] in train_songs)
    rep["test_rows_song_in_train_share"] = round(
        rep["test_rows_song_in_train"] / len(data["test"]), 4
    )
    train_pl = P["train"]
    rep["test_rows_playlist_in_train"] = sum(1 for r in data["test"] if r["playlist"] in train_pl)
    rep["test_rows_playlist_in_train_share"] = round(
        rep["test_rows_playlist_in_train"] / len(data["test"]), 4
    )

    # --- leakage-free re-splitability ----------------------------------------
    # group-aware split by song with 80/10/10 on unique songs
    ids = sorted(songs)
    rep["group_split_song"] = {
        "n_groups": len(ids), "train": int(0.8 * len(ids)),
        "valid": int(0.1 * len(ids)), "test": len(ids) - int(0.9 * len(ids)),
    }
    # a strict single-label subset: keep only songs with exactly one label
    single = [sid for sid, c in song_labels.items() if len(c) == 1]
    rep["songs_single_label"] = len(single)
    rep["songs_single_label_share"] = round(len(single) / len(songs), 4)
    sl_rows = sum(sum(song_labels[sid].values()) for sid in single)
    rep["rows_in_single_label_subset"] = sl_rows
    rep["single_label_subset_dist"] = dict(collections.Counter(
        next(iter(song_labels[sid])) for sid in single
    ))

    # --- master vs published splits ------------------------------------------
    mset = {(r["id"], r["playlist"], r["label"]) for r in master}
    sset = {(r["id"], r["playlist"], r["label"]) for s in SPLITS for r in data[s]}
    rep["master_unique_triples"] = len(mset)
    rep["split_unique_triples"] = len(sset)
    rep["master_minus_splits"] = len(mset - sset)
    rep["duplicate_rows_in_splits"] = rep["row_total"] - len(sset)
    rep["master_duplicate_rows"] = len(master) - len(mset)

    with open(os.path.join(OUT, "audit.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
