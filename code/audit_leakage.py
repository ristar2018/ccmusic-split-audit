"""Second audit pass: leakage baselines, playlist purity, per-class coverage."""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")
NAMES = ["Nostalgic", "Fresh", "Romantic", "Melancholic", "Healing", "Relaxing",
         "Lonely", "Touching", "Exciting", "Joyful", "Peaceful", "Missing"]
SPLITS = ["train", "valid", "test"]


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    data = {s: load(os.path.join(REPO, "data", f"songs-{s}.jsonl")) for s in SPLITS}
    master = load(os.path.join(REPO, "data-processor", "songs.jsonl"))
    rep = {}

    # playlist -> set(labels); purity by construction check
    pl = collections.defaultdict(set)
    for r in master:
        pl[r["playlist"]].add(r["label"])
    rep["playlists_total"] = len(pl)
    rep["playlists_multi_label"] = sum(1 for v in pl.values() if len(v) > 1)
    rep["playlist_label_purity"] = round(
        sum(1 for v in pl.values() if len(v) == 1) / len(pl), 4)

    # playlists per label + songs per label
    pl_per_label = collections.Counter(next(iter(v)) for v in pl.values())
    songs_by_label = collections.defaultdict(set)
    for r in master:
        songs_by_label[r["label"]].add(r["id"])
    rep["playlists_per_label"] = dict(pl_per_label)
    rep["unique_songs_per_label"] = {k: len(v) for k, v in songs_by_label.items()}

    # ---- cheating baselines on the *official* random split ----
    train = data["train"]
    test = data["test"]
    pl2lab = {}
    for r in train:
        pl2lab.setdefault(r["playlist"], collections.Counter())[r["label"]] += 1
    pl2lab = {k: v.most_common(1)[0][0] for k, v in pl2lab.items()}
    song2lab = collections.defaultdict(collections.Counter)
    for r in train:
        song2lab[r["id"]][r["label"]] += 1
    song2lab = {k: v.most_common(1)[0][0] for k, v in song2lab.items()}
    majority = collections.Counter(r["label"] for r in train).most_common(1)[0][0]

    pl_hit = sum(1 for r in test if r["playlist"] in pl2lab)
    pl_correct = sum(1 for r in test if pl2lab.get(r["playlist"]) == r["label"])
    song_hit = sum(1 for r in test if r["id"] in song2lab)
    song_correct = sum(1 for r in test if song2lab.get(r["id"]) == r["label"])

    rep["official_split_baselines"] = {
        "majority_label": majority,
        "majority_acc": round(sum(1 for r in test if r["label"] == majority) / len(test), 4),
        "playlist_lookup_coverage": round(pl_hit / len(test), 4),
        "playlist_lookup_acc": round(pl_correct / len(test), 4),
        "song_lookup_coverage": round(song_hit / len(test), 4),
        "song_lookup_acc": round(song_correct / len(test), 4),
    }

    # ---- what a leakage-free (song-grouped) protocol would look like ----
    # hold out 10% of unique songs; oracle accuracy for a song is its majority label
    song_labels = collections.defaultdict(collections.Counter)
    for r in master:
        song_labels[r["id"]][r["label"]] += 1
    ids = sorted(song_labels)
    held = set(ids[-len(ids) // 10:])
    rows = [r for r in master if r["id"] in held]
    rep["group_split_protocol"] = {
        "held_out_songs": len(held),
        "held_out_rows": len(rows),
        "oracle_song_majority_acc_on_held_out": round(
            sum(1 for r in rows if song_labels[r["id"]].most_common(1)[0][0] == r["label"]) / len(rows), 4),
        "uniform_chance": round(1 / 12, 4),
        "majority_class_rate_on_held_out": round(
            max(collections.Counter(r["label"] for r in rows).values()) / len(rows), 4),
    }

    # ---- id structure ------------------------------------------------------
    lens = collections.Counter(len(r["id"]) for r in master)
    numeric = sum(1 for r in master if r["id"].isdigit())
    vals = sorted(int(r["id"]) for r in master if r["id"].isdigit())
    rep["id_lengths"] = dict(sorted(lens.items()))
    rep["id_all_numeric"] = numeric == len(master)
    rep["id_min"] = vals[0]
    rep["id_max"] = vals[-1]
    rep["id_quantiles"] = {q: vals[int(len(vals) * q)] for q in (0.1, 0.25, 0.5, 0.75, 0.9)}

    # ---- playlist size vs label -------------------------------------------
    size_by_label = collections.defaultdict(list)
    pl_size = collections.Counter(r["playlist"] for r in master)
    pl_label = {r["playlist"]: r["label"] for r in master}
    for pid, n in pl_size.items():
        size_by_label[pl_label[pid]].append(n)
    rep["mean_playlist_size_per_label"] = {
        k: round(sum(v) / len(v), 1) for k, v in size_by_label.items()}

    with open(os.path.join(OUT, "audit2.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
