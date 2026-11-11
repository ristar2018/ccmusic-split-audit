"""Pilot for the DMLR audit paper: three split protocols x four baselines.

Protocols
  P1 official  : the published row-level random 80/10/10 split
  P2 song      : 5-fold, grouped by song id (no song in train and test)
  P3 playlist  : 5-fold, grouped by playlist id (no playlist in train and test)

Baselines
  B0 majority class
  B1 playlist lookup   (playlist id -> majority label seen in train)
  B2 song lookup       (song id -> majority label seen in train)
  B3 playlist aggregation (sum the train label counts of all songs of the test
     playlist, predict the argmax for every row of that playlist)  <-- the honest
     "what can a weak-supervision model actually reach" reference

Everything is computed with plain Python + numpy-free logic, so it runs on CPU.
"""
import collections
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")
OUT = os.path.join(ROOT, "results")
NAMES = ["Nostalgic", "Fresh", "Romantic", "Melancholic", "Healing", "Relaxing",
         "Lonely", "Touching", "Exciting", "Joyful", "Peaceful", "Missing"]


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def stable_hash(s, mod):
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) % 1000003
    return h % mod


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d * 100, (c + m) / d * 100)


def evaluate(train, test):
    """Return accuracy for B0..B3 plus coverage diagnostics."""
    lab_ct = collections.Counter(r["label"] for r in train)
    majority = lab_ct.most_common(1)[0][0]

    pl_ct = collections.defaultdict(collections.Counter)
    song_ct = collections.defaultdict(collections.Counter)
    for r in train:
        pl_ct[r["playlist"]][r["label"]] += 1
        song_ct[r["id"]][r["label"]] += 1
    pl_map = {k: v.most_common(1)[0][0] for k, v in pl_ct.items()}

    # B3: aggregate the label evidence of a playlist's songs, then predict once
    # per playlist (rows of the same playlist share the prediction).
    pl_song_evidence = collections.defaultdict(collections.Counter)
    for r in test:
        ev = pl_song_evidence[r["playlist"]]
        for lab, c in song_ct.get(r["id"], {}).items():
            ev[lab] += c
    agg_map = {p: (ev.most_common(1)[0][0] if ev else majority)
               for p, ev in pl_song_evidence.items()}

    res = collections.Counter()
    n = len(test)
    for r in test:
        gold = r["label"]
        if gold == majority:
            res["B0"] += 1
        if pl_map.get(r["playlist"], majority) == gold:
            res["B1"] += 1
        if song_ct.get(r["id"]) and song_ct[r["id"]].most_common(1)[0][0] == gold:
            res["B2"] += 1
        if agg_map[r["playlist"]] == gold:
            res["B3"] += 1

    diag = {
        "n_rows": n,
        "playlist_coverage": sum(1 for r in test if r["playlist"] in pl_map) / n,
        "song_coverage": sum(1 for r in test if r["id"] in song_ct) / n,
        "playlists_with_any_seen_song": sum(
            1 for p in pl_song_evidence if pl_song_evidence[p]) / max(1, len(pl_song_evidence)),
    }
    return {k: 100.0 * v / n for k, v in res.items()}, diag


def summarize(name, acc, diag):
    out = {"protocol": name, "diagnostics": {k: round(v, 4) for k, v in diag.items()},
           "accuracy_pct": {k: round(v, 2) for k, v in acc.items()}}
    print(f"\n== {name} ==")
    for k, v in sorted(acc.items()):
        print(f"   {k}: {v:6.2f}%")
    print("   diagnostics:", out["diagnostics"])
    return out


def main():
    master = load(os.path.join(REPO, "data-processor", "songs.jsonl"))
    rep = {"n_rows": len(master)}

    # P1 official
    tr = load(os.path.join(REPO, "data", "songs-train.jsonl"))
    te = load(os.path.join(REPO, "data", "songs-test.jsonl"))
    acc, diag = evaluate(tr, te)
    rep["official"] = summarize("P1 official random split (train -> test)", acc, diag)

    # P2 / P3 five-fold grouped CV
    for tag, key in (("song", "id"), ("playlist", "playlist")):
        res = collections.Counter()
        diag_acc = collections.Counter()
        for f in range(5):
            tr = [r for r in master if stable_hash(r[key], 5) != f]
            te = [r for r in master if stable_hash(r[key], 5) == f]
            a, d = evaluate(tr, te)
            for k, v in a.items():
                res[k] += v * len(te) / 100.0
            for k, v in d.items():
                diag_acc[k] += v * len(te)
        n = sum(1 for r in master)
        acc = {k: 100.0 * v / n for k, v in res.items()}
        diag = {k: v / n for k, v in diag_acc.items()}
        rep[f"grouped_{tag}"] = summarize(f"P{'2' if tag == 'song' else '3'} 5-fold grouped by {tag}", acc, diag)

    # leakage premium = official - grouped(playlist)
    rep["leakage_premium_pp"] = {
        k: round(rep["official"]["accuracy_pct"][k] - rep["grouped_playlist"]["accuracy_pct"][k], 2)
        for k in rep["official"]["accuracy_pct"]
    }
    print("\nleakage premium (official - playlist-grouped):", rep["leakage_premium_pp"])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "pilot_protocols.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
