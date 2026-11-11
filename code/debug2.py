import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(ROOT, "data", "repo")


def load(p):
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


train = load(os.path.join(REPO, "data", "songs-train.jsonl"))
test = load(os.path.join(REPO, "data", "songs-test.jsonl"))
song2lab = collections.defaultdict(collections.Counter)
for r in train:
    song2lab[r["id"]][r["label"]] += 1
n = 0
for r in test:
    if r["id"] in song2lab:
        print("TEST", r, "-> TRAIN", dict(song2lab[r["id"]]))
        n += 1
        if n >= 5:
            break
print("label sample repr:", repr(test[0]["label"]), repr(train[0]["label"]))
