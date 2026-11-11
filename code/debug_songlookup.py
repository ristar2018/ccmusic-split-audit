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
flat = {k: v.most_common(1)[0][0] for k, v in song2lab.items()}
print("n songs train:", len(flat))
print("sample values:", list(flat.items())[:3])
print("value types:", collections.Counter(type(v).__name__ for v in flat.values()))
same = sum(1 for r in test if flat.get(r["id"]) == r["label"])
hit = sum(1 for r in test if r["id"] in flat)
print("hit:", hit, "same:", same, "acc:", same / len(test))
