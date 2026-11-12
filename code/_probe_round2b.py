# -*- coding: utf-8 -*-
"""Probe 2: whether the npz records keys align with the recording keys in the split CSV; sizes the exposed group."""
import csv
from pathlib import Path
import numpy as np

z = np.load("runs/sealed_eval/CTIS_A1_alexnet/seed0_qtest.npz", allow_pickle=True)
rec = z["records"].astype(str)
print("records dtype", z["records"].dtype, "uniq", len(np.unique(rec)))
print("first 5 records:", [repr(r) for r in rec[:5]])
print("contains 'C0090'?", any("C0090" in r for r in rec[:50]))

split = Path("data/splits/v2")


def read(name):
    with (split / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


test = read("CTIS_qtest.csv")
a1 = read("CTIS_a1_train.csv")
pool = read("CTIS_qpool.csv")
print("n test rows", len(test), "n test uids", len({r["recording_uid"] for r in test}),
      "n test canon", len({r["recording_canonical"] for r in test}))
print("a1 rows", len(a1), "a1 uids", len({r["recording_uid"] for r in a1}))
poolids = {r["sample_id"] for r in pool}
testids = {r["sample_id"] for r in test}
print("overlap sample_id pool&test:", len(poolids & testids))
inj = [r for r in a1 if r["sample_id"] in poolids]
print("injected rows", len(inj), "inj recs", len({r["recording_uid"] for r in inj}))
tru = {r["recording_uid"] for r in test}; trc = {r["recording_canonical"] for r in test}
iru = {r["recording_uid"] for r in inj}; irc = {r["recording_canonical"] for r in inj}
print("exposed by uid:", len(tru & iru), "/", len(tru), "| by canon:", len(trc & irc), "/", len(trc))
print("npz uniq rec == test uids:", set(rec) == tru, "| == test canon:", set(rec) == trc)
print("example npz rec:", repr(rec[0]), "| csv uid:", test[0]["recording_uid"], "| csv canon:", test[0]["recording_canonical"][:40])
