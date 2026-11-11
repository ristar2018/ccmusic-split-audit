# -*- coding: utf-8 -*-
"""一次性探针：只看结构（只打印标量/键名，避免打印混淆矩阵）。"""
import json
from pathlib import Path
import numpy as np

root = Path("runs/sealed_eval")


def scalars(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, dict)):
            out[k] = "<%s len=%d>" % (type(v).__name__, len(v))
    return out


for arm in ("A0", "A1", "A2", "C1"):
    for bb in ("alexnet", "swin_t"):
        d = root / ("CTIS_%s_%s" % (arm, bb))
        if not d.is_dir():
            print("MISSING", d)
            continue
        sg = json.loads((d / "SEALED.json").read_text(encoding="utf-8"))
        print("==", d.name, "seeds=", sg.get("seeds"), "| files:", sorted(p.name for p in d.glob("seed*_qtest.npz")))
        sj = json.loads((d / "seed0.json").read_text(encoding="utf-8"))
        print("   seed0.json keys:", [(k, type(sj[k]).__name__) for k in sj])
        if "qtest" in sj:
            print("   qtest scalars:", scalars(sj["qtest"]))
        if "config" in sj:
            print("   config:", scalars(sj["config"]))
        z = np.load(d / ("seed0_qtest.npz"), allow_pickle=True)
        print("   npz: probs=%s y=%s rec=%s nrec=%d" % (
            z["probs"].shape, z["y"].shape, z["records"].shape, len(np.unique(z["records"]))))
        print("   y label range:", int(z["y"].min()), int(z["y"].max()),
              "| probs sum1:", float(np.abs(z["probs"].sum(1) - 1).max()))
