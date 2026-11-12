#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Three-arm training: A0 / A1 / A2 × backbone × seed, writing two log tiers (train_log readable any time, sealed_eval sealed).

Conventions (must not be changed in code):
- `design note v2.8 :265` frozen recipe: 224×224, ImageNet init, 10 epochs,
  AdamW lr 3e-4 / wd 0.05, cosine, warmup 1 epoch, label smoothing 0.1, batch 64, AMP, seeds 0/1/2.
- Same lines `:267-269`: each run writes `train_log` (split_hash, the sample_id lists of train/val/test,
  per-epoch val macro-F1, loss, gradient norm, lr) and `sealed_eval` (Q_test per-epoch F1, confusion matrix, per-sample predictions);
  **writing to disk seals it: unsealed only after all 3 seeds of that library and backbone have run**; model selection uses R only; Q_test may not
  be used for early stopping, and seeds may not be chosen by test results.
- `:273` main metric macro-F1, two views (slice-level + recording-level one vote); this script computes both.
- `worklog/exp-plan/KICKOFF.md` §3.5: E0 smoke = CTIS + A0 + AlexNet 10 epochs, two seeds within 1 point of macro-F1 (G-B gate).

How the seal is enforced (harder than a "convention"): while training, each seed writes **only** val metrics to `result.json`;
the Q_test evaluation and per-sample predictions go to `sealed_eval/pending/<group>/seed<N>.json`, and only when
all seeds listed in `--seeds` have finished is the pending directory renamed to `sealed_eval/<group>/` with `SEALED.json` written.
Mid-run you see val, never test: to look at test you must finish the whole group first.

Usage:
  # E0 smoke (G-B gate)
  python code/train_e1.py --arm A0 --bb alexnet --seeds 0,1 --epochs 10 --tag e0
  # single-seed trial run (at most 2000 training images, pipeline self-check)
  python code/train_e1.py --arm A0 --bb alexnet --seeds 0 --epochs 1 --limit 2000 --tag smoke
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PROJ, "data", "cache", "mel224")
SPLITS = os.path.join(PROJ, "data", "splits", "v2")
RUNS = os.path.join(PROJ, "runs")

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
EPOCHS = 10
LR = 3e-4
WD = 0.05
WARMUP_EPOCHS = 1
LS = 0.1
BATCH = 64
NUM_WORKERS = 0  # multiprocessing named pipes are refused in the sandbox (WinError 5), so the main process reads the memmap; fast enough in practice

ARMS = {
    # arm -> (train, val, test lists)
    "A0": (["a0_train"], ["r"], ["qtest"]),
    "A1": (["a1_train"], ["r"], ["qtest"]),
    "A2": (["a2_train"], ["a2_val"], ["qtest", "a2_selftest"]),
    # C1 (attribution control for external review M1): same size and class balance as A1, but the injected content comes from non-test recordings;
    # splits and leakage assertions in code/c1_inject.py and data/splits/v2/c1_audit.json.
    "C1": (["c1_train"], ["r"], ["qtest"]),
}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cache_index(lib):
    """sample_id -> (row, label, recording_uid); the row is the memmap row."""
    idx = {}
    for r in read_csv(os.path.join(CACHE, lib + "_index.csv")):
        idx[r["sample_id"]] = (int(r["row"]), r["label"], r["recording_uid"])
    return idx


class MelCache(Dataset):
    """Reads the memmap directly, returns normalised float32 CHW. Each __getitem__ copies, so no view is shared across workers."""

    def __init__(self, lib, split_files, label_of, index, limit=None, seed=0):
        self.mm = None
        self.lib = lib
        self.samples = []
        for f in split_files:
            for r in read_csv(os.path.join(SPLITS, f if f.startswith(lib) else "%s_%s.csv" % (lib, f))):
                row = index[r["sample_id"]][0]
                self.samples.append((row, label_of[r["label"]], r["recording_uid"], r["sample_id"]))
        self.samples.sort()  # row order, deterministic
        if limit:
            rng = random.Random(seed)
            self.samples = sorted(rng.sample(self.samples, min(limit, len(self.samples))))

    def __len__(self):
        return len(self.samples)

    def _open(self):
        if self.mm is None:
            n = os.path.getsize(os.path.join(CACHE, self.lib + ".u8")) // (SIZE * SIZE * 3)
            self.mm = np.memmap(os.path.join(CACHE, self.lib + ".u8"), dtype=np.uint8,
                                mode="r", shape=(n, SIZE, SIZE, 3))
        return self.mm

    def __getitem__(self, i):
        row, y, _, _ = self.samples[i]
        img = np.asarray(self._open()[row], dtype=np.float32)
        img = (img - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1)), y


def build_model(bb, n_classes):
    import torchvision
    if bb == "alexnet":
        m = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(4096, n_classes)
    elif bb == "swin_t":
        m = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, n_classes)
    else:
        raise SystemExit("unknown backbone: " + bb)
    return m


@torch.no_grad()
def evaluate(model, loader, n_classes, device, records=None):
    model.eval()
    ys, ps, recs, probs = [], [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        p = out.float().softmax(1)
        ps.append(p.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    prob = np.concatenate(probs)
    f1_slice = f1_score(y, p, average="macro", zero_division=0)
    out = {"macro_f1_slice": round(float(f1_slice), 4), "acc": round(float((y == p).mean()), 4),
           "n": int(len(y))}
    if records is not None:
        # recording-level one vote: average the probabilities within a recording, then argmax (fixed 2026-11-05: the first version took the first slice's probability)
        order = {}
        for i, rec in enumerate(records):
            order.setdefault(rec, []).append(i)
        ry, rp = [], []
        for rec, idxs in order.items():
            pr = prob[idxs].mean(0)
            yy = np.bincount(y[idxs], minlength=n_classes).argmax()
            ry.append(int(yy))
            rp.append(int(pr.argmax()))
        out["macro_f1_recording"] = round(float(f1_score(ry, rp, average="macro", zero_division=0)), 4)
        out["n_recordings"] = len(ry)
        out["confusion"] = confusion_matrix(y, p, labels=list(range(n_classes))).tolist()
        out["probs"] = prob.astype(np.float16)
        out["y"] = y
        out["records"] = records
    return out


def grad_norm(model):
    tot = 0.0
    for p in model.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return round(tot ** 0.5, 4)


def cosine_lr(step, total, warmup):
    if step < warmup:
        return LR * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return LR * 0.5 * (1 + np.cos(np.pi * t))


def run_one(lib, arm, bb, seed, epochs, limit, out_root, tag, workers=NUM_WORKERS):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index = load_cache_index(lib)
    labels = sorted({v[1] for v in index.values()}) if False else None
    # the label space must match the kept classes in class_universe.csv; all arms share one mapping
    uni = [r for r in read_csv(os.path.join(SPLITS, "class_universe.csv")) if r["library"] == lib]
    universe = sorted(r["class"] for r in uni if r["kept"] == "True")
    label_of = {c: i for i, c in enumerate(universe)}
    n_classes = len(universe)

    tr_files, va_files, te_files = ARMS[arm]
    tr = MelCache(lib, tr_files, label_of, index, limit=limit, seed=seed)
    va = MelCache(lib, va_files, label_of, index, limit=limit, seed=seed + 100)
    te = [MelCache(lib, [f], label_of, index) for f in te_files]
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("empty split: %s %s" % (lib, arm))

    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    tr_ld = DataLoader(tr, batch_size=BATCH, shuffle=True, drop_last=True, **kw)
    va_ld = DataLoader(va, batch_size=BATCH, shuffle=False, **kw)
    te_ld = [DataLoader(t, batch_size=BATCH, shuffle=False, **kw) for t in te]

    model = build_model(bb, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss(label_smoothing=LS)
    scaler = torch.amp.GradScaler("cuda")

    run_dir = os.path.join(out_root, lib, arm, bb, "seed%d" % seed)
    os.makedirs(run_dir, exist_ok=True)
    logp = os.path.join(run_dir, "train_log.csv")
    steps_per_epoch = len(tr_ld)
    total = steps_per_epoch * epochs
    warm = steps_per_epoch * WARMUP_EPOCHS
    step = 0
    t0 = time.time()
    best = {"macro_f1_slice": -1.0, "epoch": -1}

    with open(logp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "lr", "train_loss", "train_acc", "grad_norm",
                    "val_macro_f1_slice", "val_macro_f1_recording", "val_loss", "elapsed_s"])
        for ep in range(1, epochs + 1):
            model.train()
            tot_loss, correct, seen = 0.0, 0, 0
            for x, y in tr_ld:
                lr = cosine_lr(step, total, warm)
                for g in opt.param_groups:
                    g["lr"] = lr
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(x)
                    loss = lossf(out, y)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)          # unscale first, else grad_norm is inflated by ~65536x
                gn = grad_norm(model)
                scaler.step(opt)
                scaler.update()
                tot_loss += float(loss.detach()) * len(y)
                correct += int((out.argmax(1) == y).sum())
                seen += len(y)
                step += 1
            # validation (R only / the arm's own val split)
            va_ev = evaluate(model, va_ld, n_classes, device, records=[s[2] for s in va.samples])
            vl = float("nan")
            model.eval()
            with torch.no_grad():
                vl_acc = 0.0
                for x, y in va_ld:
                    x = x.to(device)
                    y = y.to(device)
                    with torch.autocast("cuda", dtype=torch.float16):
                        vl_acc += float(nn.functional.cross_entropy(model(x).float(), y, reduction="sum"))
                vl = vl_acc / max(1, len(va))
            w.writerow([ep, step, round(lr, 8), round(tot_loss / max(1, seen), 5),
                        round(correct / max(1, seen), 4), gn,
                        va_ev["macro_f1_slice"], va_ev.get("macro_f1_recording"),
                        round(vl, 5), round(time.time() - t0, 1)])
            f.flush()
            print("  seed%d ep%d/%d loss %.4f acc %.4f | val F1 %.4f (rec %.4f) lr %.2e  %.0fs"
                  % (seed, ep, epochs, tot_loss / max(1, seen), correct / max(1, seen),
                     va_ev["macro_f1_slice"], va_ev.get("macro_f1_recording") or -1, lr, time.time() - t0),
                  flush=True)
            if va_ev["macro_f1_slice"] > best["macro_f1_slice"]:
                best = {"macro_f1_slice": va_ev["macro_f1_slice"], "epoch": ep,
                        "macro_f1_recording": va_ev.get("macro_f1_recording")}

    cfg = {"lib": lib, "arm": arm, "bb": bb, "seed": seed, "epochs": epochs,
           "batch": BATCH, "lr": LR, "wd": WD, "warmup_epochs": WARMUP_EPOCHS,
           "label_smoothing": LS, "amp": True, "limit": limit, "tag": tag,
           "n_train": len(tr), "n_val": len(va), "n_classes": n_classes,
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "size": SIZE, "normalize": "imagenet", "augment": "none"}
    res = dict(cfg)
    res["val_best"] = best
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["train_log"] = os.path.relpath(logp, PROJ)
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))

    # ---- sealed_eval: write pending, unseal only when the whole group is done ----
    group = "%s_%s_%s" % (lib, arm, bb)
    pend = os.path.join(out_root, "sealed_eval", "pending", group)
    os.makedirs(pend, exist_ok=True)
    seal = {"config": cfg, "val_best": best}
    for ds, ld, mc in zip(te_files, te_ld, te):
        ev = evaluate(model, ld, n_classes, device, records=[s[2] for s in mc.samples])
        probs = ev.pop("probs")
        y = ev.pop("y")
        recs = ev.pop("records")
        np.savez_compressed(os.path.join(pend, "seed%d_%s.npz" % (seed, ds)),
                            probs=probs, y=y, records=np.array(recs))
        seal[ds] = ev
    with open(os.path.join(pend, "seed%d.json" % seed), "w", encoding="utf-8") as f:
        json.dump(seal, f, ensure_ascii=False, indent=2)
    return res, group, pend


def try_seal(out_root, group, want_seeds):
    pend = os.path.join(out_root, "sealed_eval", "pending", group)
    have = sorted(int(n[4:-5]) for n in os.listdir(pend) if n.startswith("seed") and n.endswith(".json"))
    if sorted(want_seeds) != have:
        print("[seal] %s still missing seeds: want=%s have=%s (not unsealing)" % (group, want_seeds, have))
        return None
    final = os.path.join(out_root, "sealed_eval", group)
    if os.path.isdir(final):
        raise SystemExit("sealed directory already exists: " + final)
    agg = {"group": group, "seeds": have, "sealed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    for s in have:
        with open(os.path.join(pend, "seed%d.json" % s), encoding="utf-8") as f:
            agg["runs"]["seed%d" % s] = json.load(f)
    # gate: spread of Q_test slice-level macro-F1 across seeds
    vals = [agg["runs"]["seed%d" % s]["qtest"]["macro_f1_slice"] for s in have]
    agg["seed_spread_qtest_slice"] = round(max(vals) - min(vals), 4) if len(vals) > 1 else None
    with open(os.path.join(pend, "SEALED.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    os.rename(pend, final)
    print("[seal] unsealed -> %s (seed spread %s)" % (
        final, "n/a (single seed)" if agg["seed_spread_qtest_slice"] is None
        else "%.4f" % agg["seed_spread_qtest_slice"]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="CTIS")
    ap.add_argument("--arm", default="A0", choices=sorted(ARMS))
    ap.add_argument("--bb", default="alexnet", choices=["alexnet", "swin_t"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="max images per split (for pipeline self-checks)")
    ap.add_argument("--out", default=RUNS)
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--resume", action="store_true",
                    help="skip seeds whose runs/<lib>/<arm>/<bb>/seedN/result.json already exists (use to fill in missing seeds)")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    os.makedirs(a.out, exist_ok=True)
    group = None
    for s in seeds:
        res_path = os.path.join(a.out, a.lib, a.arm, a.bb, "seed%d" % s, "result.json")
        if a.resume and os.path.exists(res_path):
            print("== %s %s %s seed%d exists, skipping" % (a.lib, a.arm, a.bb, s))
            continue
        print("== %s %s %s seed%d (%d epoch%s)" % (a.lib, a.arm, a.bb, s, a.epochs,
                                                   ", limit %d" % a.limit if a.limit else ""))
        res, group, _ = run_one(a.lib, a.arm, a.bb, s, a.epochs, a.limit, a.out, a.tag, a.workers)
        print("   val_best %s" % res["val_best"])
    if group is None:
        group = "%s_%s_%s" % (a.lib, a.arm, a.bb)
    agg = try_seal(a.out, group, seeds)
    if agg is not None:
        print("G-B seed spread (Q_test slice-level macro-F1) = %s" % agg["seed_spread_qtest_slice"])


if __name__ == "__main__":
    main()
