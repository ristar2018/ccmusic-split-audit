#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三臂训练：A0 / A1 / A2 × 骨干 × seed，落两档日志（train_log 随时可看，sealed_eval 封存）。

口径来源（不许在代码里改）：
- `report/聚焦版实验设计方案v2.8.md:265` 配方冻结：224×224、ImageNet 初始化、10 epoch、
  AdamW lr 3e-4 / wd 0.05、cosine、warmup 1 epoch、label smoothing 0.1、batch 64、AMP、seed 0/1/2。
- 同行 `:267-269`：每 run 落 `train_log`（split_hash、train/val/test 的 sample_id 列表、逐 epoch
  的 val macro-F1、loss、梯度范数、lr）与 `sealed_eval`（Q_test 逐 epoch F1、混淆矩阵、逐样本预测），
  **写盘即封存，等该库该骨干 3 个 seed 全跑完才解封**；选型只用 R；禁止用 Q_test 做 early stopping、
  禁止按测试结果挑 seed。
- `:273` 主指标 macro-F1，双口径（切片级 + 录音级一票）；本脚本两个都算。
- `worklog/exp-plan/KICKOFF.md` §3.5：E0 冒烟 = CTIS + A0 + AlexNet 10 epoch，两 seed 宏 F1 差 <1 点（G-B 门）。

封存机制的实现方式（比"约定"硬）：每个 seed 训练时**只**写 val 指标到 `result.json`；
Q_test 的评估结果与逐样本预测写到 `sealed_eval/pending/<group>/seed<N>.json`，当 `--seeds` 里
的所有 seed 都跑完，才把 pending 目录重命名成 `sealed_eval/<group>/` 并写 `SEALED.json`。
中途看得到的是 val，不是 test——想看 test 必须先把整组跑完。

用法：
  # E0 冒烟（G-B 门）
  python code/train_e1.py --arm A0 --bb alexnet --seeds 0,1 --epochs 10 --tag e0
  # 单 seed 试跑（限 2000 张训练，管线自检）
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
NUM_WORKERS = 0  # 沙箱内 multiprocessing 建命名管道被拒（WinError 5），只能主进程读 memmap；实测够快

ARMS = {
    # arm -> (train, val, test 列表)
    "A0": (["a0_train"], ["r"], ["qtest"]),
    "A1": (["a1_train"], ["r"], ["qtest"]),
    "A2": (["a2_train"], ["a2_val"], ["qtest", "a2_selftest"]),
    # C1（外审 M1 的归因对照）：体积与类别平衡同 A1，但注入内容来自非测试录音；
    # 切分与泄漏断言见 code/c1_inject.py 与 data/splits/v2/c1_audit.json。
    "C1": (["c1_train"], ["r"], ["qtest"]),
}


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_cache_index(lib):
    """sample_id -> (row, label, recording_uid)；行号即 memmap 行号。"""
    idx = {}
    for r in read_csv(os.path.join(CACHE, lib + "_index.csv")):
        idx[r["sample_id"]] = (int(r["row"]), r["label"], r["recording_uid"])
    return idx


class MelCache(Dataset):
    """memmap 直读，返回归一化 float32 CHW。每次 __getitem__ 拷一份，避免跨 worker 共享视图。"""

    def __init__(self, lib, split_files, label_of, index, limit=None, seed=0):
        self.mm = None
        self.lib = lib
        self.samples = []
        for f in split_files:
            for r in read_csv(os.path.join(SPLITS, f if f.startswith(lib) else "%s_%s.csv" % (lib, f))):
                row = index[r["sample_id"]][0]
                self.samples.append((row, label_of[r["label"]], r["recording_uid"], r["sample_id"]))
        self.samples.sort()  # 行号序，确定性
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
        # 录音级一票：同录音概率**平均**后取 argmax（2026-11-05 修：首版只取了该录音第一片的概率）
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
    # 标签空间必须与 class_universe.csv 的 kept 类一致，跨臂共用同一个映射
    uni = [r for r in read_csv(os.path.join(SPLITS, "class_universe.csv")) if r["library"] == lib]
    universe = sorted(r["class"] for r in uni if r["kept"] == "True")
    label_of = {c: i for i, c in enumerate(universe)}
    n_classes = len(universe)

    tr_files, va_files, te_files = ARMS[arm]
    tr = MelCache(lib, tr_files, label_of, index, limit=limit, seed=seed)
    va = MelCache(lib, va_files, label_of, index, limit=limit, seed=seed + 100)
    te = [MelCache(lib, [f], label_of, index) for f in te_files]
    if len(tr) == 0 or len(va) == 0:
        raise SystemExit("空切分：%s %s" % (lib, arm))

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
                scaler.unscale_(opt)          # 先还原梯度，否则 grad_norm 被放大约 65536 倍
                gn = grad_norm(model)
                scaler.step(opt)
                scaler.update()
                tot_loss += float(loss.detach()) * len(y)
                correct += int((out.argmax(1) == y).sum())
                seen += len(y)
                step += 1
            # 验证（只用 R / 自身验证）
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

    # ---- sealed_eval：写 pending，整组跑完才解封 ----
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
        print("[seal] %s 还差 seed：want=%s have=%s（先不解封）" % (group, want_seeds, have))
        return None
    final = os.path.join(out_root, "sealed_eval", group)
    if os.path.isdir(final):
        raise SystemExit("已封存目录存在：" + final)
    agg = {"group": group, "seeds": have, "sealed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    for s in have:
        with open(os.path.join(pend, "seed%d.json" % s), encoding="utf-8") as f:
            agg["runs"]["seed%d" % s] = json.load(f)
    # 门：两 seed 的 Q_test 切片级 macro-F1 极差
    vals = [agg["runs"]["seed%d" % s]["qtest"]["macro_f1_slice"] for s in have]
    agg["seed_spread_qtest_slice"] = round(max(vals) - min(vals), 4) if len(vals) > 1 else None
    with open(os.path.join(pend, "SEALED.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    os.rename(pend, final)
    print("[seal] 解封 → %s（seed 极差 %s）" % (
        final, "n/a（仅 1 个 seed）" if agg["seed_spread_qtest_slice"] is None
        else "%.4f" % agg["seed_spread_qtest_slice"]))
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="CTIS")
    ap.add_argument("--arm", default="A0", choices=sorted(ARMS))
    ap.add_argument("--bb", default="alexnet", choices=["alexnet", "swin_t"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="每切分最多用多少张（管线自检用）")
    ap.add_argument("--out", default=RUNS)
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--resume", action="store_true",
                    help="跳过已有 runs/<lib>/<arm>/<bb>/seedN/result.json 的 seed（补跑缺失 seed 用）")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    os.makedirs(a.out, exist_ok=True)
    group = None
    for s in seeds:
        res_path = os.path.join(a.out, a.lib, a.arm, a.bb, "seed%d" % s, "result.json")
        if a.resume and os.path.exists(res_path):
            print("== %s %s %s seed%d 已存在，跳过" % (a.lib, a.arm, a.bb, s))
            continue
        print("== %s %s %s seed%d (%d epoch%s)" % (a.lib, a.arm, a.bb, s, a.epochs,
                                                   ", limit %d" % a.limit if a.limit else ""))
        res, group, _ = run_one(a.lib, a.arm, a.bb, s, a.epochs, a.limit, a.out, a.tag, a.workers)
        print("   val_best %s" % res["val_best"])
    if group is None:
        group = "%s_%s_%s" % (a.lib, a.arm, a.bb)
    agg = try_seal(a.out, group, seeds)
    if agg is not None:
        print("G-B seed 极差（Q_test 切片级 macro-F1）= %s" % agg["seed_spread_qtest_slice"])


if __name__ == "__main__":
    main()
