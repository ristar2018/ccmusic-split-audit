#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""code/train_p3.py — P3：按乐器的 3 折交叉 + 乐器对抗（GRL）。

预注册口径见 `worklog/exp-plan/P3-预注册草案.md`（v1 冻结稿）与 `data/p3/p3_folds.json`。

三臂（同一批 7,528 片 / 1,388 录音 / 43 类）：
  A0  随机**切片级** 3 折（同一录音的兄弟切片跨折 → 上界，等于 P1 的 A2 逻辑）
  A1  `p3_splits.csv` 的乐器分组折（同乐器同折 → 真实跨乐器泛化，主结论）
  A2  A1 + 表征层乐器对抗头（GRL，λ 0→1、预热 2 epoch，辅助损失按 1/sqrt(n_c) 重加权）

每折：训练 = 另两折，另从训练侧按录音留 10% 做选型；held-out 折只在最后评一次。
主指标：三折 held-out 预测**合并**后算 pooled macro-F1（切片级 + 录音级）。

产物：runs_p3/<arm>/<bb>/seed<N>/{fold<f>_log.csv, fold<f>.npz, result.json, pooled.json}
      fold<f>.npz 键：probs(float16) / y / records
用法：
  python code/train_p3.py --arms A1 --bbs alexnet --seeds 0 --folds 1 --epochs 2   # 冒烟
  python code/train_p3.py --arms A0,A1,A2 --bbs alexnet,swin_t --seeds 0,1
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from train_e1 import (BATCH, LS, LR, MEAN, PROJ, SIZE, STD, WD, WARMUP_EPOCHS,
                      cosine_lr, grad_norm)

P3 = os.path.join(PROJ, "data", "p3")
CACHE = os.path.join(PROJ, "data", "cache", "mel224")
RUNS = os.path.join(PROJ, "runs_p3")
LIB = "CTIS"
EPOCHS = 10
SEEDS = [0, 1]
N_FOLDS = 3
GRL_WARMUP_EPOCHS = 2
NUM_WORKERS = 0
LAM_MAX = 0.02     # 对抗强度上限：λ=1 推爆 loss（1e6），λ=0.1 直接塌缩（探针 top1=0、pooled F1 0.21）
AMP = True         # --fp32 关掉（GRL + fp16 组合在 43 类小样本上数值不稳，见 worklog 记录）
INST_W = None      # 乐器辅助损失类权重，main() 里按 1/sqrt(n_c) 计算
AUX_W = 0.2        # 辅助损失总权重（对抗项只做正则，不与主损失同权）


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_p3():
    folds = json.load(open(os.path.join(P3, "p3_folds.json"), encoding="utf-8"))
    rows = read_csv(os.path.join(P3, "p3_splits.csv"))
    classes = folds["classes"]
    label_of = {c: i for i, c in enumerate(classes)}
    insts = sorted({r["instrument_canon"] for r in rows})
    inst_of = {c: i for i, c in enumerate(insts)}
    idx = {r["sample_id"]: int(r["row"]) for r in read_csv(os.path.join(CACHE, LIB + "_index.csv"))}
    # B1（外审 M7）：录音分组折 + 乐器两侧都出现，用于 A1r 臂
    b1_path = os.path.join(P3, "p3b_splits.csv")
    b1 = {}
    if os.path.exists(b1_path):
        b1 = {r["sample_id"]: int(r["fold_b1"]) for r in read_csv(b1_path)}
    for r in rows:
        r["row"] = idx[r["sample_id"]]
        r["y"] = label_of[r["technique"]]
        r["inst"] = inst_of[r["instrument_canon"]]
        r["fold"] = int(r["fold"])
        if b1:
            if r["sample_id"] not in b1:
                raise SystemExit("p3b_splits.csv 缺少样本：" + r["sample_id"])
            r["fold_b1"] = b1[r["sample_id"]]
    return folds, rows, classes, insts


def split_a0(rows, seed=20261105):
    """切片级随机 3 折（按类分层），故意让同一录音的切片跨折 —— 这就是被批判的捷径。"""
    rng = random.Random(seed)
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["y"], []).append(r)
    for c in sorted(by_cls):
        sub = sorted(by_cls[c], key=lambda r: r["sample_id"])
        rng.shuffle(sub)
        for i, r in enumerate(sub):
            r["fold_a0"] = i % N_FOLDS + 1
    return rows


class SliceCache(Dataset):
    """只用给定行号的 memmap；与 train_e1.MelCache 同归一化。"""

    def __init__(self, rows):
        self.mm = None
        self.samples = sorted(((r["row"], r["y"], r["recording_uid"], r["sample_id"], r["inst"])
                               for r in rows))

    def __len__(self):
        return len(self.samples)

    def _open(self):
        if self.mm is None:
            n = os.path.getsize(os.path.join(CACHE, LIB + ".u8")) // (SIZE * SIZE * 3)
            self.mm = np.memmap(os.path.join(CACHE, LIB + ".u8"), dtype=np.uint8,
                                mode="r", shape=(n, SIZE, SIZE, 3))
        return self.mm

    def __getitem__(self, i):
        row, y, _, _, inst = self.samples[i]
        img = np.asarray(self._open()[row], dtype=np.float32)
        img = (img - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1)), y, inst


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def feats_logits(model, x, bb):
    """返回 (表征 h, 分类 logits)。"""
    # 分类头强制 fp32：fp16 下「反向梯度 × 类权重」会在 classifier 权重上溢出成 inf（实测 A2 首跑 97 个 inf）
    if bb == "alexnet":
        f = model.avgpool(model.features(x)).flatten(1)
        h = model.classifier[:6](f)
        with torch.autocast("cuda", enabled=False):
            return h, model.classifier[6](h.float())
    f = model.flatten(model.avgpool(model.permute(model.norm(model.features(x)))))
    with torch.autocast("cuda", enabled=False):
        return f, model.head(f.float())


def build_model(bb, n_classes, n_inst=None):
    import torchvision
    if bb == "alexnet":
        m = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        m.classifier[6] = nn.Linear(4096, n_classes)
    else:
        m = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, n_classes)
    head = None
    if n_inst:
        dim = 4096 if bb == "alexnet" else m.head.in_features
        head = nn.Linear(dim, n_inst)
    return m, head


@torch.no_grad()
def predict(model, loader, bb, device):
    model.eval()
    ys, ps, probs, recs = [], [], [], []
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            _, out = feats_logits(model, x, bb)
        p = out.float().softmax(1)
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


class ProbeCache(Dataset):
    """探针用：返回 (x, inst)。"""

    def __init__(self, rows, mm=None):
        self.mm = mm
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _open(self):
        if self.mm is None:
            n = os.path.getsize(os.path.join(CACHE, LIB + ".u8")) // (SIZE * SIZE * 3)
            self.mm = np.memmap(os.path.join(CACHE, LIB + ".u8"), dtype=np.uint8,
                                mode="r", shape=(n, SIZE, SIZE, 3))
        return self.mm

    def __getitem__(self, i):
        r = self.rows[i]
        img = np.asarray(self._open()[r["row"]], dtype=np.float32)
        img = (img - MEAN) / STD
        return torch.from_numpy(img.transpose(2, 0, 1)), int(r["inst"])


@torch.no_grad()
def extract_feats(model, loader, bb, device):
    model.eval()
    fs, ins = [], []
    for x, inst in loader:
        x = x.to(device, non_blocking=True)
        # 探针特征用 fp32 前向（autocast fp16 在对抗训练下会溢出成 inf，sklearn 直接报错）
        with torch.autocast("cuda", enabled=False):
            h, _ = feats_logits(model, x.float(), bb)
        fs.append(h.float().cpu().numpy())
        ins.append(np.asarray(inst))
    return np.concatenate(fs), np.concatenate(ins)


def instrument_probe(model, bb, device, tr_rows, te_rows, n_inst):
    """冻结表征上的乐器身份线性探针（AUC）。

    评估集必须与拟合集**共享乐器类**。留乐器折（A1/A2）的 held-out 乐器在训练侧完全没出现过，
    旧实现拿 held-out 折当评估集 → top1 恒为 0、AUC 无定义（实测 A2 fold1：probe_top1 0.0、
    auc_classes 0）。改为**训练侧按录音 8:2 切**：拟合 80%、评估 20%，两边乐器类相同，
    度量的是「该表征里乐器身份的可线性解码程度」，与 arms 间的差异可比。
    """
    rng = random.Random(0)
    recs = sorted({r["recording_uid"] for r in tr_rows})
    rng.shuffle(recs)
    n_ev = max(1, int(round(0.2 * len(recs))))
    ev_rec = set(recs[:n_ev])
    tr_sel = [r for r in tr_rows if r["recording_uid"] not in ev_rec]
    ev_sel = [r for r in tr_rows if r["recording_uid"] in ev_rec]
    if len(tr_sel) > 3000:
        tr_sel = [tr_sel[i] for i in sorted(rng.sample(range(len(tr_sel)), 3000))]
    if len(ev_sel) > 3000:
        ev_sel = [ev_sel[i] for i in sorted(rng.sample(range(len(ev_sel)), 3000))]
    ftr, itr = extract_feats(model, DataLoader(ProbeCache(tr_sel), batch_size=BATCH,
                                               shuffle=False, num_workers=0), bb, device)
    fte, ite = extract_feats(model, DataLoader(ProbeCache(ev_sel), batch_size=BATCH,
                                               shuffle=False, num_workers=0), bb, device)
    ftr = np.nan_to_num(ftr, nan=0.0, posinf=0.0, neginf=0.0)
    fte = np.nan_to_num(fte, nan=0.0, posinf=0.0, neginf=0.0)
    # 标准化后再拟合：冻结特征范数量级差别大，未标准化的 SGD 会在 predict_proba 里 exp 溢出成 nan
    sc0 = StandardScaler().fit(ftr)
    ftr, fte = sc0.transform(ftr), sc0.transform(fte)
    clf = LogisticRegression(C=1.0, max_iter=200, n_jobs=1)  # n_jobs>1 起 joblib 进程池，沙箱 PermissionError
    clf.fit(ftr, itr)
    pred = clf.predict(fte)
    top1 = float((pred == ite).mean())
    base_rate = float(np.bincount(ite).max() / len(ite))   # 多数类基线，top1 要跟它比而不是跟 0 比
    auc = None
    try:
        sc = clf.predict_proba(fte)
        sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
        cols = {int(c): j for j, c in enumerate(clf.classes_)}
        present = sorted(c for c in cols if 0 < int((ite == c).sum()) < len(ite))
        aucs = [roc_auc_score((ite == c).astype(int), sc[:, cols[c]]) for c in present]
        auc = float(np.mean(aucs)) if aucs else None
    except ValueError:
        auc = None
    return {"probe_top1": round(top1, 4), "probe_majority_baseline": round(base_rate, 4),
            "probe_auc_ovr_macro": (round(auc, 4) if auc is not None else None),
            "probe_auc_classes": len(present) if auc is not None else 0,
            "probe_n_train": len(ftr), "probe_n_test": len(fte),
            "probe_eval": "train_side_recording_split_80_20"}


def pooled_metrics(y, prob, records, n_classes):
    p = prob.argmax(1)
    out = {"macro_f1_slice": round(float(f1_score(y, p, average="macro", zero_division=0)), 4),
           "acc": round(float((y == p).mean()), 4), "n": int(len(y))}
    order = {}
    for i, rec in enumerate(records):
        order.setdefault(rec, []).append(i)
    ry, rp = [], []
    for rec, idxs in order.items():
        ry.append(int(np.bincount(y[idxs], minlength=n_classes).argmax()))
        rp.append(int(prob[idxs].mean(0).argmax()))
    out["macro_f1_recording"] = round(float(f1_score(ry, rp, average="macro", zero_division=0)), 4)
    out["n_recordings"] = len(ry)
    return out


def run_fold(rows, folds, classes, insts, arm, bb, seed, fold, epochs, out_root, tag):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes, n_inst = len(classes), len(insts)

    key = {"A0": "fold_a0", "A1r": "fold_b1"}.get(arm, "fold")
    te_rows = [r for r in rows if r[key] == fold]
    tr_pool = [r for r in rows if r[key] != fold]

    # 选型集：训练侧按录音留 10%（同录音不跨侧）
    rng = random.Random(20261105 + fold)
    recs = sorted({r["recording_uid"] for r in tr_pool})
    rng.shuffle(recs)
    val_recs = set(recs[:max(1, int(0.1 * len(recs)))])
    tr_rows = [r for r in tr_pool if r["recording_uid"] not in val_recs]
    va_rows = [r for r in tr_pool if r["recording_uid"] in val_recs]

    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    tr_ds = SliceCache(tr_rows)
    va_ds = SliceCache(va_rows)
    te_ds = SliceCache(te_rows)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, drop_last=True, **kw)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, **kw)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, **kw)

    model, ahead = build_model(bb, n_classes, n_inst if arm == "A2" else None)
    model = model.to(device)
    if ahead is not None:
        ahead = ahead.to(device)

    params = list(model.parameters()) + (list(ahead.parameters()) if ahead else [])
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss(label_smoothing=LS)
    auxf = nn.CrossEntropyLoss(weight=INST_W.to(device)) if arm == "A2" else None
    scaler = torch.amp.GradScaler("cuda")

    run_dir = os.path.join(out_root, arm, bb, "seed%d" % seed)
    os.makedirs(run_dir, exist_ok=True)
    logp = os.path.join(run_dir, "fold%d_log.csv" % fold)
    spe = len(tr_ld)
    total, warm = spe * epochs, spe * WARMUP_EPOCHS
    step, t0 = 0, time.time()
    best = {"macro_f1_slice": -1.0, "epoch": -1}

    with open(logp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "lr", "train_loss", "train_acc", "grad_norm", "grl_lambda",
                    "val_macro_f1_slice", "elapsed_s"])
        for ep in range(1, epochs + 1):
            # DANN 式平滑爬升：前 2 个 epoch 不施加对抗（预热），之后按步数进度线性升到 LAM_MAX。
            # 直接用 ep/2 会在第 1 个 epoch 就上 λ=0.5，Adam 的自适应步长让表征迅速漂移、loss 爆到 1e6；
            # λ=0.1 时虽不发散但表征塌缩（探针 top1=0、test F1 0.29 vs A1 0.51），故上限降到 0.02 并留预热。
            p = step / max(1, total)
            lam = 0.0 if ep <= GRL_WARMUP_EPOCHS else LAM_MAX * min(1.0, (p - GRL_WARMUP_EPOCHS / float(EPOCHS)) / 0.4)
            model.train()
            tot_loss, correct, seen = 0.0, 0, 0
            for batch in tr_ld:
                x, y = batch[0], batch[1]
                inst = batch[2] if arm == "A2" else None
                lr = cosine_lr(step, total, warm)
                for g in opt.param_groups:
                    g["lr"] = lr
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                    h, out = feats_logits(model, x, bb)
                    loss = lossf(out, y)
                if arm == "A2":
                    # 辅助分支恒走 fp32：fp16 下「反向梯度 × 类权重」在分类头上溢出成 inf（实测 97 个）
                    with torch.autocast("cuda", enabled=False):
                        li = ahead(GRL.apply(h.float(), lam))
                        loss = loss + AUX_W * auxf(li, inst.to(device, non_blocking=True))
                if AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                gn = grad_norm(model)
                if AMP:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                tot_loss += float(loss.detach()) * len(y)
                correct += int((out.argmax(1) == y).sum())
                seen += len(y)
                step += 1
            yv, pv = predict(model, va_ld, bb, device)
            f1v = float(f1_score(yv, pv.argmax(1), average="macro", zero_division=0))
            w.writerow([ep, step, round(lr, 8), round(tot_loss / max(1, seen), 5),
                        round(correct / max(1, seen), 4), gn, round(lam, 3), round(f1v, 4),
                        round(time.time() - t0, 1)])
            f.flush()
            print("  %s %s seed%d fold%d ep%d/%d loss %.4f acc %.4f | val F1 %.4f  %.0fs"
                  % (arm, bb, seed, fold, ep, epochs, tot_loss / max(1, seen),
                     correct / max(1, seen), f1v, time.time() - t0), flush=True)
            if f1v > best["macro_f1_slice"]:
                best = {"macro_f1_slice": round(f1v, 4), "epoch": ep}
    elapsed = round(time.time() - t0, 1)

    y, prob = predict(model, te_ld, bb, device)
    recs_te = [s[2] for s in te_ds.samples]
    np.savez_compressed(os.path.join(run_dir, "fold%d.npz" % fold),
                        probs=prob.astype(np.float16), y=y, records=np.array(recs_te))
    res = {"lib": LIB, "arm": arm, "bb": bb, "seed": seed, "fold": fold, "epochs": epochs,
           "n_train": len(tr_ds), "n_val": len(va_ds), "n_test": len(te_ds),
           "n_classes": n_classes, "n_inst": n_inst, "tag": tag,
           "val_best": best, "elapsed_s": elapsed,
           "test_macro_f1_slice": round(float(f1_score(y, prob.argmax(1), average="macro",
                                                       zero_division=0)), 4),
           "probe": None}
    json.dump(res, open(os.path.join(run_dir, "fold%d_result.json" % fold), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    torch.save(model.state_dict(), os.path.join(run_dir, "last_fold%d.pt" % fold))

    # 诊断：乐器身份线性探针（AUC）。训练侧抽 3,000 片拟合，held-out 折全量评估
    if arm in ("A0", "A2") and n_inst > 1:
        res["probe"] = instrument_probe(model, bb, device, tr_rows, te_rows, n_inst)
        json.dump(res, open(os.path.join(run_dir, "fold%d_result.json" % fold), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    return res, res["probe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A1")
    ap.add_argument("--bbs", default="alexnet")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--folds", default="1,2,3")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag", default="p3")
    ap.add_argument("--fp32", action="store_true", help="关掉 AMP（A2 的 GRL 在 fp16 下不稳）")
    ap.add_argument("--out", default=RUNS)
    a = ap.parse_args()
    global AMP
    AMP = not a.fp32
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    bbs = [s.strip() for s in a.bbs.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    folds_want = [int(s) for s in a.folds.split(",") if s.strip()]

    folds, rows, classes, insts = load_p3()
    rows = split_a0(rows)
    n_cls = len(classes)
    insts_cnt = np.bincount([r["inst"] for r in rows], minlength=len(insts)).astype(np.float64)
    global INST_W, AUX_W
    w = 1.0 / np.sqrt(np.maximum(insts_cnt, 1))
    INST_W = torch.tensor(w / w.mean(), dtype=torch.float32)
    AUX_W = 1.0
    print("P3：%d 片 / %d 录音 / %d 类 / %d 乐器（对抗头类数 = 宇宙内出现的归一生乐器数，与预注册 §9 修订 A1 一致）"
          % (len(rows), len({r["recording_uid"] for r in rows}), n_cls, len(insts)))

    for arm in arms:
        for bb in bbs:
            for seed in seeds:
                run_dir = os.path.join(a.out, arm, bb, "seed%d" % seed)
                if all(os.path.exists(os.path.join(run_dir, "fold%d.npz" % f)) for f in folds_want):
                    print("[skip] %s %s seed%d 已完成" % (arm, bb, seed))
                else:
                    for fold in folds_want:
                        print("[run] %s %s seed%d fold%d" % (arm, bb, seed, fold), flush=True)
                        run_fold(rows, folds, classes, insts, arm, bb, seed, fold,
                                 a.epochs, a.out, a.tag)
                # 合并三折 → pooled
                ys, ps, rs = [], [], []
                for fold in range(1, N_FOLDS + 1):
                    p = os.path.join(run_dir, "fold%d.npz" % fold)
                    if not os.path.exists(p):
                        continue
                    d = np.load(p, allow_pickle=True)
                    ys.append(d["y"])
                    ps.append(d["probs"].astype(np.float32))
                    rs.extend(d["records"].tolist())
                if len(ys) == N_FOLDS:
                    y = np.concatenate(ys)
                    prob = np.concatenate(ps)
                    pooled = pooled_metrics(y, prob, rs, n_cls)
                    pooled.update({"arm": arm, "bb": bb, "seed": seed,
                                   "folds_pooled": N_FOLDS, "tag": a.tag})
                    json.dump(pooled, open(os.path.join(run_dir, "pooled.json"), "w", encoding="utf-8"),
                              ensure_ascii=False, indent=2)
                    print("[pooled] %s %s seed%d  F1(slice) %.4f  F1(rec) %.4f  n=%d"
                          % (arm, bb, seed, pooled["macro_f1_slice"],
                             pooled["macro_f1_recording"], pooled["n"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
