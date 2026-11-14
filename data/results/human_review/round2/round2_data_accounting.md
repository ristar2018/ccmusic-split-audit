# Round-2 数据核算：CTIS 三臂切片预算（A）与 §6.4 折构成（B）

核算日期：2026-11-11 · 复跑脚本：`code/round2_data_accounting.py`（57 条断言，全部 PASS）
本文件只做核算与披露，**不改 paper/**；所有数字由脚本从产物直接读取或重算，脚本退出码 0。

---

## 0. 结论先行

**A（CTIS 三臂预算）**：三臂**不是按类完全等量**。A0 = 25,502、C1 = 25,502、A1 = 25,498；逐类比对后**只有 C0296（唢呐）一类差 4 片**（A0/C1 各 241 片，A1 237 片）。原因是 Q_pool 中 C0296 的 4 张切片与 Q_test 中的音频 `content_hash` 重复，被全局同哈希去重（J4）剔除；而 `A1_slice_budget` 断言在去重**之前**结算，所以 `assertions.json` 仍然 `all_pass=true`。论文应把“equal”改为 **approximately equal**，并注明 C0296 与原因。

**B（§6.4 折构成）**：L-inst 三折按乐器互斥（135/135 只属一折），每折训练 4,477–4,528 片、测试 2,509–2,510 片；训练侧每折 43 类，**测试侧 fold1 只有 42 类**（缺“自然泛音”），fold2/3 为 43 类。43 类 = 49 个技法标签剔除 6 个内容类后，要求「(技法,乐器) 格内 ≥2 条录音的乐器 ≥3 个」。论文 pooled macro-F1 0.4483 / 0.5295 与 `data/results/p3/p3_metrics.csv` 的 A1 行一致。

---

## 1. 数据来源（文件 + 字段）

### A：CTIS 三臂训练清单

| 用途 | 文件 | 字段 |
| :-- | :-- | :-- |
| A0 训练清单 | `data/splits/v2/CTIS_a0_train.csv` | `label`（类别）、`sample_id`、`content_hash`、`recording_uid` |
| A1 训练清单 | `data/splits/v2/CTIS_a1_train.csv` | 同上 |
| C1 训练清单 | `data/splits/v2/CTIS_c1_train.csv` | 同上 |
| 预算账面 | `data/splits/v2/feasibility.csv` | `m_c`、`qpool_slices`、`n_a0`、`n_a1`、`a0_short`、`a1_short` |
| Q_pool 实际清单 | `data/splits/v2/CTIS_qpool.csv` | `label`、`content_hash` |
| 断言记录 | `data/splits/v2/assertions.json` | `libraries.CTIS.ok.A1_slice_budget`、`dropped_dup_slices` |
| C1 审计 | `data/splits/v2/c1_audit.json` | `assertions.K5_per_class_equal_to_a0`、`detail.n_train` |

说明：`runs/CTIS/<arm>/<bb>/seed*/` 下只有 `result.json`、`train_log.csv`、`last.pt`，**没有**训练清单，训练样本只能回查 `data/splits/v2/` 的冻结 CSV。三臂 CSV 的 `sample_id` 是各臂重新抽取的，**跨臂不可按 sample_id 对齐**，本核算按 `label` 计数、按 `content_hash` 比对。

### B：§6.4 折定义与使用

| 用途 | 文件 | 字段 |
| :-- | :-- | :-- |
| L-inst 折定义 | `data/p3/p3_splits.csv` | `fold`（1–3）、`technique`、`instrument_canon`、`recording_uid` |
| R-rec（B1）折定义 | `data/p3/p3b_splits.csv` | `fold_b1` |
| 折说明 | `data/p3/p3_folds.json`、`data/p3/p3b_folds.json` | `classes`、`per_fold`、`assertions`、`disclosed` |
| 实际训练/测试 | `runs_p3/<arm>/<bb>/seed0/fold<f>_result.json` | `n_train`、`n_val`、`n_test`、`n_classes`、`n_inst` |
| 实际测试集 | `runs_p3/<arm>/<bb>/seed0/fold<f>.npz` | `records`（录音）、`y`（标签） |
| 类别宇宙 | `data/p3/universe_before_drop.csv`、`universe_43.csv`、`P3_universe.json` | `technique`、`recordings`、`instruments` |
| 指标 | `data/results/p3/p3_metrics.csv`、`data/results/p3b_instr/p3_metrics.csv` | `macro_f1_slice` |

---

## 2. A：CTIS 三臂逐类切片预算

### 2.1 判定

| 臂 | 训练切片 | 类别数 | 与 A0 的差 |
| :-- | --: | --: | :-- |
| A0（干净基线） | 25,502 | 201 | — |
| A1（等量池替换） | 25,498 | 201 | **−4**，全部在 C0296 |
| C1（对照） | 25,502 | 201 | 0（201 类逐类相等） |

逐类全量对照见 `data/results/human_review/round2/slice_budget_by_class.csv`（`lib,arm,class,n_slices`，603 行 = 3 臂 × 201 类）。4/25,502 = **0.0157%**（约 0.016%，与审稿人估计一致）。

### 2.2 差异类别

| 类 | A0 | A1 | C1 | 差 |
| :-- | --: | --: | --: | --: |
| C0296（唢呐） | 241 | 237 | 241 | A1 −4 |
| 其余 200 类 | — | — | — | 0 |

### 2.3 原因链

1. `feasibility.csv` 的逐类预算：C0296 `m_c=241`、`qpool_slices=39`、`n_a1=241`、`a1_short=0`。CTIS 全部 201 类 `a1_short=0`，账面 `Σqpool_slices=4,182`、`Σn_a1=25,502`。
2. 实际 `CTIS_qpool.csv` 合计 **4,178** 片，逐类只差 C0296（39 → 35，少 4）。A1 = Q_pool + C 侧补齐，故 A1 的 C0296 只剩 237（35 + 202）。
3. 去重规则：`code/splits_v2.py:343-367` `trim_global_dups`（J4）对同一 `content_hash` 跨侧只保留优先级最高的一份，优先级 `qtest > qpool > r > c`；`a1_train` 的 allowed 侧是 `{c, qpool}`。本次共丢弃 **12** 行 = 4 个 `content_hash` × 3 个键（`qpool`、`a1_train`、`a2_train`），全部属于 C0296，录音为「C0296/【唢呐】演奏技法 垫音 第二遍」，保留侧均为 `qtest`。4 个哈希前缀：`6b3fd0cdeaefc3bc`、`a13ebc00004afe50`、`91a789f6dae7c286`、`ab51db544d703020`。
4. 断言时序问题：类循环在 `code/splits_v2.py:415` 收集 `cls_stats`；`trim_global_dups` 在 `:418` 才执行；而 `A1_slice_budget` 在 `:476-495`（具体 `:486`）用的是 **trim 前**的 `cls_stats`。因此 `assertions.json` 记录 `A1_slice_budget=true`、`all_pass=true`、`dropped_dup_slices=12`，与最终 CSV 的 25,498 自相矛盾——不是数据错，是**断言结算口径早于去重**。
5. A0 不含 Q_pool 侧，C1 又是按实际注入数逐类回抽（`code/c1_inject.py:121-126`、`:154`），故二者都保持 241；`c1_audit.json` 的 `K5_per_class_equal_to_a0=true` 与 `n_train=25,502` 亦证实 C1 逐类等于 A0。

### 2.4 断言（脚本输出，全部 PASS）

- A0/A1/C1 行数 = 25,502 / 25,498 / 25,502；三臂各 201 类。
- A0−A1 逐类差 = `{C0296: 4}`；A0−C1 逐类差 = `{}`。
- qpool 计划 4,182 vs 实际 4,178，差 `{C0296: 4}`；feasibility 的 `Σn_a1=25,502`、`a1_short` 全 0（去重前口径）。
- `assertions.json` `A1_slice_budget=true`、`dropped_dup_slices=12`；`c1_audit.json` `K5=true`、`n_train=25,502`。

---

## 3. B：§6.4 折构成

### 3.1 折定义

- **L-inst（论文 §6.4 主线，A1/A2）**：`data/p3/p3_splits.csv` 的 `fold`，按乐器分组、同乐器同折、贪心均衡切片数；135 个规范化乐器**全部只出现在一折**（断言实测 multi=0）。折规模 2,509 / 2,509 / 2,510 片。
- **R-slice（A0，乐观参照）**：`code/train_p3.py:85-96` 的 `split_a0`，类内按 `sample_id` 排序后 `random.Random(20261105)` 洗牌、`i % 3 + 1` 分折，**同一录音的切片可跨折**（脚本 docstring 明说这是被批判的捷径）。fold1 训练与测试录音交集实测 **819** 条。
- **R-rec（A1r）**：`data/p3/p3b_splits.csv` 的 `fold_b1`，录音分组、允许 110 个乐器同时出现在两折。用 npz 的 `records` 逐折核对，A1r 匹配 `fold_b1`、不匹配 `fold`。
- val 划分（三臂一致）：`random.Random(20261105+fold)` 洗牌 `sorted(recording_uid)`，取前 `max(1, int(0.1*n))` 条录音（`code/train_p3.py:282-299`）。

### 3.2 折构成（`p3_fold_composition.csv`）

| fold | arm | 训练录音 | 训练切片 | 测试录音 | 测试切片 | 训练类数 | 测试类数 |
| --: | :-- | --: | --: | --: | --: | --: | --: |
| 1 | A0 | 1,151 | 4,522 | 1,019 | 2,524 | 43 | 43 |
| 2 | A0 | 1,146 | 4,520 | 1,014 | 2,508 | 43 | 43 |
| 3 | A0 | 1,164 | 4,519 | 997 | 2,496 | 43 | 43 |
| 1 | A1 | 829 | 4,528 | 467 | 2,509 | 43 | **42** |
| 2 | A1 | 818 | 4,494 | 480 | 2,509 | 43 | 43 |
| 3 | A1 | 853 | 4,477 | 441 | 2,510 | 43 | 43 |
| 1 | A2 | 829 | 4,528 | 467 | 2,509 | 43 | **42** |
| 2 | A2 | 818 | 4,494 | 480 | 2,509 | 43 | 43 |
| 3 | A2 | 853 | 4,477 | 441 | 2,510 | 43 | 43 |
| 1 | A1r | 835 | 4,511 | 461 | 2,509 | 43 | 43 |
| 2 | A1r | 834 | 4,528 | 462 | 2,509 | 43 | 43 |
| 3 | A1r | 831 | 4,514 | 465 | 2,510 | 43 | 43 |

`fold` 的语义随臂不同（CSV 靠 `arm` 列区分）：A0 = `fold_a0`（切片级随机折），A1/A2 = `fold`（L-inst），A1r = `fold_b1`（R-rec）；定义见 §3.1。A2 与 A1 的行、折、val 完全相同，只差 GRL。

校验：12 行全部与 `fold<f>_result.json` 的 `n_train/n_val/n_test` 逐位一致；每折测试切片数等于 npz `y` 长度，测试录音集合等于 npz `records` 集合。

### 3.3 43 类纳入规则（`code/p3_universe.py`）

单位是 `recording_uid`（不是切片）。先算单元格 `(技法, 乐器)` 的录音数，要求：

- `MIN_CELL = 2`：格内 ≥2 条录音；
- `MIN_INSTR = 3`：该技法有 ≥2 条录音的乐器数 ≥3；
- 另剔除 6 个内容类标签（非演奏技法）：`音阶`、`空弦音`、`筒音`、`单音`、`乐曲片段`、`特有演奏技法`。

结果：**49 类 / 4,037 条录音 → 43 类 / 1,388 条录音**；乐器名 212 → 204（6 组 14 名合并），折分组用合并后 135 个 canon。被剔除的 6 个标签全部是上述内容类，不是门槛淘汰（明细见 `universe_before_drop.csv` 与 `P3_universe.json`）。

### 3.4 论文 0.4483 / 0.5295 核对

| 产物 | 字段 | alexnet | swin_t |
| :-- | :-- | --: | --: |
| `data/results/p3/p3_metrics.csv` | `arm=A1, macro_f1_slice` | 0.4483 | 0.5295 |
| 脚本由 npz 重算（两 seed 全精度 mean） | `_pooled_seed_f1` | 0.44835 → 0.4483 | 0.52945 → 0.5295 |
| `runs_p3/A1/<bb>/seed{0,1}/pooled.json` | `macro_f1_slice` | 0.4503 / 0.4464 | 0.5435 / 0.5154 |

论文数字与产物**一致**。注意：若把 `pooled.json` 已舍入到 4 位的 seed 值再平均，会得到 0.4484 / 0.5294；论文口径是 `code/analyze_p3.py:142-146,171` 的全精度 seed F1 求均值后 `round(...,4)`，故 0.4483 / 0.5295 是对的，不构成差异。

另：`data/results/p3b_instr/p3_metrics.csv` 的 A0/A1/A2 与 `p3/p3_metrics.csv` **逐位相同**（同一批 p3 训练产物，A1r 才是在 `fold_b1` 上新训的），引用时不要当成两套独立实验。

---

## 4. 缺口与建议

1. **论文措辞**：§5/§6 若出现“equal per-class budget”，应改为 approximately equal，并补一句“A1 的 C0296 少 4 片，因 Q_pool 中 4 张切片与 Q_test 音频重复被同哈希去重剔除”。当前稿件未改（paper/ 只读）。
2. **断言口径**：`data/splits/v2/assertions.json` 的 `A1_slice_budget` 在 trim 前结算，建议在 `trim_global_dups` 之后补一条 J5（post-trim per-class == m_c），否则 `all_pass=true` 会继续掩盖此类差。
3. **L-inst fold1 类别覆盖**：测试侧 42/43（“自然泛音”无测试录音），`p3_folds.json` 已用 `disclosed.A3_missing_class_fold_cells=1` 记录；论文正文只声明 R-rec 每折 43 类，未提 L-inst 的这处缺口，建议在表注补一句或引用 `P3_universe.json`。
4. **无法核**：论文 PDF/`.tex` 是否已按上述修改、以及 `paper/` 中其它位置是否复述了 25,502/25,498——本次未改稿，需作者侧统一。
5. 本次核算不覆盖 §6.4 的 Δ/CI 重算（另有 `a1_minus_c1.csv` 等产物由并行任务生成，本文件不引用其数值）。

---

## 5. 复跑

```bat
cd /d D:\VSCode\科研任务\音乐情感研究
D:\XiaBao\AIArtDetect\envs\fapn\python.exe code\round2_data_accounting.py
```

输出：`data/results/human_review/round2/slice_budget_by_class.csv`、`p3_fold_composition.csv`；末行 `ALL PASS`，任一断言失败退出码 1。脚本只读 `data/`、`runs_p3/`，不写 `paper/`、不改划分。
