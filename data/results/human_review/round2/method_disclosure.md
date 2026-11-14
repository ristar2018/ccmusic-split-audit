# 复现细节披露核验（round 2）

对象：审稿意见第 17/19/21 条（`report/审稿意见/human-2026-11-11/review.txt:439-567`）点名的四处方法表述。
方法：逐行读 `code/` 实现，取行号证据；数值由只读脚本 `code/round2_method_facts.py` 从 `data/manifest/summary.json`、`data/splits/v2/*.csv`、`data/results/e1_metrics.csv`、`data/results/paper_verify.json` 重算，结果落 `data/results/human_review/round2/method_facts.json`。
复跑：`python3 code/round2_method_facts.py --root . --json data/results/human_review/round2/method_facts.json`（纯只读，无 subprocess）。

三处需要改写的表述（先给结论）：

1. §4.2「verified against the audio where audio is shipped」——只有 CTIS 真的做了音频核对（4,956 个 WAV 全量 SHA-1）；其余四库是纯文件名规则，其中 pianos、CNPM 盘面有音频但与 eval 命名不同源、未做任何关联。现有抽查表（5 份 `*_spotcheck.csv`）两列人工核对项全空。
2. §4.3 的「group」不是录音：组是同一录音内部的**相邻两片块**，一条录音的切片会同时进入两臂（CTIS 963 条测试录音里 578 条也在注入池；bel canto 40/40 全部如此）。「singleton recordings assigned to test」实际指「只贡献 1 片的录音」。
3. §5.3/§5.4 的宏平均分母在 `analyze_e1.py` 里是**逐次重抽样动态重算**（空类剔除），CTIS 平均每次重抽样缺 8.09/201 个类；而 `sealed_eval_delta.py`（封存评测表）用**固定分母**（缺席类计 0 分）。论文只描述了前者。

---

## A. 录音身份与音频核对（§4.2）

### 论文原话

`paper/sections/04_materials.tex:43-50`：

> Every slice is mapped to a `recording_uid` by a rule specific to the library, derived from the release format and verified against the audio where audio is shipped: CTIS by the per-class file stem, prefixed by the class code when a stem repeats across classes; music genre by the pair of track index and duration that the file name encodes; pianos by brand, recording number and duration; CNPM by joining the file name to the dataset's own `label.csv`; and bel canto by class and recording number. Parsing succeeded for 110,415 of 110,415 slices.

### 代码事实

**1) 核对动作只有一种：对 CTIS 全部 WAV 算 SHA-1 并折叠同哈希录音。**

```python
# code/manifest.py:76-84
def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
```

```python
# code/manifest.py:239-243  CTIS：对全部 n_rec 条录音逐个算哈希，不是抽样
if do_audio_hash:
    print("  音频去重：对 %d 个 wav 取字节 sha1 ..." % n_rec, flush=True)
    for i, gidx in enumerate(sorted(recs), 1):
        cls, fn = recs[gidx]
        audio_hash[uid_of[gidx]] = sha1_file(os.path.join(adir, cls, fn))
```

```python
# code/manifest.py:252-254  同哈希折叠到「插入顺序首个 uid」
for h, uids in by_hash.items():
    for uid in uids:
        canon[uid] = uids[0]
```

录音号本身是**位置枚举**，不是语义解析：`code/manifest.py:189-197` 按 `sorted(classes)` + 类内 `sorted(filenames)` 累加得到全局号，因此「解析成功 110,415/110,415」在 CTIS 侧是恒真（每个文件都被分配到一个号），它只证明 mel 与 wav 两套枚举**计数一致**，不证明号与录音**内容**一致。真正的内容侧证据有两项：mel→wav 与 wav→mel 的集合差为 0（`recordings_without_mel=0`、`mel_gidx_without_audio=0`，字段见 `code/manifest.py:332-333`），以及 43,054 张切片的 JPEG SHA-1 去重（`code/manifest.py:277-282`）。

**2) 四个库的身份规则是纯文件名，`audio_hash` 直接写空串。**

- music genre：`code/manifest.py:403` `audio_hash=""`；`audio_note` 429 自述「盘面无音频，仅 eval 特征」；`dedup_rule` 441 自述「盘面无音频，无法做音频级去重」。
- pianos：`code/manifest.py:512` `audio_hash=""`；478-479 自述「音频在 `data/audio/<品牌>/<录音号>.wav`，与 eval 的录音号不同名（6009 vs 009），无可靠映射，故不填 audio_hash」。
- CNPM：`code/manifest.py:578` `audio_hash=""`；同库只做了 `label.csv` 连接（533-537 读 GBK 表，589-591 写 `csv_join`），实测 `csv_rows=287`、`mel_recs=287`、`mel_not_in_csv=0`。
- bel canto：`code/manifest.py:626` `audio_hash=""`。

**3) 五个库的分库核对面（数值来自 `data/manifest/summary.json` 与 `data/datasets_inventory.json`，脚本 A 段）：**

| 库 | 核对对象 | 记录数 / 切片数 | 盘面音频 | 代码里的核对方法 | 无法核对的范围 |
|---|---|---|---|---|---|
| CTIS | WAV × JPEG | 4,956 / 43,054 | 4,956 WAV（`manifest.py:646` 备注 ~13.3 GB） | 全量 SHA-1（`manifest.py:76-84,239-243`）；同哈希折组 2 组 / 4 条录音；mel↔wav 集合差 0；JPEG SHA-1 去重 | 录音号正确性无外部权威可比对；无人工复核（抽查表为空） |
| music genre | 无（仅特征） | 1,713 / 36,375 | 0 | 仅文件名三段解析 `<track>_<dur>_<pos>`（`357-411`），`bad_filename_count=0` | 无音频，身份不可验证 |
| pianos | 盘面有音频但不同源 | 668 / 18,352 | 668 WAV（约 0.9 h） | 仅文件名解析（`474-523`）；`audio_hash=""`（512） | 音频文件名与 eval 录音号无可靠映射，未关联 |
| CNPM | `label.csv` × 文件名 | 287 / 2,724 | 287（196 mp3 + 91 wav） | 文件名去 `_<片位>` 后与 `label.csv` 连接，287/287 命中（`533-591`） | 盘面音频未参与身份核对；标签依赖数据集自带 csv |
| bel canto | 无（仅特征） | 203 / 9,910 | 0 | 仅文件名解析 `<cls>/<rid>`（`595-637`） | 无音频，身份不可验证 |

合计 110,415 片（43,054 + 36,375 + 18,352 + 2,724 + 9,910），与论文数字一致。

**4) 五处会误导披露的实现细节。**

- `uid_parse_rate` 字段名与实现不符：`code/manifest.py:320` 实为「带乐器标签的切片占比」（CTIS 0.9949），**不是**文件名→录音 ID 的解析成功率，不能拿它回应审稿人。
- CTIS mel 名解析无异常保护：`code/manifest.py:210` `gidx, pos = fn.rsplit(".", 1)[0].split("_")` 一旦命名不合规即抛 `ValueError` 整库失败（`--limit` 冒烟只截断行数，不跳过坏名）；这是「解析 100%」的另一半原因——坏名不是被统计，而是会让构建直接崩。
- `canon` 的选取口径文档与实现不一致：实现取插入顺序首个 uid（`252-254`），元数据写「组内字典序最小 uid」（`349-350`）。本数据两对重复均同类且两种顺序同解，未造成差异，但披露前应改掉这句自述。
- `audio_hash_done` 不能当「是否做过音频核对」读：`code/manifest.py:346` 写的是 `bool(dup_audio)`，即「是否**发现**了重复音频」；CTIS 为 True 只因为找到 2 组重复，另四库为 False 是因为根本没算哈希。
- 缓存脚本的「核对」是另一回事：`code/ccmusic_cache.py:127-146` 对每库随机 200 张 memmap 回读、与磁盘原图逐字节比（`max_abs_diff`/`n_nonzero`），只管缓存完整性，不能当身份核对引用。
- 人工核对没有发生过：`code/manifest.py:142-179` 生成的 5 份 `*_spotcheck.csv`，`check_image_matches_recording`、`check_no_other_recording_same` 两列**全部为空**（表头之外无任何非空单元格）。

### 可披露的准确表述

> Recording identity was verified against shipped audio only for CTIS, where all 4,956 WAV files were hashed (SHA-1; two duplicate pairs folded to 4 canonical recordings) and the mel/WAV index sets matched exactly; for the other four libraries the mapping is a pure file-name rule, and although pianos and CNPM do ship audio, its file names are not in the same namespace as the eval release, so it was not linked.

### 无法披露的部分

- 无法声称「逐条核对身份」或「人工复核」：CTIS 的核对是对音频文件本身去重，不是对「第 k 条录音 = 某段音频」的语义核验；另四库无任何内容级证据。
- 无法给出每库「未匹配/歧义计数」：CTIS 侧不存在歧义计数（枚举是全函数），四库侧只统计了坏文件名（均为 0），没有「解析到错误录音」的检测机制。
- pianos / CNPM 的音频能否与 eval 对齐，代码里查不到：`manifest.py:478-479` 只说无可靠映射，没有留下尝试记录。

---

## B. 分组键与两臂分配（§4.3）

### 论文原话

`paper/sections/04_materials.tex:72-75`：

> Groups are sorted by their candidate-group key and assigned alternately to sealed test and injection pool, with singleton recordings assigned to test.

同节 55-60 还称：CTIS 的 77 组相同图像覆盖 175 片；「one music-genre group spans three classes」；重复组一律压到单侧。

### 代码事实

**1) 组的构造（不是按录音建组，是按「相邻两片」建块）。**

```python
# code/splits_v2.py:258-272
dsu = DSU()
for rec in Q:
    rs = by_rec[rec]
    for i, r in enumerate(rs):
        dsu.find(r["sample_id"])
        if i % 2 == 1:
            dsu.union(rs[i - 1]["sample_id"], r["sample_id"])   # 相邻两片成块
by_hash = defaultdict(list)
for r in cls_rows:                     # 整类，不只 Q
    if r["content_hash"]:
        by_hash[r["content_hash"]].append(r["sample_id"])
for ids in by_hash.values():
    for x in ids[1:]:
        dsu.union(ids[0], x)
```

`by_rec` 内的切片按 `slice_index` 排序（`code/splits_v2.py:211-212`；`slice_index` 在 `86` 行已转 int，排序是数值序），所以块是 (第 0,1 片)、(第 2,3 片)……，**同一录音的不同块可以落到不同臂**。哈希合并只在类内做（`by_hash` 由 `cls_rows` 建，`266-272`），跨类不并组。

**2) 组键是「组内最小的切片候选键」，候选键是切片 JPEG 的 SHA-1 前 12 位。**

```python
# code/manifest.py:277-282
h = sha1_file(abs_p) if do_hash else ""
...
"candidate_group": ("g" + h[:12]) if h else "",
```

```python
# code/splits_v2.py:278-280
gkey = {g: min((r["candidate_group"] or r["sample_id"]) for r in rs) for g, rs in groups.items()}
order = sorted(groups, key=lambda g: (gkey[g], g))
```

即：key 不是录音 ID，也不是文件名前缀，而是**该组内字典序最小的 12 位图像哈希前缀**；缺键时回落 `sample_id`；同键再按 DSU 根 id 排序。

**3) 交替分配与 singleton 分支。**

```python
# code/splits_v2.py:283-288
for i, g in enumerate(order):
    rs = groups[g]
    if i % 2 == 0 or all(r["n_slices"] <= 1 for r in rs):
        qtest_rows.extend(rs)
    else:
        qpool_rows.extend(rs)
```

`n_slices` 是该录音在**类内**的切片总数（`code/splits_v2.py:92-94`），因此 `all(n_slices <= 1)` 等价于「这条录音只贡献 1 片」。序号为偶数的组进测试，奇数进注入池，singleton 组无论奇偶一律进测试。

**4) 重复图像的单侧约束有两处：**

```python
# code/splits_v2.py:290-301  测试侧已出现的哈希，把该录音整条从 R/C 拉进测试（shadow 记账）
# code/splits_v2.py:343-367  trim_global_dups：pri = qtest>qpool>r>c，allowed 侧映射，越侧即丢弃并计数
```

**5) 断言（`code/splits_v2.py:425-495`，任一失败 `exit 2`，见 `567`）：** `A0_disjoint`(477)、`A1_test_val_disjoint`(478)、`A1_train_has_q`(479)、`A1_alpha_material`(480)、`A2_sample_disjoint`(488)、`A1_n_c_side_match`(491-492)、`per_class_test_ge1`(493)、`hash_same_side`(494)；结果写 `assertions.json`（`560-561`）。

**6) 实测（脚本 B 段，从 `data/splits/v2/*.csv` 重算并重建组）：**

| 库 | 测试片/池片 | 测试录音/池录音 | 两臂共有录音 | 组数（单片组/两片组） | 跨录音组 | 跨侧组 | 相邻对被拆 | 单记录录音（全在测试） |
|---|---|---|---|---|---|---|---|---|
| CTIS | 4,605 / 4,178 | 963 / 753 | **578** | 4,682 (581 / 4,101) | 1 | 0 | 0 | 214 |
| music genre | 3,648 / 3,613 | 337 / 340 | 335 | 3,716 (171 / 3,545) | 0 | 0 | 0 | 0 |
| pianos | 1,884 / 1,862 | 135 / 134 | 133 | 1,908 (70 / 1,838) | 0 | 0 | 0 | 0 |
| CNPM | 271 / 268 | 54 / 50 | 47 | 285 (31 / 254) | 0 | 0 | 0 | 0 |
| bel canto | 979 / 977 | 40 / 40 | **40** | 989 (22 / 967) | 0 | 0 | 0 | 0 |

「测试 ∩ (R ∪ C) 录音 = 0」五库全部成立；「同哈希跨侧 = 0」五库全部成立。也就是说：**录音级**上测试与训练/校准不交，但**测试与注入池之间录音是重叠的**——这正是注入池的作用（同录音的新切片），论文原句没有讲清这一点。

另：CTIS 重建出 1 个跨录音组（同哈希把两条录音并进同组，`266-272` 的产物），与论文「two identical pairs」相符；五库均无跨类组，**代码里没有跨类并组的分支**——论文 §4.3 那句「one music-genre group spans three classes」在 `splits_v2.py` 中找不到对应实现（跨类同哈希只能靠 `trim_global_dups` 压到单侧，不是并组）。

### 可披露的准确表述

> Each library is partitioned into blocks of two consecutive slices of the same recording, blocks are merged when they share an identical JPEG hash, and the resulting groups are sorted by the smallest 12-bit hash prefix of their slice keys and dealt alternately to sealed test and injection pool; groups whose single slice comes from a one-slice recording always go to test, so a recording's slices can appear in both arms while no recording is shared between the sealed test set and the training or calibration arms.

### 无法披露的部分

- 不能沿用「Groups」指录音：代码里组是 1–2 片的块，录音跨两臂是常态（CTIS 578/963）。
- 不能披露「跨类同哈希并组」：代码无此分支；若论文明指审计脚本的分组，需另行给证据（本次在 `splits_v2.py` 中查不到）。
- 无法给出排序的具体键值序列：`candidate_group` 只在中间 CSV 里，`order` 未落盘；能复现的是键的构造与排序规则。
- 论文未交代的常量无法从代码「读出含义」：`q_c=round(0.20n)`、`r_c=round(0.15n)`（`splits_v2.py:217-219`，`Q_SHARE=0.20`/`R_SHARE=0.15` 见 `65-66`）、`MIN_REC=5` 的小类过滤（`67`、`406`）是容量设计，与本节表述无关但应在补充材料列出。

---

## C. macro-F1 与录音簇自助（§5.3 / §5.4）

### 论文原话

`paper/sections/05_protocol.tex:70-71`：

> Macro-F1 averages only over classes with non-zero support or predictions; classes absent from a panel are excluded rather than scored as zero.

### 代码事实

**1) 类别集合来自参照臂标签的最大值，分母逐次重算。**

```python
# code/analyze_e1.py:96-106
def macro_f1(conf):
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    keep = conf.sum(1) + conf.sum(0) > 0        # 真值或预测出现过
    f1 = f1[keep]
    return float(np.nanmean(f1)), int(keep.sum())
```

```python
# code/analyze_e1.py:162
nc = int(y_ref.max()) + 1    # 标签已重映射为 0..C-1，C = 该库测试集的类数
```

`C` 因此等于该库 Q_test 的类数；`e1_metrics.csv` 的 `n_classes` 与之逐库一致（201/16/8/5/4，只比计数，未核类标签到 `class_idx` 的对应）。类在测试集里全部出现过（`splits_v2.py:493` 断言 `per_class_test_ge1`），所以点估计时分母就是 C。

**2) 自助重抽样时，分母是动态的。**

```python
# code/analyze_e1.py:206-218
pick = rng.integers(0, nrec, size=(a.boot, nrec))
mult = np.zeros((a.boot, nrec), dtype=np.float64)
for b in range(a.boot):
    mult[b] = np.bincount(pick[b], minlength=nrec)
w_slice = mult[:, rec_idx]                     # (boot, n)
...
    f1s = np.array([macro_f1(conf_s[b])[0] for b in range(a.boot)])
```

某次重抽样缺了某个类，该类的 `conf` 行列全 0 → `keep=False` → 被剔除，**分母变小**；不会记 0 分。这就是论文 70-71 行描述的行为。

**3) 另一套脚本口径不同（封存评测表用固定分母）。**

```python
# code/sealed_eval_delta.py:143
mask = np.bincount(y, minlength=C) > 0        # 点样本一次算定
# code/sealed_eval_delta.py:45
return f1[:, mask].mean(1)                    # 分母固定为点样本出现过的类
```

而 `sealed_eval_delta.py:21-26` 的 `macro_f1` 对**全部 C** 取均值（缺席类按 F1=0 计入），点估计（`164`）走的是这一支。两份产物因此不严格同口径。

**4) 缺类的规模（脚本 C 段，解析式期望，非重跑）：**

| 库 | 类数 | 测试录音 | 每类录音数 | <5 条录音的类 | E[每次缺类] | P(至少缺 1 类) |
|---|---|---|---|---|---|---|
| CTIS | 201 | 963 | 1–21 | 89 | **8.09** | **0.9999** |
| music genre | 16 | 337 | 18–41 | 0 | 0.000 | 0.0000 |
| pianos | 8 | 135 | 10–18 | 0 | 0.000 | 0.0000 |
| CNPM | 5 | 54 | 8–15 | 0 | 0.000 | 0.0002 |
| bel canto | 4 | 40 | 6–19 | 0 | 0.002 | 0.0021 |

即：四个小库几乎不会缺类（动态与固定分母等价），**CTIS 几乎每次重抽样都缺约 8 个类**，其区间是在约 193 个类上算的，而按固定分母会记 0 分。CTIS 是唯一两类口径会产生可见差异的库。

一个只用算术就能给出的量级：若缺席的约 8 个类在固定口径下记 0 分，则动态口径相对固定口径的抬升上限约为 `m/(C−m) ≈ 8/193 ≈ 4.1%`（相对于均值，不是百分点）；缺席类的真实 F1 未知，实际抬升只会更小。

### 可披露的准确表述

> Macro-F1 is averaged over the classes present in the evaluated sample, so the denominator is re-derived inside every bootstrap replicate (a class absent from that resample is omitted rather than scored zero); for the CTIS panel this affects essentially every replicate, since about 8 of its 201 classes are missing from a typical recording-level resample.

### 无法披露的部分

- 无法披露「每次重抽样的分母」：脚本只回报 `float`（`analyze_e1.py:106` 的 `keep.sum()` 被丢弃），未落盘。
- 无法披露两套口径在 CTIS 上的数值差：需读 `runs/sealed_eval/*.npz` 并重跑自助（WSL 侧解释器无 numpy，Windows 侧 python 本轮不可用），本次未做。
- `sealed_eval_delta.py` 的固定分母与论文 70-71 行冲突，属未说明的口径混用；若要统一，得选一侧并重算该表。

---

## D. 独立验证脚本的覆盖边界（§5.7）

### 论文原话

`paper/sections/05_protocol.tex:155`：

> All reported numbers were re-derived from stored artifacts by an independent verification script.

`paper/sections/09_statements.tex:36-44`（第 24 页）已自我限定：

> The numerical verifier checks artifact identities, recalculates seed means and differences, and compares generated table rows with the manuscript. Bootstrap intervals are produced by separate scripts from sealed predictions; the verifier does not independently rerun every bootstrap or validate every prose claim.

### 代码事实

**1) 两层、189 项断言（`data/results/paper_verify.json:2-3`：`passes=189, fails=0`）。**

- identity 层：重算派生量（均值、差值、占比、求和、区间包含关系）并断言与产物一致。
- text 层：按表格行的排版重建字符串，在 LaTeX 归一化文本里查存在（`code/verify_paper_numbers.py:5-16` 的 docstring 描述；`419-441` 的散点令牌断言含 `"110,415"`）。

**2) 它读哪些产物（按行号）：**

| 产物 | 位置 |
|---|---|
| `data/results/e1_metrics.csv` | `95-106` |
| `data/results/e1_delta.csv`、`data/results/delta_A2/e1_delta.csv` | `189-190` |
| `data/splits/v2/<lib>_{qtest,qpool,r,c}.csv` | `113-137` |
| `data/results/feasibility.csv` | `140-145` |
| `data/results/p3/p3_metrics.csv`、`p3_delta.csv` | `251-262` |
| `data/results/cost.json` | `265-271` |
| `data/results/b2/e1_delta.csv` | `347-364` |
| `data/splits/negcontrol/negcontrol.json` | `350-352` |
| `data/results/p3b_leak*`、`p3b_instr*` | `375-381` |
| `data/results/c1/e1_delta.csv` | `464-476` |
| `data/results/human_review/{corrected_recording_delta,selftest,ctis_seed_stratified_bootstrap}.csv` | `303-304, 479-498` |
| `code/analyze_e1.py` 源码文本 | `500-502`（断言源码不含 `nc, w=rec_n`） |

**3) 它断言什么（抽样）：** 总量 109,831 片 / 7,756 录音 / 234 类（`200-204`）；Δ = 两侧均值差（`217-221`）；CI 包住点估计（`227-231`）；SESOI 判定与 `meets_sesoi` 列一致（`232-236`）；headroom 落在 [0,1]（`243-248`）；P3、cost、A2 占比、表 1–6 的逐行文本、B1 分解、C1、S1 补充表（`258-502` 各区段）。

**4) 「independent」相对谁：** 相对**稿件的数字文本**，不相对分析管线。它从不打开 `runs/sealed_eval/*.npz`，CI 是直接读 `e1_delta.csv` 的 `ci_lo/ci_hi`（`229`），bootstrap 一次也没重跑——这与 09_statements 的自我限定一致。

**5) 能查出 / 查不出：**

- 能查出：稿件表格里被改动或抄错的数字（text 层逐行比对）；`e1_delta.csv` 里 Δ 与 `e1_metrics.csv` 的 seed 均值不自洽；CI 不包住点估计；某库类数/录音数与划分 CSV 不符；SESOI 标记与 Δ/CI 矛盾。
- 查不出（举例）：
  1. 重抽样粒度写错——若脚本把「切片」当簇，`e1_delta.csv` 与验证器都会一致地错，验证器照样 189/0；
  2. 宏平均分母口径错（如把缺席类记 0）——验证器只比对 CSV 之间的算术关系，不重算混淆矩阵；这正是 C 节发现的 `analyze_e1` vs `sealed_eval_delta` 分歧所在，验证器无法发现；
  3. 数字串位置错——text 层只要求重建出的令牌在**整份**归一化 LaTeX 文本里作为子串出现（`163-170` 的 `needle in self.text`），把某表的数搬到另一张表仍会通过；
  4. 散文式断言（「independent verification」「parsing succeeded for 110,415/110,415」这类无量词描述）不在覆盖内，除 `419-441` 里被硬编码成令牌的少数数字外。
- 另：`paper_verify.json` 只落 `passes` 计数与 `fails` 明细（`505-512`），通过项的 check 名不落盘，因此「189 项覆盖了哪些断言」在产物里不可审计，只能读源码。

### 可披露的准确表述

> A verification script re-derives 189 numeric checks from stored aggregate artifacts (result tables, split CSVs and cost files), including seed means, paired differences, bootstrap-interval containment and the text of every reported table row; it does not re-read the sealed per-sample predictions or re-run the bootstrap, so it is independent of the manuscript text but not of the analysis pipeline that produced those artifacts.

### 无法披露的部分

- 无法把「independent」升级为「独立重算全流程」：npz→CSV 那一段没有被任何第二实现复算。
- 无法披露它检查过的断言清单全文：`paper_verify.json` 只落了 `passes/fails` 与部分面板合计；逐条 check 名未落盘（如需，应让脚本把每个 check 名写入 JSON）。
- 无法证明「189 项覆盖全部正文数字」：断言是人工挑选的令牌集，不是全文数字抽取。

---

## 附：本次核验的边界

- Windows 侧 `pwsh` 后端在本轮中途不可用（`sandbox mode "workspace-write" is requested but no sandbox backend is usable on this host ... Runner failure: windows-acl-run: --temp is not an existing directory: C:\Users\hmj\AppData\Local\Temp\dsh-TMIimL`），因此只读核算改在 WSL 下用 `python3` 跑；`D:\XiaBao\AIArtDetect\envs\fapn\python.exe` 未使用（其 subprocess 另被拒，且本次无需 torch）。
- 全流程只读：仅新建 `code/round2_method_facts.py`、`data/results/human_review/round2/method_disclosure.md`、`data/results/human_review/round2/method_facts.json`。
- 未核对项（明确缺口）：`runs/sealed_eval/*.npz` 未打开（WSL 无 numpy、Windows 侧 python 不可用），故 C 节只有解析式缺类期望，没有两类分母口径的实测差值；`data/splits/_audit_summary.csv` 的跨类同哈希组口径未追（与 B 节「one group spans three classes」相关）；CTIS 的 2 组重复音频明细在 `data/manifest/summary.json` 的 `duplicate_audio_groups` 里（脚本按 `pairs_summary.json`/`ctis_pairs.json` 两个候选名查找未命中，故未在报告里展开对名）。
