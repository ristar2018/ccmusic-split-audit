# P3 pre-registration draft v0.2 (CTIS technique × instrument: cross-instrument generalisation and instrument-shortcut diagnosis)

Status: **draft v0.2, 2026-11-05, not frozen**. Changes relative to `worklog/exp-plan/P3-预注册草案.md` (v0.1): added instrument-name normalisation with a persisted merge table, re-checked the class universe against the sample manifest, reconciled §7.2 cell by cell (three items not reproducible), and rewrote the exclusion rule as pre-registrable sentences. **Note**: this session had `worklog/` read-only (writes refused by the sandbox), so the draft sits in `data/p3/` for now; before freezing, someone must fold it into `worklog/exp-plan/`.

Recomputation entry points (read-only, all rerunnable):
- `code/p3_universe_norm.py` → `data/p3/P3_universe_norm.json`, `data/p3/universe_norm_43.csv`
- `code/ctis_inst_norm.py` → `data/splits/ctis_inst_norm.json` (every merge traced in the table)
- baseline `code/p3_universe.py` → `data/p3/P3_universe.json` (this round's re-check did not change its basis)

## 1 Data reconciliation (source: `data/manifest/CTIS.csv`)

| Item | Count |
| :-- | --: |
| Slice rows | 43,054 |
| Recordings (`recording_uid` de-duplicated) | 4,956 |
| Recordings with both instrument and technique | 4,940 |
| Dropped (missing attribute) | 16 |
| Raw instrument names | 212 |
| Technique label strings | 386 |
| Instruments after normalisation | **201** |

**Basis declaration (important)**: the class universe recognises only the manifest's `technique` column, not "tokens cut straight from the file name". The two paths disagree in measurement (386 vs 329 label strings; cutting the file name directly adds a `演奏技巧` fallback class of 368), the difference coming from the manifest's parsing rules. The body text must state "the manifest's technique column is authoritative", otherwise a reviewer recomputing from file names will get a different set of numbers.

## 2 Instrument-name normalisation (22 raw names → 11 canonical names, 638 recordings; 212 → 201)

**L1 mechanical normalisation** (no semantics involved, 12 names / 395 recordings): strip the key designation (`G调梆笛→梆笛`, `A调曲笛→曲笛`, `G调新笛→新笛`, `F调巴乌→巴乌`, `壮剧马骨胡 D调→壮剧马骨胡`, `扬剧主胡 D调/F调→扬剧主胡`), strip the generation (`高腔赣胡 第2代→高腔赣胡`), strip the ordinal (`尖子号2→尖子号`), strip the hyphen (`莱芜梆子-梆胡→莱芜梆子梆胡`), unify whitespace and full/half-width forms (`滇葫（小二胡）`).

**L2 same-object merging** (requires judgement, 10 names / 243 recordings): version markers `（传统）/（改良）/（新）` folded into the base name — `盖板(传统)+盖板(新)→盖板`, `奚琴(传统)+奚琴(改良)→奚琴`, `中音奚琴(改良)→中音奚琴`, `陇剧陇胡(传统)+(改良)→陇剧陇胡`, `短箫(传统)→短箫`, `伽倻琴(改良)→伽倻琴`; alias `雅筝→雅托嘎` (雅托嘎 is the Mongolian zither); register qualifier `扬剧主胡（小西皮）→扬剧主胡` (小西皮 is a vocal style, not an instrument).

**PENDING, not merged (8 entries, awaiting a human decision)**: `萧` (probably a variant writing of "箫", which is absent from the disk), `中虎` (probably a miswriting of "中胡", which has another 36 entries), `二馨`, `福[新]`/`禄[新]`/`寿[新]` (probably new versions of the three-star drum), `铙` (whether the same instrument as "铙钹"), `南音三弦` (whether a regional variant of the sanxian).

## 3 Freezing rule for the class universe (pre-registrable text)

> Let n(t,i) be the number of recordings in the recording-level cell "technique t × instrument i" (`recording_uid` de-duplicated, slices not counted),
> I(t) = { i : n(t,i) ≥ 2 }. Technique t enters the class universe if and only if **|I(t)| ≥ 3** and **t ∉ E**;
> E = {音阶, 空弦音, 筒音, 单音, 乐曲片段, 特有演奏技法} (see §5).
> The sample size reported in the method section is always the **in-cell basis** Σ_{t∈L} Σ_{i∈I(t)} n(t,i), not the class-level total.

## 4 Frozen result

| Item | Count |
| :-- | --: |
| Classes | **43** |
| Usable recordings (in-cell basis) | **1,388** |
| Median / minimum per class | **10 / 6** |
| Cells (class×instrument units with n ≥ 2) | 581 |
| Classes with in-class n < 10 | 20 (of which 18 have n = 6, all sitting on the lower bound) |

Threshold sweep (after normalisation, content classes not yet excluded, equivalent to the baseline basis):

| Basis | Classes | Usable recordings | Median per class | Minimum | Cells |
| :-- | --: | --: | --: | --: | --: |
| cell≥1 and ≥2 instruments | 116 | 4,446 | 6 | 2 | 1,159 |
| cell≥2 and ≥2 instruments | 95 | 4,271 | 6 | 4 | 1,004 |
| cell≥3 and ≥2 instruments | 20 | 3,029 | 29 | 6 | 397 |
| cell≥5 and ≥2 instruments | 10 | 2,548 | 37.5 | 12 | 273 |
| cell≥1 and ≥3 instruments | 63 | 4,208 | 8 | 4 | 1,053 |
| **cell≥2 and ≥3 instruments** | **49** | **4,039** | **12** | **6** | 912 |
| cell≥3 and ≥3 instruments | 14 | 2,963 | 59.5 | 14 | 385 |
| cell≥5 and ≥3 instruments | 7 | 2,495 | 142 | 23 | 267 |

Details of the 43 classes (class | recordings):

| Class | Recordings |
| :-- | --: |
| 打音 | 191 |
| 滑音 | 167 |
| 长音 | 158 |
| 华彩 | 148 |
| 连音 | 142 |
| 颤弓 | 84 |
| 颤音 | 78 |
| 揉弦 | 56 |
| 连打 | 46 |
| 滚奏 | 18 |
| 单吐 | 17 |
| 下滑音 | 16 |
| 三吐 | 16 |
| 轮指 | 16 |
| 上滑音 | 14 |
| 双吐 | 14 |
| 泛音 | 14 |
| 滑揉 | 14 |
| 鼓音 | 12 |
| 扫弦 | 12 |
| 花舌 | 11 |
| 传统和声 | 10 |
| 抛弓 | 10 |
| 闷击 | 8 |
| 垫指滑音 | 8 |
| 腹颤音 | 6 |
| 叠音 | 6 |
| 鼓梆 | 6 |
| 鼓边 | 6 |
| 擦击 | 6 |
| 磕击 | 6 |
| 磕边 | 6 |
| 薄击 | 6 |
| 回滑音 | 6 |
| 垫弓 | 6 |
| 大上滑音 | 6 |
| 大下滑音 | 6 |
| 自然泛音 | 6 |
| 扫 | 6 |
| 摇指 | 6 |
| 琶音 | 6 |
| 长轮 | 6 |
| 弹挑 | 6 |

## 5 Exclusion rule (pre-registration text)

> Exclude 6 **content / playing-form** labels, on the grounds that they describe pitch material or playing form rather than a playing action:
> 音阶 (1,222), 空弦音 (679), 筒音 (466), 单音 (63), 乐曲片段 (31), 特有演奏技法 (190).
> Criterion = exact label-name match; all 6 names **exist** in the manifest (`dropped_missing` = empty).
> Same-family but literally different `跨弦音阶`/`单弦音阶`/`轻六音阶`/`筒音D`/`音程` are not on the list; measured, none of them passes the |I(t)| ≥ 3 threshold, so none enters the universe.

Before exclusion 49 classes / 4,039 recordings → after exclusion **43 classes / 1,388 recordings** (median 10, minimum 6).

## 6 Cell-by-cell alignment with the §7.2 table in the design draft

| §7.2 row | Design draft v2.8 `:547-549` | Measured this round (after merge normalisation) | Verdict |
| :-- | :-- | :-- | :-- |
| cell ≥5, all technique strings | 5 classes / 2,445 | cell≥5 and ≥2 instruments: 10 classes / 2,548; cell≥5 and ≥3 instruments: 7 classes / 2,495 | **not reproducible** (no setting gives 5 classes / 2,445) |
| cell ≥2 and ≥3 instruments | 49 / 4,037 / 12 (6) / test 2,043 (6, 4) | **49 / 4,039 / 12 (6)** | class count, median and minimum align; recordings +2 (merge normalisation); the test column is not reproducible |
| 6 content classes excluded (main basis) | 43 / 1,388 / 10 (6) / test 715 (6, 4) | **43 / 1,388 / 10 (6)** | all four items align; the test column is not reproducible |

Three things need handling:
1. **The first row's 5 classes / 2,445 has no corresponding setting**; change it to "cell≥5 and ≥2 instruments = 10 classes / 2,548" or drop the row outright (its original role was to show that "only content classes survive a strict threshold", which 10 classes / 7 classes serve equally, but the names of those 10 classes must be re-read before writing the conclusion).
2. **The test column (2,043 / 715) is not reproducible**: the two natural "half the instruments" splits give 659 (alternating by descending recording count; median 4, minimum 0, 10 classes with <4) and 666/722 (alternating in alphabetical order). This column should never have entered the main table — §7.2 `:558` already states that the main metric uses three-fold pooled (per-class test support = all 1,388 recordings), and the half-holdout is for reference only.
3. **Normalisation has zero effect on the main basis** (the 43-class list, 1,388 and the per-class counts are all identical; only the usable recordings of the "cell≥2 and ≥3 instruments" setting move 4,037→4,039). So the normalisation can be done safely, but **the body text must report 201 instruments after normalisation** and must stop writing 212.

## 7 Sensitivity

- The 20 classes with in-class n < 10 get a separate detail table and do not enter the main-table means; the 18 classes with n = 6 are declared in the pre-registration as "sensitivity/descriptive only".
- Technique label strings were not merged by synonymy (`扫`(6) vs `扫弦`(12), `滚奏`(18) etc.): per risk ⑤ in §7.2 the file-name string is authoritative, but when interpreting macro-F1 it must be stated that "one technique split into two classes depresses the score".
- Backbone/recipe follow §7.2 and v0.1: AlexNet + Swin-T, 224×224, ImageNet init, 10 epoch, AdamW 3e-4 / wd 0.05, cosine + warmup 1, LS 0.1, batch 64, AMP, seed 2.

## 8 Open questions and risks (must be handled before freezing)

① 8 PENDING instruments (§2) — the merge table is traced, so changing one row allows recomputation; if `福[新]/禄[新]/寿[新]` really are new versions of the three-star drum, 3 cells are affected.
② The numbers in the first row of §7.2 and in the two test columns (§6): they must be changed or deleted before the body cites them, otherwise recomputation from the json will not match.
③ The parsing basis must be pinned to the manifest's `technique` column (§1), with the generation rule for that column given in the method appendix; otherwise the 386 vs 329 difference cannot be explained.
④ The `演奏技巧` fallback class (which appears when tokens are cut straight from the file name, 368 entries) **does not exist** under the manifest basis — that is both good news and a warning: regenerating the manifest changes the class universe, and the source must not be swapped after freezing.

## 9 Recomputation and assertions

```
python code/ctis_inst_norm.py      # 212 → 201, 22 merged pairs, 638 recordings
python code/p3_universe_norm.py    # 43 classes / 1,388 recordings / median 10 / minimum 6
```
Assertions: `recordings(4,956) = with_attrs(4,940) + 16`; `sum(cells) == 4,940`; before exclusion 49/4,039, after exclusion 43/1,388; `dropped_missing == []`.

## 10 Change log

- v0.2 (2026-11-05): added the instrument-name normalisation merge table (212→201), re-checked the manifest basis (43/1,388/10/6 reproduced), reconciled §7.2 cell by cell (first row and test columns not reproducible), and rewrote the exclusion rule as pre-registrable three-part prose.
- v0.1: `worklog/exp-plan/P3-预注册草案.md` (splits/arms/metrics/decision rules, without normalisation and reconciliation).
