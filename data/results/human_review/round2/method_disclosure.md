# Verification of reproducibility-detail disclosures (round 2)

Scope: the four method statements named in review comments 17/19/21 (`report/human-review-2026-11-11/review.txt:439-567`).
Method: read the `code/` implementation line by line and cite line numbers as evidence; values are recomputed by the read-only script `code/round2_method_facts.py` from `data/manifest/summary.json`, `data/splits/v2/*.csv`, `data/results/e1_metrics.csv` and `data/results/paper_verify.json`, with results written to `data/results/human_review/round2/method_facts.json`.
Rerun: `python3 code/round2_method_facts.py --root . --json data/results/human_review/round2/method_facts.json` (read-only, no subprocess).

Three statements need rewriting (conclusions first):

1. §4.2 "verified against the audio where audio is shipped" — only CTIS actually verified against audio (SHA-1 over all 4,956 WAV files); the other four libraries use a pure file-name rule, and although pianos and CNPM do ship audio, its names are not in the same namespace as the eval release, so nothing was linked. The two manual-check columns of the existing spot-check tables (5 × `*_spotcheck.csv`) are entirely empty.
2. The "group" in §4.3 is not a recording: a group is a **block of two adjacent slices within one recording**, and one recording's slices can enter both arms at once (578 of the 963 CTIS test recordings are also in the injection pool; bel canto 40/40 all are). "singleton recordings assigned to test" actually means "recordings that contribute only 1 slice".
3. In `analyze_e1.py` the macro-average denominator of §5.3/§5.4 is **recomputed dynamically in every resample** (empty classes dropped), and a CTIS resample misses 8.09/201 classes on average; whereas `sealed_eval_delta.py` (the sealed evaluation table) uses a **fixed denominator** (absent classes score 0). The paper describes only the former.

---

## A. Recording identity and audio verification (§4.2)

### What the paper says

`paper/sections/04_materials.tex:43-50`:

> Every slice is mapped to a `recording_uid` by a rule specific to the library, derived from the release format and verified against the audio where audio is shipped: CTIS by the per-class file stem, prefixed by the class code when a stem repeats across classes; music genre by the pair of track index and duration that the file name encodes; pianos by brand, recording number and duration; CNPM by joining the file name to the dataset's own `label.csv`; and bel canto by class and recording number. Parsing succeeded for 110,415 of 110,415 slices.

### What the code does

**1) There is only one verification action: hash every CTIS WAV file with SHA-1 and fold same-hash recordings.**

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
# code/manifest.py:239-243  CTIS: hash all n_rec recordings one by one, not a sample
if do_audio_hash:
    print("  audio de-duplication: byte sha1 over %d wav files ..." % n_rec, flush=True)
    for i, gidx in enumerate(sorted(recs), 1):
        cls, fn = recs[gidx]
        audio_hash[uid_of[gidx]] = sha1_file(os.path.join(adir, cls, fn))
```

```python
# code/manifest.py:252-254  fold same-hash recordings onto the first uid in insertion order
for h, uids in by_hash.items():
    for uid in uids:
        canon[uid] = uids[0]
```

The recording number is itself a **positional enumeration**, not a semantic parse: `code/manifest.py:189-197` accumulates a global number from `sorted(classes)` plus `sorted(filenames)` within each class, so "parsing succeeded 110,415/110,415" is trivially true on the CTIS side (every file is assigned a number); it only shows that the mel and wav enumerations **agree in count**, not that a number agrees with the recording's **content**. There are two real content-side pieces of evidence: the mel→wav and wav→mel set difference is 0 (`recordings_without_mel=0`, `mel_gidx_without_audio=0`, fields at `code/manifest.py:332-333`), and the JPEG SHA-1 de-duplication of 43,054 slices (`code/manifest.py:277-282`).

**2) The identity rule for the four other libraries is pure file name, and `audio_hash` is written as an empty string.**

- music genre: `audio_hash=""` at `code/manifest.py:403`; `audio_note` (429) states "no audio on disk, eval features only"; `dedup_rule` (441) states "no audio on disk, so audio-level de-duplication is impossible".
- pianos: `audio_hash=""` at `code/manifest.py:512`; 478-479 states "audio lives in `data/audio/<brand>/<recording number>.wav`, its names do not match the eval recording numbers (6009 vs 009), there is no reliable mapping, so audio_hash is not filled".
- CNPM: `audio_hash=""` at `code/manifest.py:578`; for this library only the `label.csv` join was done (533-537 read the GBK table, 589-591 write `csv_join`), measured `csv_rows=287`, `mel_recs=287`, `mel_not_in_csv=0`.
- bel canto: `audio_hash=""` at `code/manifest.py:626`.

**3) Per-library verification surface for the five libraries (values from `data/manifest/summary.json` and `data/datasets_inventory.json`, script section A):**

| Library | Verified against | Recordings / slices | Audio on disk | Verification method in code | Out of reach |
|---|---|---|---|---|---|
| CTIS | WAV × JPEG | 4,956 / 43,054 | 4,956 WAV (`manifest.py:646` notes ~13.3 GB) | full SHA-1 (`manifest.py:76-84,239-243`); 2 same-hash groups / 4 recordings; mel↔wav set difference 0; JPEG SHA-1 de-duplication | recording-number correctness has no external authority to compare against; no manual review (spot-check tables empty) |
| music genre | none (features only) | 1,713 / 36,375 | 0 | file-name three-field parse only `<track>_<dur>_<pos>` (`357-411`), `bad_filename_count=0` | no audio, identity unverifiable |
| pianos | audio on disk but not the same source | 668 / 18,352 | 668 WAV (about 0.9 h) | file-name parse only (`474-523`); `audio_hash=""` (512) | audio file names have no reliable mapping to eval recording numbers, not linked |
| CNPM | `label.csv` × file name | 287 / 2,724 | 287 (196 mp3 + 91 wav) | file name with `_<slice position>` stripped, joined to `label.csv`, 287/287 matched (`533-591`) | on-disk audio took no part in identity verification; labels depend on the dataset's own csv |
| bel canto | none (features only) | 203 / 9,910 | 0 | file-name parse only `<cls>/<rid>` (`595-637`) | no audio, identity unverifiable |

Total 110,415 slices (43,054 + 36,375 + 18,352 + 2,724 + 9,910), consistent with the paper.

**4) Five implementation details that would mislead a disclosure.**

- The field name `uid_parse_rate` does not match the implementation: at `code/manifest.py:320` it is actually "share of slices carrying an instrument label" (CTIS 0.9949), **not** the parse success rate from file name to recording ID; it cannot be used to answer the reviewer.
- CTIS mel name parsing has no exception guard: `gidx, pos = fn.rsplit(".", 1)[0].split("_")` at `code/manifest.py:210` raises `ValueError` and fails the whole library as soon as a name is non-conforming (`--limit` smoke runs only truncate the row count, they do not skip bad names); this is the other half of the "100% parsed" figure — bad names are not counted, they crash the build outright.
- The documented basis for choosing `canon` disagrees with the implementation: the implementation takes the first uid in insertion order (`252-254`), while the metadata says "lexicographically smallest uid in the group" (`349-350`). In this data both duplicate pairs are within one class and the two orders give the same answer, so no discrepancy arose, but that self-description should be fixed before disclosure.
- `audio_hash_done` cannot be read as "was audio verification performed": `code/manifest.py:346` writes `bool(dup_audio)`, i.e. "was duplicate audio **found**"; CTIS is True only because 2 duplicate groups were found, and the other four are False because no hash was computed at all.
- The "verification" in the cache script is something else: `code/ccmusic_cache.py:127-146` reads back 200 random memmap images per library and compares them byte by byte with the originals on disk (`max_abs_diff`/`n_nonzero`); it only covers cache integrity and cannot be cited as identity verification.
- Manual review never happened: in the 5 `*_spotcheck.csv` files generated by `code/manifest.py:142-179`, the `check_image_matches_recording` and `check_no_other_recording_same` columns are **entirely empty** (no non-empty cell beyond the header).

### Accurate statement that can be disclosed

> Recording identity was verified against shipped audio only for CTIS, where all 4,956 WAV files were hashed (SHA-1; two duplicate pairs folded to 4 canonical recordings) and the mel/WAV index sets matched exactly; for the other four libraries the mapping is a pure file-name rule, and although pianos and CNPM do ship audio, its file names are not in the same namespace as the eval release, so it was not linked.

### What cannot be disclosed

- Cannot claim "identity verified recording by recording" or "manually reviewed": the CTIS check de-duplicates the audio files themselves, it is not a semantic check that "recording k equals some audio segment"; the other four libraries have no content-level evidence at all.
- Cannot give a per-library "unmatched/ambiguous count": on the CTIS side no ambiguity count exists (the enumeration is a total function), and on the four-library side only bad file names were counted (all 0); there is no mechanism to detect "parsed to the wrong recording".
- Whether the pianos / CNPM audio can be aligned with eval cannot be determined from the code: `manifest.py:478-479` only says there is no reliable mapping and leaves no record of an attempt.

---

## B. Group key and two-arm assignment (§4.3)

### What the paper says

`paper/sections/04_materials.tex:72-75`:

> Groups are sorted by their candidate-group key and assigned alternately to sealed test and injection pool, with singleton recordings assigned to test.

Lines 55-60 of the same section also state: CTIS has 77 groups of identical images covering 175 slices; "one music-genre group spans three classes"; duplicate groups are always pushed to a single side.

### What the code does

**1) How groups are built (not per recording, but as blocks of "two adjacent slices").**

```python
# code/splits_v2.py:258-272
dsu = DSU()
for rec in Q:
    rs = by_rec[rec]
    for i, r in enumerate(rs):
        dsu.find(r["sample_id"])
        if i % 2 == 1:
            dsu.union(rs[i - 1]["sample_id"], r["sample_id"])   # two adjacent slices form a block
by_hash = defaultdict(list)
for r in cls_rows:                     # whole class, not just Q
    if r["content_hash"]:
        by_hash[r["content_hash"]].append(r["sample_id"])
for ids in by_hash.values():
    for x in ids[1:]:
        dsu.union(ids[0], x)
```

Slices inside `by_rec` are sorted by `slice_index` (`code/splits_v2.py:211-212`; `slice_index` is already cast to int at line `86`, so the sort is numeric), so the blocks are (slice 0,1), (slice 2,3), … and **different blocks of the same recording can land in different arms**. Hash merging is done only within a class (`by_hash` is built from `cls_rows`, `266-272`); groups are never merged across classes.

**2) The group key is "the smallest slice candidate key in the group", and a candidate key is the first 12 hex digits of the slice JPEG's SHA-1.**

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

That is: the key is neither the recording ID nor a file-name prefix, but **the lexicographically smallest 12-digit image-hash prefix in the group**; when the key is missing it falls back to `sample_id`; ties are then ordered by DSU root id.

**3) Alternating assignment and the singleton branch.**

```python
# code/splits_v2.py:283-288
for i, g in enumerate(order):
    rs = groups[g]
    if i % 2 == 0 or all(r["n_slices"] <= 1 for r in rs):
        qtest_rows.extend(rs)
    else:
        qpool_rows.extend(rs)
```

`n_slices` is the recording's total slice count **within the class** (`code/splits_v2.py:92-94`), so `all(n_slices <= 1)` is equivalent to "this recording contributes only 1 slice". Groups with an even index go to test, odd ones to the injection pool, and singleton groups go to test regardless of parity.

**4) The single-side constraint on duplicate images appears in two places:**

```python
# code/splits_v2.py:290-301  a hash already present on the test side pulls the whole recording from R/C into test (shadow bookkeeping)
# code/splits_v2.py:343-367  trim_global_dups: pri = qtest>qpool>r>c, mapped to the allowed sides, wrong side means dropped and counted
```

**5) Assertions (`code/splits_v2.py:425-495`, any failure `exit 2`, see `567`):** `A0_disjoint`(477), `A1_test_val_disjoint`(478), `A1_train_has_q`(479), `A1_alpha_material`(480), `A2_sample_disjoint`(488), `A1_n_c_side_match`(491-492), `per_class_test_ge1`(493), `hash_same_side`(494); results are written to `assertions.json` (`560-561`).

**6) Measurements (script section B, recomputed from `data/splits/v2/*.csv` with groups rebuilt):**

| Library | Test/pool slices | Test/pool recordings | Recordings in both arms | Groups (one-slice/two-slice) | Cross-recording groups | Cross-side groups | Adjacent pairs split | One-slice recordings (all in test) |
|---|---|---|---|---|---|---|---|---|
| CTIS | 4,605 / 4,178 | 963 / 753 | **578** | 4,682 (581 / 4,101) | 1 | 0 | 0 | 214 |
| music genre | 3,648 / 3,613 | 337 / 340 | 335 | 3,716 (171 / 3,545) | 0 | 0 | 0 | 0 |
| pianos | 1,884 / 1,862 | 135 / 134 | 133 | 1,908 (70 / 1,838) | 0 | 0 | 0 | 0 |
| CNPM | 271 / 268 | 54 / 50 | 47 | 285 (31 / 254) | 0 | 0 | 0 | 0 |
| bel canto | 979 / 977 | 40 / 40 | **40** | 989 (22 / 967) | 0 | 0 | 0 | 0 |

"Test ∩ (R ∪ C) recordings = 0" holds for all five libraries; "same hash across sides = 0" holds for all five. That is: at the **recording level** test does not intersect train/calibration, but **recordings do overlap between test and the injection pool** — which is exactly what the injection pool is for (new slices of the same recording), and the paper's sentence does not make this clear.

Also: the CTIS rebuild yields 1 cross-recording group (a same hash merged two recordings into one group, a product of `266-272`), consistent with the paper's "two identical pairs"; no library has a cross-class group and **the code has no branch that merges across classes** — the paper's §4.3 sentence "one music-genre group spans three classes" has no counterpart in `splits_v2.py` (a cross-class same hash can only be pushed to one side by `trim_global_dups`, not merged into a group).

### Accurate statement that can be disclosed

> Each library is partitioned into blocks of two consecutive slices of the same recording, blocks are merged when they share an identical JPEG hash, and the resulting groups are sorted by the smallest 12-bit hash prefix of their slice keys and dealt alternately to sealed test and injection pool; groups whose single slice comes from a one-slice recording always go to test, so a recording's slices can appear in both arms while no recording is shared between the sealed test set and the training or calibration arms.

### What cannot be disclosed

- "Groups" cannot be used to mean recordings: in the code a group is a block of 1–2 slices, and a recording spanning both arms is the norm (CTIS 578/963).
- Cannot disclose "cross-class same-hash group merging": the code has no such branch; if the paper means the audit script's grouping, separate evidence is needed (not found in `splits_v2.py` this round).
- Cannot give the concrete key-value sequence used for ordering: `candidate_group` exists only in intermediate CSVs and `order` is not persisted; what can be reproduced is the construction of the key and the ordering rule.
- Constants the paper does not explain cannot have their meaning "read off" the code: `q_c=round(0.20n)` and `r_c=round(0.15n)` (`splits_v2.py:217-219`; `Q_SHARE=0.20`/`R_SHARE=0.15` at `65-66`) and the `MIN_REC=5` small-class filter (`67`, `406`) are capacity design, unrelated to this section's statement but worth listing in the supplement.

---

## C. macro-F1 and recording-cluster bootstrap (§5.3 / §5.4)

### What the paper says

`paper/sections/05_protocol.tex:70-71`:

> Macro-F1 averages only over classes with non-zero support or predictions; classes absent from a panel are excluded rather than scored as zero.

### What the code does

**1) The class set comes from the maximum label of the reference arm, and the denominator is recomputed each time.**

```python
# code/analyze_e1.py:96-106
def macro_f1(conf):
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    keep = conf.sum(1) + conf.sum(0) > 0        # present in truth or prediction
    f1 = f1[keep]
    return float(np.nanmean(f1)), int(keep.sum())
```

```python
# code/analyze_e1.py:162
nc = int(y_ref.max()) + 1    # labels already remapped to 0..C-1, C = number of classes in that library's test set
```

`C` therefore equals the class count of that library's Q_test; `n_classes` in `e1_metrics.csv` agrees per library (201/16/8/5/4, counting only, without checking the mapping from class labels to `class_idx`). Every class appears in the test set (`splits_v2.py:493` asserts `per_class_test_ge1`), so for the point estimate the denominator is exactly C.

**2) Under bootstrap resampling the denominator is dynamic.**

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

If a resample is missing a class, that class's `conf` row and column are all 0 → `keep=False` → it is dropped and **the denominator shrinks**; it is not scored 0. This is the behaviour the paper describes at lines 70-71.

**3) Another script uses a different basis (the sealed evaluation table uses a fixed denominator).**

```python
# code/sealed_eval_delta.py:143
mask = np.bincount(y, minlength=C) > 0        # fixed once from the point sample
# code/sealed_eval_delta.py:45
return f1[:, mask].mean(1)                    # denominator fixed to the classes present in the point sample
```

Whereas `macro_f1` at `sealed_eval_delta.py:21-26` averages over **all C** (absent classes counted with F1=0), and the point estimate (`164`) takes that branch. The two artifacts are therefore not strictly on the same basis.

**4) Scale of the class shortfall (script section C, closed-form expectation, not a rerun):**

| Library | Classes | Test recordings | Recordings per class | Classes with <5 recordings | E[missing classes per resample] | P(at least 1 class missing) |
|---|---|---|---|---|---|---|
| CTIS | 201 | 963 | 1–21 | 89 | **8.09** | **0.9999** |
| music genre | 16 | 337 | 18–41 | 0 | 0.000 | 0.0000 |
| pianos | 8 | 135 | 10–18 | 0 | 0.000 | 0.0000 |
| CNPM | 5 | 54 | 8–15 | 0 | 0.000 | 0.0002 |
| bel canto | 4 | 40 | 6–19 | 0 | 0.002 | 0.0021 |

So: the four small libraries almost never miss a class (the dynamic and fixed denominators are equivalent there), **CTIS misses about 8 classes in nearly every resample**, its interval is computed over about 193 classes, and under a fixed denominator those classes would score 0. CTIS is the only library where the two bases produce a visible difference.

A magnitude obtainable by arithmetic alone: if the ~8 absent classes score 0 under the fixed basis, the uplift of the dynamic basis over the fixed one is bounded by about `m/(C−m) ≈ 8/193 ≈ 4.1%` (relative to the mean, not percentage points); the true F1 of the absent classes is unknown, so the actual uplift can only be smaller.

### Accurate statement that can be disclosed

> Macro-F1 is averaged over the classes present in the evaluated sample, so the denominator is re-derived inside every bootstrap replicate (a class absent from that resample is omitted rather than scored zero); for the CTIS panel this affects essentially every replicate, since about 8 of its 201 classes are missing from a typical recording-level resample.

### What cannot be disclosed

- Cannot disclose "the denominator of each resample": the script returns only a `float` (`keep.sum()` at `analyze_e1.py:106` is discarded) and it is not persisted.
- Cannot disclose the numerical difference between the two bases on CTIS: that requires reading `runs/sealed_eval/*.npz` and rerunning the bootstrap (the WSL-side interpreter has no numpy and the Windows-side python was unavailable this round); not done.
- The fixed denominator in `sealed_eval_delta.py` conflicts with lines 70-71 of the paper and is an undisclosed mixing of bases; unifying it means picking one side and recomputing that table.

---

## D. Coverage boundaries of the independent verification script (§5.7)

### What the paper says

`paper/sections/05_protocol.tex:155`:

> All reported numbers were re-derived from stored artifacts by an independent verification script.

`paper/sections/09_statements.tex:36-44` (page 24) already limits itself:

> The numerical verifier checks artifact identities, recalculates seed means and differences, and compares generated table rows with the manuscript. Bootstrap intervals are produced by separate scripts from sealed predictions; the verifier does not independently rerun every bootstrap or validate every prose claim.

### What the code does

**1) Two layers, 189 assertions (`data/results/paper_verify.json:2-3`: `passes=189, fails=0`).**

- identity layer: recompute derived quantities (means, differences, shares, sums, interval containment) and assert agreement with the artifacts.
- text layer: rebuild the string from the table row's typesetting and search for its presence in the normalised LaTeX text (described in the docstring at `code/verify_paper_numbers.py:5-16`; the scattered token assertions at `419-441` include `"110,415"`).

**2) Which artifacts it reads (by line number):**

| Artifact | Location |
|---|---|
| `data/results/e1_metrics.csv` | `95-106` |
| `data/results/e1_delta.csv`, `data/results/delta_A2/e1_delta.csv` | `189-190` |
| `data/splits/v2/<lib>_{qtest,qpool,r,c}.csv` | `113-137` |
| `data/results/feasibility.csv` | `140-145` |
| `data/results/p3/p3_metrics.csv`, `p3_delta.csv` | `251-262` |
| `data/results/cost.json` | `265-271` |
| `data/results/b2/e1_delta.csv` | `347-364` |
| `data/splits/negcontrol/negcontrol.json` | `350-352` |
| `data/results/p3b_leak*`, `p3b_instr*` | `375-381` |
| `data/results/c1/e1_delta.csv` | `464-476` |
| `data/results/human_review/{corrected_recording_delta,selftest,ctis_seed_stratified_bootstrap}.csv` | `303-304, 479-498` |
| `code/analyze_e1.py` source text | `500-502` (asserts the source does not contain `nc, w=rec_n`) |

**3) What it asserts (sample):** totals of 109,831 slices / 7,756 recordings / 234 classes (`200-204`); Δ = difference of the two side means (`217-221`); the CI contains the point estimate (`227-231`); the SESOI verdict agrees with the `meets_sesoi` column (`232-236`); headroom falls in [0,1] (`243-248`); P3, cost, the A2 share, the row-by-row text of tables 1–6, the B1 decomposition, C1 and the S1 supplementary table (`258-502` across sections).

**4) Independent of what:** of the **manuscript's numeric text**, not of the analysis pipeline. It never opens `runs/sealed_eval/*.npz`; the CI is read straight from `ci_lo/ci_hi` in `e1_delta.csv` (`229`); and the bootstrap is not rerun even once — consistent with the self-limitation in 09_statements.

**5) What it can and cannot catch:**

- Can catch: numbers in the manuscript's tables that were altered or miscopied (text layer, row by row); a Δ in `e1_delta.csv` inconsistent with the seed means in `e1_metrics.csv`; a CI that does not contain the point estimate; a library's class/recording count disagreeing with the split CSV; a SESOI flag contradicting Δ/CI.
- Cannot catch (examples):
  1. a wrong resampling unit — if the script treated "slices" as clusters, `e1_delta.csv` and the verifier would be wrong in the same way, and the verifier would still report 189/0;
  2. a wrong macro-average denominator basis (e.g. scoring absent classes 0) — the verifier only compares arithmetic relations between CSVs and never recomputes a confusion matrix; this is exactly where `analyze_e1` vs `sealed_eval_delta` diverge, as found in section C, and the verifier cannot detect it;
  3. a number string in the wrong place — the text layer only requires the rebuilt token to appear as a substring anywhere in the **whole** normalised LaTeX text (`needle in self.text`, `163-170`), so moving one table's number into another table still passes;
  4. prose-level claims ("independent verification", "parsing succeeded for 110,415/110,415" and similar unquantified descriptions) are not covered, apart from the few numbers hard-coded as tokens at `419-441`.
- Also: `paper_verify.json` persists only the `passes` count and the `fails` details (`505-512`); the check names of passing items are not persisted, so "which assertions the 189 cover" is not auditable from the artifacts and can only be read from the source.

### Accurate statement that can be disclosed

> A verification script re-derives 189 numeric checks from stored aggregate artifacts (result tables, split CSVs and cost files), including seed means, paired differences, bootstrap-interval containment and the text of every reported table row; it does not re-read the sealed per-sample predictions or re-run the bootstrap, so it is independent of the manuscript text but not of the analysis pipeline that produced those artifacts.

### What cannot be disclosed

- "independent" cannot be upgraded to "independently recomputes the whole pipeline": the npz→CSV step is not recomputed by any second implementation.
- Cannot disclose the full list of assertions it checked: `paper_verify.json` persists only `passes/fails` and some panel totals; the individual check names are not persisted (if needed, the script should write every check name into the JSON).
- Cannot prove "the 189 cover every number in the body": the assertions are a hand-picked token set, not an extraction of all numbers in the text.

---

## Appendix: boundaries of this verification

- The Windows-side `pwsh` backend became unavailable partway through this round (`sandbox mode "workspace-write" is requested but no sandbox backend is usable on this host ... Runner failure: windows-acl-run: --temp is not an existing directory: C:\Users\hmj\AppData\Local\Temp\dsh-TMIimL`), so the read-only accounting was run under WSL with `python3`; `D:\XiaBao\AIArtDetect\envs\fapn\python.exe` was not used (its subprocess was refused separately, and torch was not needed here).
- Read-only throughout: the only files created were `code/round2_method_facts.py`, `data/results/human_review/round2/method_disclosure.md` and `data/results/human_review/round2/method_facts.json`.
- Items not verified (explicit gaps): `runs/sealed_eval/*.npz` was not opened (no numpy under WSL, Windows-side python unavailable), so section C has only the closed-form expectation of missing classes, not a measured difference between the two denominator bases; the cross-class same-hash grouping basis of `data/splits/_audit_summary.csv` was not pursued (relevant to "one group spans three classes" in section B); the details of the 2 duplicate audio groups in CTIS are in `duplicate_audio_groups` of `data/manifest/summary.json` (the script looked under two candidate names, `pairs_summary.json`/`ctis_pairs.json`, and missed, so the pairing is not expanded in this report).
