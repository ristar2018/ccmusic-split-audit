# -*- coding: utf-8 -*-
"""Build D:/VSCode/Datasets/README.md

Sources: local files (inventory) + data/ccmusic/*.readme.md (official notes and BibTeX)
+ data/actual_stats.json (measured audio duration).
BibTeX is always copied verbatim from the official README, never rewritten by hand, so citation metadata cannot drift.
"""
import io
import json
import os
import re

DATA = r"D:\VSCode\Datasets"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_DIR = os.path.join(ROOT, "data", "ccmusic")
OUT = os.path.join(ROOT, "report", "_datasets_README.md")

MS_ORG = "https://www.modelscope.cn/organization/ccmusic-database?tab=dataset"
LIB = "https://ccmusic-database.github.io"

# ---------------------------------------------------------------- dataset definitions
# key: ModelScope repo name
DS = [
    dict(
        key="pianos", zh="Piano sound-quality evaluation", task="brand identification / pitch detection / subjective sound-quality regression",
        docs="668 piano recordings from 8 brands; each recording is cut per key, with pitch and 4 dimensions of subjective sound-quality scores annotated.",
        page=LIB + "/database/ccm.html#shou1",
        origin="CCMusic multi-purpose library (ccm.html#shou1); the raw collection is brand piano audition recordings; paper: Zhou et al. (2024).",
        intro=[
            "Perceptual sound-quality study of what actually differs between brands/models on the same instrument and the same piece.",
            "The default subset is split by brand (8 directories) with single-key/short-phrase clips; the eval subset crops each recording into 496×369 feature images.",
        ],
        extra="`data/mel/*/<id>.jpg` are full-recording mel strips (height 128, width 482–854), one of the few representations that keeps whole-recording context.",
    ),
    dict(
        key="music_genre", zh="Music genre classification", task="genre classification (3 tiers: 2 coarse classes / 9 classes / 16 fine classes)",
        docs="1,713 tracks with three-tier genre labels: classic vs non-classic -> 9 classes -> 16 fine classes.",
        page="https://doi.org/10.5281/zenodo.5676893",
        origin="CCMusic multi-purpose library (ccm.html, Zenodo 10.5281/zenodo.5676893); no standalone paper, released with the CCMusic main reference.",
        intro=[
            "109,125 feature images (second only to CTIS), the highest download count among the 13 datasets (13,745).",
            "The three tiers support hierarchical classification and analysis of whether fine-grained errors concentrate in adjacent classes.",
            "Only the eval feature images are held locally, no raw audio — audio-domain methods (augmentation, resampling, SSL front ends) need `audio.zip`.",
        ],
        extra="Tier three is not present for every class: `11_rock` has 3 fine-class subdirectories, whereas `3_symphony` is slice images directly.",
    ),
    dict(
        key="chest_falsetto", zh="Chest vs falsetto voice", task="4-class vocal production classification (male/female × chest/falsetto)",
        docs="1,280 clips, 10.7 minutes, 4 classes.",
        page=LIB + "/database/ccm.html#shou3",
        origin="CCMusic multi-purpose library (ccm.html#shou3), released with Zenodo 10.5281/zenodo.5676893.",
        intro=[
            "One of the smallest tasks; the label is the vocal production mode, not the pitch content.",
            "Clips are very short (10.7 min / 1,280 clips ≈ 0.5 s each) and feature images map one-to-one onto clips, so there is no slice inflation from one track yielding many slices.",
        ],
        extra="It is therefore **not** a victim of slice leakage: K=1.0, each recording appears exactly once.",
    ),
    dict(
        key="acapella", zh="A cappella vocal-skill rating", task="9-dimensional subjective score regression / vocal-skill evaluation",
        docs="132 clips, 22 singers, 6 songs, 3.47 hours, 9 dimensions of 10-point subjective scores.",
        page=LIB + "/database/ccm.html#shou2",
        origin="CCMusic multi-purpose library (ccm.html#shou2); paper: Li & Zhang (2022), Applied Sciences 12(19):9931.",
        intro=[
            "**Audio not downloaded in this round**: only `dataset_infos.json` was fetched for this dataset, and the directory was removed during cleanup.",
            "To fetch it again: `python code/ccmusic_download.py --dataset acapella` (about 1.3 GB).",
        ],
        extra="It is the only vocal data with human-vs-human subjective scores (22 singers × 9 dimensions); the sample is small for score regression, so it is better used as an external validation set.",
    ),
    dict(
        key="CNPM", zh="Chinese pentatonic mode identification", task="mode identification (four axes: gong system 12 / tonic 12 / mode type 5 / class 6)",
        docs="287 tracks, 14.3 hours, four axes annotated jointly; eval uses 20 s slices.",
        page=LIB + "/database/csmtd.html#shou10",
        origin="China Conservatory of Music Chinese pentatonic modes dataset, integrated by CCMusic (csmtd.html#shou10); paper: Wang et al., ISMIR 2022.",
        intro=[
            "The four axes split mode identification into diagnosable sub-problems: is the gong system wrong, or the tonic.",
            "`data/label.csv` is **GBK**-encoded (Excel export) with 11 bilingual column headers; reading it as UTF-8 yields mojibake.",
        ],
        extra="Audio mixes mp3 and wav (196 + 91) at inconsistent 44.1/48 kHz sample rates, so preprocessing must resample to one rate.",
    ),
    dict(
        key="Guzheng_Tech99", zh="Guzheng playing technique (frame-level)", task="frame-level multi-label technique detection (7 classes)",
        docs="99 guzheng pieces, 151 minutes, 7 techniques annotated per frame, 63,352 labels; the official split is 79/10/10 by piece.",
        page=LIB + "/database/csmtd.html#Tech99",
        origin="CSMTD guzheng frame-level technique dataset, integrated by CCMusic (csmtd.html#Tech99); paper: Li et al., ICASSP 2023.",
        intro=[
            "The **only one of the 13 datasets split by piece** (train 79 / validation 10 / test 10), and the only one with frame-level annotation.",
            "Each piece has 1 flac + 1 frame-level CSV + 1 mel strip (5,360–7,853 × 128), suited to frame-level sequence modelling.",
        ],
        extra="Its 0.862 frame-level F1 is therefore not comparable with the 0.98+ slice-level accuracy of the other datasets — the evaluation protocols differ.",
    ),
    dict(
        key="instrument_timbre", zh="Instrument timbre perception", task="16-dimensional timbre-descriptor score regression",
        docs="61 instruments (37 Chinese / 24 Western), each rated on 16 descriptors with a 9-point scale.",
        page=LIB + "/database/ccm.html#shou4",
        origin="CCMusic multi-purpose library (ccm.html#shou4); paper: Jiang et al. (2020), Applied Sciences 10(3):789.",
        intro=[
            "Grounds timbre in 16 interpretable descriptors rather than an abstract embedding, which suits representation-semantics alignment.",
            "`data/mel/*/<id>.jpg` are full-recording mel strips (417–566 × 128); `Chinese.csv` / `Western.csv` each hold 16 score columns.",
        ],
        extra="12 of the 16 terms in the official card differ from the actual CSV column names (the card says warm, the CSV says soft), so label mapping must follow the CSV.",
    ),
    dict(
        key="song_structure", zh="Song structure analysis", task="9-class functional segment annotation (intro/verse/chorus/bridge ...)",
        docs="300 tracks, 9 structural function labels, 4.25 GB.",
        page="https://doi.org/10.5281/zenodo.5676893",
        origin="CCMusic multi-purpose library (Zenodo 10.5281/zenodo.5676893), released with the CCMusic main reference.",
        intro=[
            "**Not downloaded in this round**: it is in none of the p1/p2/p3/p5/p7 research presets, and `--project all` does not include it either.",
            "To fetch it: `python code/ccmusic_download.py --dataset song_structure` (4.25 GB, audio included).",
        ],
        extra="Structure labels are segments on a time axis; the P1 slice-leakage audit needs its segment boundaries to define slices that come from the same segment.",
    ),
    dict(
        key="GZ_IsoTech", zh="Guzheng technique (clip-level)", task="8-class clip-level technique classification",
        docs="2,824 clips (2,328 virtual instrument + 496 human performance), 8 techniques; the official release provides train/test.",
        page=LIB + "/database/csmtd.html#GZTech",
        origin="CSMTD guzheng technique dataset, integrated by CCMusic (csmtd.html#GZTech); paper: Li et al., ISMIR 2022.",
        intro=[
            "Same source as Guzheng_Tech99 at a different granularity: one technique per clip here, one technique per frame there.",
            "Virtual instrument sources are 82% of the data, which gives a built-in domain-shift contrast (synthetic vs human) and allows cross-domain experiments with no extra collection.",
        ],
        extra="`data/audio/{train,test}/<8 classes>/` already follows the official split; the 2,899 eval images are clip-level feature images.",
    ),
    dict(
        key="bel_canto", zh="Bel canto / Chinese folk singing", task="4-class singing-style classification",
        docs="203 clips, 5.08 hours, 4 classes (bel canto / folk and their sub-classes).",
        page=LIB + "/database/ccm.html#shou9",
        origin="CCMusic multi-purpose library (ccm.html#shou9), released with the CCMusic main reference.",
        intro=[
            "The most specialised singing data: separating bel canto from folk depends on resonance and timbre, not melody.",
            "Only the eval feature images (29,730) are held locally, no audio; fetch `audio.zip` if audio is needed.",
        ],
        extra="The official README cites the CCMusic main reference directly; there is no standalone dataset paper.",
    ),
    dict(
        key="erhu_playing_tech", zh="Erhu playing technique", task="11-class technique classification (plus 4-class / 7-class subsets)",
        docs="1,253 clips, 25.8 minutes, 11 classes; the 4-class and 7-class subsets hold 635 clips each.",
        page=LIB + "/database/ccm.html#shou8",
        origin="CCMusic multi-purpose library (ccm.html#shou8); paper: Wang et al., arXiv:1910.09021 (FCN technique detection).",
        intro=[
            "`data/audio/` holds three parallel directories: `11_classes` (1,253), `4_classes` (635), `7_classes` (635).",
            "The 4/7-class sets are **re-labellings** of the same recordings, so the local audio file count 2,523 ≠ the 1,253 distinct recordings — duration totals double-count.",
        ],
        extra="The measured 0.83 h therefore sums the three directories; counting distinct recordings gives ≈ 25.8 min, matching the official documentation.",
    ),
    dict(
        key="CTIS", zh="Chinese traditional instrument sound", task="instrument identification (219 classes) / technique identification (embedded in file names)",
        docs="4,956 recordings, 32.6 hours, 209 instruments (219 class directories), recorded in an anechoic chamber.",
        page=LIB + "/database/ctis.html",
        origin="Chinese Traditional Instrument Sound dataset (CTIS), integrated by CCMusic; original paper: Liang et al., CSMT 2019.",
        intro=[
            "The only subset recorded in a standard anechoic chamber and usable as a clean reference signal (32 B&K microphones, 44.1/48/96 kHz mixed).",
            "219 directories under `data/audio/<C0xxx instrument id>/`; the playing technique is embedded in the file names — technique labels must be parsed from file names, there is no separate CSV.",
        ],
        extra="Extremely imbalanced (219 classes / 4,956 recordings ≈ 22 per class), the most natural test bed for long-tail and few-shot methods.",
    ),
    dict(
        key="timbre_range", zh="Timbre and vocal range", task="timbre similarity / range-span analysis (multi-label)",
        docs="775 clips, 630 MB, 9 singers; the least documented subset.",
        page="https://doi.org/10.5281/zenodo.5676893",
        origin="CCMusic multi-purpose library (Zenodo 10.5281/zenodo.5676893), released with the CCMusic main reference.",
        intro=[
            "Locally this is two subdirectories: `data/timbre/audios/` 672 + `data/range/audios/` 726 = 1,398 clips, more than the 775 documented.",
            "`timbre.csv` has only 32 rows and `range.csv` 729 (label ∈ {0,1,2}) — label coverage is incomplete, so they serve only as weak labels.",
        ],
        extra="Mixed sample rates (16/44.1/48 kHz), and the `audios` directories also hold jpgs of the same name, so traversal must filter by extension.",
    ),
]

CITENOTE = {
    "pianos": "The formally published version is Springer LNEE 2024, pp. 3–17, DOI 10.1007/978-981-97-7962-8_1; the official README still lists the 2023 conference version, so cite the online publication record.",
    "music_genre": "This subset has no standalone paper; cite the CCMUSIC Zenodo dataset entry plus the CCMusic main reference in section 3.",
    "chest_falsetto": "This subset has no standalone paper; cite the CCMUSIC Zenodo dataset entry plus the CCMusic main reference in section 3.",
    "song_structure": "This subset has no standalone paper; cite the CCMUSIC Zenodo dataset entry plus the CCMusic main reference in section 3.",
    "timbre_range": "This subset has no standalone paper; cite the CCMUSIC Zenodo dataset entry plus the CCMusic main reference in section 3.",
    "bel_canto": "The official README cites the CCMusic main reference directly and there is no subset paper — the text must make the attribution of the integration work explicit.",
    "instrument_timbre": "The official BibTeX lacks volume/issue and DOI; the full publication record is Applied Sciences 10(3): 789, DOI 10.3390/app10030789.",
    "acapella": "The official BibTeX lacks volume/issue and DOI; the full publication record is Applied Sciences 12(19): 9931, DOI 10.3390/app12199931, by Rongfeng Li & Mingtong Zhang. Audio not downloaded locally.",
}

STATS = json.load(io.open(os.path.join(ROOT, "data", "actual_stats.json"), encoding="utf-8"))


# ---------------------------------------------------------------- helpers
def read_readme(key):
    p = os.path.join(README_DIR, key + ".readme.md")
    if not os.path.exists(p):
        return ""
    return io.open(p, encoding="utf-8", errors="replace").read()


def bibtex_of(key):
    """Return the first bibtex block after the Cite/citation heading of the official README."""
    txt = read_readme(key)
    if not txt:
        return ""
    heads = [m.start() for m in re.finditer(r"(?im)^#{1,4}.*(cite|引用|citation)", txt)]
    blocks = [(m.start(), m.group(1).strip()) for m in re.finditer(r"```bibtex\s*(.*?)```", txt, re.S)]
    if not blocks:
        return ""
    if heads:
        after = [b for b in blocks if b[0] > heads[0]]
        if after:
            return after[0][1]
    return blocks[0][1]


def tree_lines(key, max_depth=2):
    base = os.path.join(DATA, key, "data")
    if not os.path.isdir(base):
        return ["<not downloaded>"]
    out = []

    def walk(d, depth, prefix=""):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            return
        dirs = [n for n in names if os.path.isdir(os.path.join(d, n))]
        files = [n for n in names if os.path.isfile(os.path.join(d, n))]
        keep_dirs = dirs[:4]
        more = len(dirs) > 4
        for i, n in enumerate(keep_dirs):
            last = (i == len(keep_dirs) - 1) and not files and not more
            out.append(f"{prefix}{'└─ ' if last else '├─ '}{n}/")
            if depth + 1 < max_depth:
                walk(os.path.join(d, n), depth + 1, prefix + ("   " if last else "│  "))
        if more:
            out.append(f"{prefix}\u2514\u2500 ... ({len(dirs) - 4:,} more directories of this kind)")
        moref = len(files) > 3
        for j, n in enumerate(files[:3]):
            last = j == min(len(files), 3) - 1 and not dirs and not moref
            out.append(f"{prefix}{'└─ ' if last else '├─ '}{n}")
        if moref:
            out.append(f"{prefix}\u2514\u2500 ... ({len(files) - 3:,} more files)")

    walk(base, 0)
    return out


def count_files(d):
    n = 0
    for _, _, fs in os.walk(d):
        n += len(fs)
    return n


def human(n):
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024.0


def du(d):
    tot = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                tot += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return tot


# ---------------------------------------------------------------- main flow
def main():
    rows, body = [], []
    for i, d in enumerate(DS, 1):
        key = d["key"]
        st = STATS.get(key, {})
        bits = []
        if st.get("audio_files"):
            bits.append(f"audio {st['audio_files']:,} files / {st['duration_h']:.2f} h")
        else:
            bits.append("audio 0 files (feature images only locally)")
        if os.path.isdir(os.path.join(DATA, key)):
            nimg = sum(1 for dp, _, fs in os.walk(os.path.join(DATA, key))
                       for f in fs if f.lower().endswith((".jpg", ".png", ".jpeg")))
            if nimg:
                bits.append(f"{nimg:,} images")
        base = os.path.join(DATA, key)
        vol = human(du(base)) if os.path.isdir(base) else "—"
        rows.append(f"| {i} | `{key}` | {d['zh']} | {d['task']} | {vol} |")

        lines = [f"### {i}. `{key}` — {d['zh']}", ""]
        lines += [f"- **Task**: {d['task']}", f"- **Contents**: {d['docs']}"]
        lines += [f"- **Local measurement**: {'; '.join(bits)}; {count_files(base):,} files in total, {vol}"
                  if os.path.isdir(base) else "- **Local measurement**: not downloaded"]
        lines.append("")
        for t in d["intro"]:
            lines.append(t)
            lines.append("")
        if d.get("extra"):
            lines += [f"> {d['extra']}", ""]
        if os.path.isdir(os.path.join(DATA, key)):
            lines += ["```", f"{key}/data/"] + tree_lines(key) + ["```", ""]
        lines += [f"- **Original source**: <{d['page']}> — {d['origin']}",
                  f"- **Integrated release**: [ModelScope `ccmusic-database/{key}`](https://www.modelscope.cn/datasets/ccmusic-database/{key})",
                  f"- **Local fetch**: `modelscope download --dataset ccmusic-database/{key} --local-dir D:\\VSCode\\Datasets\\{key} --cache-dir D:\\VSCode\\Datasets\\.cache`", ""]
        bib = bibtex_of(key)
        if bib:
            lines += ["- **Citation (verbatim from the official README)**: ", "", "```bibtex", bib.strip(), "```", ""]
            if CITENOTE.get(key):
                lines += [f"> {CITENOTE[key]}", ""]
        else:
            lines += ["- **Citation**: the official README gives no standalone BibTeX; cite the CCMusic main reference (see section 3 below).", ""]
        body.append("\n".join(lines))

    head = f"""# CCMusic datasets (local copy, 13)

> Source: ModelScope organisation [`ccmusic-database`]({MS_ORG}) (CCMusic multi-purpose music database; China Conservatory of Music / Fudan University FD-LAMT / Zhejiang University NEXT, per the CCMusic paper's affiliations).
> Local path: `D:\\VSCode\\Datasets\\<dataset_id>\\data\\...`, keeping the ModelScope repo-relative layout so the official loaders apply unchanged.
> Download and extraction verified: 2026-11-02. Licence: **CC BY-NC-ND 4.0** (attribution, non-commercial, no derivatives), see section 3.

## 0. Directory conventions

```
D:\\VSCode\\Datasets\\
\u251c\u2500 README.md          this file
\u251c\u2500 _manifest.json     download manifest: 26 files, size, first 16 sha256 chars, engine used
\u251c\u2500 _extract.json      extraction reconciliation: entry count / byte count for 21 zips
\u251c\u2500 .cache\\            modelscope cache (safe to delete wholesale)
\u2514\u2500 <dataset_id>\\data\\ per-dataset roots
```

Four content kinds, different meanings, not to be conflated:

| Directory | What it is | Naming |
| :-- | :-- | :-- |
| `data/audio/` | raw audio | wav / mp3 / flac, sample rates not unified (16–96 kHz) |
| `data/eval/` | **pre-computed feature images**, not audio slice files | mostly 496×369 JPEG (476×349 for timbre_range), in three sets: `chroma/`, `cqt/`, `mel/` |
| `data/mel/` | full-recording mel strips | height 128, width follows duration (417–7,853) |
| `*.csv` / `label/` | labels | only 5 datasets have them; CNPM is GBK-encoded |

So the slice count = image count in each `data/eval/` feature directory ÷ 3, and is not the audio count. All 27 zips have been deleted, after checking entry and byte counts one by one (recorded in `_extract.json`).

## 1. Overview

| # | Dataset | Full name | Task | Local size |
| :-: | :-- | :-- | :-- | --: |
{chr(10).join(rows)}

**Three with no local audio**: `bel_canto`, `chest_falsetto` and `music_genre` hold only feature images and 0 audio files; fetch commands are in their sections.

## 2. Per dataset

"""
    tail = """
## 3. Unified citation and licence

**CCMusic main reference for the integrated library** (the integration, slicing and eval splits of all 13 datasets come from this work; cite it whenever citing a dataset):

```bibtex
@article{Zhou-2025,
  author  = {Monan Zhou and Shenyang Xu and Zhaorui Liu and Zhaowen Wang and Feng Yu and Wei Li and Baoqiang Han},
  title   = {CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research},
  journal = {Transactions of the International Society for Music Information Retrieval},
  year    = {2025},
  volume  = {8},
  number  = {1},
  pages   = {22--38},
  url     = {https://doi.org/10.5334/tismir.194},
  doi     = {10.5334/tismir.194}
}
```

**CCMUSIC Zenodo dataset** (music_genre / chest_falsetto / song_structure / timbre_range):

```bibtex
@dataset{zhaorui_liu_2021_5676893,
  author    = {Zhaorui Liu and Zijin Li},
  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},
  month     = {nov},
  year      = {2021},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.5676893},
  url       = {https://doi.org/10.5281/zenodo.5676893}
}
```

- Where a dataset has a standalone paper, its BibTeX is given verbatim in that section (source: ModelScope official README).
- The integrated release (slices, feature images, eval splits) was produced by the CCMusic team; **raw collection** attribution is in each section's "Original source". Cite both levels when re-citing, otherwise the integration work is credited to the original collectors.
- Licence **CC BY-NC-ND 4.0**: copying and redistribution are allowed with attribution, but **no commercial use and no distribution of derivative works**. Model weights, features and statistics trained on these data count as derivative works, so check compliance before public release; internal academic use is unaffected.

## 4. Usage notes (all learned from measurement)

1. **File encoding**: `CNPM/data/label.csv` is GBK; the header `样式/Pattern` does not decode as UTF-8 and comes out as replacement characters; use `pandas.read_csv(..., encoding="gbk")`.
2. **Duration is double-counted**: the `11/4/7_classes` directories of `erhu_playing_tech` are different labellings of the same recordings; summing the directories gives 0.83 h, distinct recordings only ≈ 25.8 min.
3. **Sample rates differ**: CTIS mixes 44.1/48/96 kHz, timbre_range goes down to 16 kHz, pianos mixes 44.1/48 kHz. Any batched preprocessing must resample to one rate first.
4. **Label sources are not uniform**: only 5 datasets ship CSVs (CNPM 1, Guzheng_Tech99 99 frame-level, instrument_timbre 2, timbre_range 2); the rest are parsed from directory or file names (CTIS techniques are written in the file names).
5. **Where the official documentation disagrees with the files**: `timbre_range` documents 775 clips / 9 singers but holds 1,398 and `timbre.csv` has only 32 rows; 12 of `instrument_timbre`'s 16 card descriptors differ from the CSV column names; `pianos` documents 2023 CSMT but was published in LNEE 2024. Trust the files and the official README.
6. **Do not treat the image count under `data/eval` as an independent sample count**: slices of the same recording share content, so a slice-level random split leaves 97-100% of test samples with a same-source sibling in training (see the leakage audit in the parent project).
"""
    txt = head + "\n".join(body) + tail
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(txt)
    print("wrote", OUT, len(txt), "chars")


if __name__ == "__main__":
    main()
