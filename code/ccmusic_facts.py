"""Single source of truth for the CCMusic (ModelScope) 13-dataset review.

Facts come from:
  * ModelScope dataset detail API  (/api/v1/datasets/ccmusic-database/{name})
  * ModelScope repo tree API      (/repo/tree?Revision=master&Recursive=True)
  * ModelScope model cards        (/api/v1/models/ccmusic-database/{name})
  * CCMusic project site          (ccmusic-database.github.io/database/*.html)
  * The datasets' own data/*.py and data/*.csv files

Every numeric field below is traceable to one of those sources. Fields that the
sources do not state are marked None instead of being guessed.
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "ccmusic")

# ---------------------------------------------------------------- dataset facts
DATASETS = [
    {
        "name": "pianos",
        "cn": "钢琴音质数据集",
        "en": "Piano Sound Quality Dataset",
        "task": "piano brand/model classification, pitch detection, subjective sound-quality score regression",
        "task_en": "piano brand classification, pitch detection, quality-score regression",
        "labels": "7 brand classes (default) / 8 (Yamaha included); 88/52 pitches; four continuous scores bass/mid/treble/avg",
        "n_src": 668, "n_src_raw": 1332, "n_slice": 18352,
        "duration_s": 3247.94, "sr": 44100,
        "slice_len": "0.18 s (eval) / whole piece (raw)",
        "modality": "wav + mel(.jpg)",
        "splits": "default 580 (8:1:1, by class); eval 18,352 slices (8:1:1 by class)",
        "split_level": "sclice",
        "license": "CC-BY-NC-ND",
        "downloads": 3627, "likes": 18,
        "repo_bytes": 645893739,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou1",
        "paper": "Zhou et al., A Holistic Evaluation of Piano Sound Quality, CSMT 2023 (arXiv:2310.04722)",
        "cite": "@inproceedings{zhou2023holistic,\n  title     = {A Holistic Evaluation of Piano Sound Quality},\n  author    = {Monan Zhou and Shangda Wu and Shaohua Ji and Zijin Li and Wei Li},\n  booktitle = {National Conference on Sound and Music Technology},\n  pages     = {3--17},\n  year      = {2023},\n  organization = {Springer}\n}",
        "baseline": None, "baseline_note": "the model card gives only curves, no numbers",
        "updated": "2026-10",
    },
    {
        "name": "music_genre",
        "cn": "音乐流派数据集",
        "en": "Music Genre Dataset",
        "task": "three-tier genre classification (2/9/16 classes)",
        "task_en": "hierarchical genre classification",
        "labels": "tier 1: 2 classes (classical / non-classical); tier 2: 9 classes; tier 3: 16 classes",
        "n_src": 1713, "n_src_raw": 1713, "n_slice": 36375,
        "duration_s": None, "sr": 22050,
        "slice_len": "11.4 s (eval)",
        "modality": "mp3 + mel/cqt/chroma(.jpg)",
        "splits": "8:1:1 (29,100 / 3,637 / 3,638)",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 13745, "likes": 20,
        "repo_bytes": 18309144980,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html",
        "paper": "Liu & Li, CCMUSIC DATASET, Zenodo 2021 (10.5281/zenodo.5676893)",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  month     = {nov},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": None, "baseline_note": "the model card gives only curves; the paper has a separate table",
        "updated": "2026-10",
    },
    {
        "name": "chest_falsetto",
        "cn": "真假声数据集",
        "en": "Chest Voice and Falsetto Dataset",
        "task": "falsetto vs chest voice plus gender, 4 classes (m/f × chest/falsetto)",
        "task_en": "chest/falsetto + gender classification",
        "labels": "4 classes (m_chest=0, f_chest=1, m_falsetto=2, f_falsetto=3) + 2 gender classes",
        "n_src": 1280, "n_src_raw": 1280, "n_slice": 1280,
        "duration_s": 640.05, "sr": 22050,
        "slice_len": "0.25–0.5 s",
        "modality": "wav + mel/cqt/chroma(.jpg) + 5120 feature csv files",
        "splits": "default 767/256/257 (6:2:2, by class)",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 5100, "likes": 15,
        "repo_bytes": 122578898,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou3",
        "paper": "Zenodo 10.5281/zenodo.5676893 (original); the model card evaluates the integrated version",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": 0.994, "baseline_note": "AlexNet + mel; mean 0.918; CQT mean only 0.610",
        "updated": "2026-10",
    },
    {
        "name": "acapella",
        "cn": "歌唱干声评价数据集",
        "en": "Acapella Evaluation Dataset",
        "task": "nine-dimension singing-skill score regression / singer ranking",
        "task_en": "singing performance scoring (9 dimensions)",
        "labels": "pitch, rhythm, vocal_range, timbre, pronunciation, vibrato, dynamic, breath_control, overall_performance (10-point scale)",
        "n_src": 132, "n_src_raw": 132, "n_slice": 132,
        "duration_s": 12493.83, "sr": None,
        "slice_len": "whole segment (verse + chorus)",
        "modality": "wav + mel(.jpg) + score xls/csv",
        "splits": "6 splits by song (song1–song6), 22 singers each",
        "split_level": "clip",
        "license": "CC-BY-NC-ND",
        "downloads": 4408, "likes": 18,
        "repo_bytes": 1385304433,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou2",
        "paper": "Li & Zhang, Singing-Voice Timbre Evaluations Based on Transfer Learning, Applied Sciences 2022",
        "cite": "@article{Li2022SingingVoiceTE,\n  title   = {Singing-Voice Timbre Evaluations Based on Transfer Learning},\n  author  = {Rongfeng Li and Mingtong Zhang},\n  journal = {Applied Sciences},\n  year    = {2022},\n  doi     = {10.3390/app12199931}\n}",
        "baseline": None, "baseline_note": "regression task, no unified classification baseline",
        "updated": "2026-10",
    },
    {
        "name": "CNPM",
        "cn": "中国民族五声调式数据集",
        "en": "Chinese National Pentatonic Mode Dataset",
        "task": "four-axis mode annotation: gong system (12) / tonic (12) / pattern (5) / type (6)",
        "task_en": "pentatonic mode recognition (multi-label, 4 axes)",
        "labels": "system 12, tonic 12, pattern 5 (Gong/Shang/Jue/Zhi/Yu), type 6 (five-note / six-note with qingjiao / six-note with biangong / seven-note Yayue / seven-note Qingle / seven-note Yanyue)",
        "n_src": 287, "n_src_raw": 287, "n_slice": 2724,
        "duration_s": 51517.97, "sr": 44100,
        "slice_len": "20 s (zero-padded)",
        "modality": "wav + mel/cqt/chroma(.jpg) + label.csv(GBK)",
        "splits": "train only (multi-label); eval 2,724 slices split 10/10/80 by class",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 14587, "likes": None,
        "repo_bytes": 3205207050,
        "source_url": "https://ccmusic-database.github.io/database/csmtd.html#shou10",
        "paper": "Wang et al., Automatic Chinese National Pentatonic Modes Recognition Using CNN, ISMIR 2022",
        "cite": "@inproceedings{WangCYMLX022,\n  author    = {Zhaowen Wang and Mingjin Che and Yue Yang and Wenwu Meng and Qinyu Li and Fan Xia and Wei Li},\n  title     = {Automatic Chinese National Pentatonic Modes Recognition Using Convolutional Neural Network},\n  booktitle = {ISMIR 2022},\n  pages     = {345--352},\n  year      = {2022}\n}",
        "baseline": 0.859, "baseline_note": "vit_l_16 + CQT; vgg11_bn mel 0.807; chroma at best 0.609",
        "updated": "2026-10",
    },
    {
        "name": "Guzheng_Tech99",
        "cn": "古筝真实演奏数据集",
        "en": "Guzheng Tech99 Dataset",
        "task": "frame-level multi-label playing-technique detection (7 classes)",
        "task_en": "frame-level multi-label playing technique detection",
        "labels": "chanyin/boxian/shanghua/xiahua/glissando/yaozhi/dianyin; per-note onset/offset/pitch/tech",
        "n_src": 99, "n_src_raw": 99, "n_slice": None,
        "duration_s": 9064.61, "sr": 44100,
        "slice_len": "3 s / 88x258 CQT",
        "modality": "wav + mel + per-note csv",
        "splits": "79/10/10 by piece (train/valid/test)",
        "split_level": "song",
        "license": "CC-BY-NC-ND",
        "downloads": 26306, "likes": None,
        "repo_bytes": 683293260,
        "source_url": "https://ccmusic-database.github.io/database/csmtd.html#Tech99",
        "paper": "Li et al., Frame-Level Multi-Label Playing Technique Detection..., ICASSP 2023",
        "cite": "@inproceedings{LiCMWYXL23,\n  author    = {Dichucheng Li and Mingjin Che and Wenwu Meng and Yulun Wu and Yi Yu and Fan Xia and Wei Li},\n  title     = {Frame-Level Multi-Label Playing Technique Detection Using Multi-Scale Network and Self-Attention Mechanism},\n  booktitle = {ICASSP 2023},\n  pages     = {1--5},\n  year      = {2023}\n}",
        "baseline": 0.862, "baseline_note": "VGG19 + mel; ConvNeXt-B CQT 0.849; mel mean 0.788",
        "updated": "2026-10",
    },
    {
        "name": "instrument_timbre",
        "cn": "乐器音色打分数据集",
        "en": "Instruments Timbre Evaluation",
        "task": "16-dimension semantic timbre-score regression / Chinese vs Western instrument comparison",
        "task_en": "16-dimensional semantic timbre regression",
        "labels": "16 semantic axes, 0–10 (slim/bright/dim/sharp/thick/thin/solid/clear/dry/plump/rough/pure/hoarse/harmonious/soft/turbid)",
        "n_src": 61, "n_src_raw": 61, "n_slice": 61,
        "duration_s": 643.86, "sr": 44100,
        "slice_len": "whole segment",
        "modality": "wav + mel + Chinese.csv / Western.csv",
        "splits": "Chinese 37 / Western 24 (no train/test split)",
        "split_level": "item",
        "license": "CC-BY-NC-ND",
        "downloads": 2099, "likes": None,
        "repo_bytes": 106683039,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou4",
        "paper": "Jiang et al., Analysis and Modeling of Timbre Perception Features in Musical Sounds, Applied Sciences 2020",
        "cite": "@article{Jiang2020AnalysisAM,\n  title   = {Analysis and Modeling of Timbre Perception Features in Musical Sounds},\n  author  = {Wei Jiang and Jingyu Liu and Xiaoyi Zhang and Shuang Wang and Yujian Jiang},\n  journal = {Applied Sciences},\n  year    = {2020},\n  doi     = {10.3390/app10030789}\n}",
        "baseline": None, "baseline_note": "small-sample perceptual modelling, no public baseline",
        "updated": "2026-10",
    },
    {
        "name": "song_structure",
        "cn": "歌曲结构标注数据集",
        "en": "Structure Annotation Dataset of Songs",
        "task": "segment-boundary detection + functional labels (9 classes)",
        "task_en": "music structure boundary detection + functional labelling",
        "labels": "intro/re-intro/verse/chorus/pre-chorus/post-chorus/bridge/interlude/ending",
        "n_src": 300, "n_src_raw": 300, "n_slice": None,
        "duration_s": None, "sr": 22050,
        "slice_len": "whole track",
        "modality": "mp3 + boundary/label txt + MSA_dataset.zip",
        "splits": "train only (no official split)",
        "split_level": "song",
        "license": "CC-BY-NC-ND",
        "downloads": 9425, "likes": None,
        "repo_bytes": 4558604575,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html",
        "paper": "no dedicated paper; the model card uses MSAF + mir_eval (Harmonix protocol)",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": None, "baseline_note": "gives only the MSAF pipeline, no numbers",
        "updated": "2026-10",
    },
    {
        "name": "GZ_IsoTech",
        "cn": "古筝演奏技法数据集",
        "en": "Guzheng IsoTech Dataset",
        "task": "clip-level playing-technique classification, 8 classes",
        "task_en": "clip-level playing technique classification",
        "labels": "vibrato/upward_portamento/downward_portamento/returning_portamento/glissando/tremolo/harmonics/plucks",
        "n_src": 2824, "n_src_raw": 2824, "n_slice": 2899,
        "duration_s": 3838.68, "sr": 44100,
        "slice_len": "~1.36 s (original clips)",
        "modality": "wav + mel + label columns (Italian / Chinese / pinyin)",
        "splits": "keeps the original 4:1 train/test; eval 2,899 slices",
        "split_level": "clip",
        "license": "CC-BY-NC-ND",
        "downloads": 8746, "likes": None,
        "repo_bytes": 523903564,
        "source_url": "https://ccmusic-database.github.io/database/csmtd.html#GZTech",
        "paper": "Li et al., Playing Technique Detection by Fusing Note Onset Information in Guzheng Performance, ISMIR 2022",
        "cite": "@inproceedings{LiWLZ0X022,\n  author    = {Dichucheng Li and Yulun Wu and Qinyu Li and Jiahao Zhao and Yi Yu and Fan Xia and Wei Li},\n  title     = {Playing Technique Detection by Fusing Note Onset Information in Guzheng Performance},\n  booktitle = {ISMIR 2022},\n  pages     = {314--320},\n  year      = {2022}\n}",
        "baseline": 0.855, "baseline_note": "vit_l_16 + mel; resnet101 CQT 0.798",
        "updated": "2026-10",
    },
    {
        "name": "bel_canto",
        "cn": "美声民族唱法数据集",
        "en": "Bel Canto and Chinese Folk Song Singing Dataset",
        "task": "singing style × gender, 4 classes",
        "task_en": "singing style x gender classification",
        "labels": "m_bel=0, f_bel=1, m_folk=2, f_folk=3 + singing_method + gender",
        "n_src": 203, "n_src_raw": 203, "n_slice": 9910,
        "duration_s": 18270.48, "sr": 22050,
        "slice_len": "1.6 s (cut after silence removal)",
        "modality": "wav + mel/cqt/chroma(.jpg)",
        "splits": "default 159/21/23 (8:1:1); eval 7,926/990/994",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 9092, "likes": None,
        "repo_bytes": 3310156921,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou9",
        "paper": "Zhou et al., CCMusic: An Open and Diverse Database for Chinese MIR Research, TISMIR 2025",
        "cite": "@article{Zhou-2025,\n  author  = {Monan Zhou and Shenyang Xu and Zhaorui Liu and Zhaowen Wang and Feng Yu and Wei Li and Baoqiang Han},\n  title   = {CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research},\n  journal = {Transactions of the International Society for Music Information Retrieval},\n  volume  = {8}, number = {1}, pages = {22--38}, year = {2025},\n  doi     = {10.5334/tismir.194}\n}",
        "baseline": 0.948, "baseline_note": "GoogleNet + mel; Swin-S CQT 0.936; mel mean 0.921",
        "updated": "2026-10",
    },
    {
        "name": "erhu_playing_tech",
        "cn": "二胡演奏技法数据集",
        "en": "Erhu Playing Technique Dataset",
        "task": "11-class erhu technique classification (plus 7-class / 4-class hierarchical subsets)",
        "task_en": "erhu playing technique classification (11/7/4-class)",
        "labels": "vibrato/trill/tremolo/staccato/ricochet/pizzicato/percussive/legato_slide_glissando/harmonic/diangong/detache",
        "n_src": 1253, "n_src_raw": 1253, "n_slice": 1253,
        "duration_s": 1548.36, "sr": 44100,
        "slice_len": "1.24 s mean (0.09–4.47 s)",
        "modality": "wav + mel + label columns",
        "splits": "6:2:2 (by class); the 4/7-class subsets hold 635 clips",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 6977, "likes": None,
        "repo_bytes": 357201858,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou8",
        "paper": "Wang et al., Musical Instrument Playing Technique Detection Based on FCN, arXiv:1910.09021 (2019, not peer reviewed)",
        "cite": "@article{abs-1910-09021,\n  author  = {Zehao Wang and Jingru Li and Xiaoou Chen and Zijin Li and Shicheng Zhang and Baoqiang Han and Deshun Yang},\n  title   = {Musical Instrument Playing Technique Detection Based on FCN: Using Chinese Bowed-Stringed Instrument as an Example},\n  journal = {CoRR}, volume = {abs/1910.09021}, year = {2019}\n}",
        "baseline": 0.994, "baseline_note": "Swin-T / ConvNeXt-T + mel; mel mean 0.976",
        "updated": "2026-10",
    },
    {
        "name": "CTIS",
        "cn": "中国传统乐器音响数据集",
        "en": "Chinese Traditional Instrument Sound Dataset",
        "task": "209-class Chinese instrument identification (219 labels)",
        "task_en": "Chinese traditional instrument recognition (219 labels)",
        "labels": "209 instruments / 219 labels (variants included), with Chinese names and pinyin",
        "n_src": 4956, "n_src_raw": 4956, "n_slice": 43054,
        "duration_s": 117482.75, "sr": 44100,
        "slice_len": "2 s (loop-padded)",
        "modality": "wav + mel/cqt/chroma(.jpg)",
        "splits": "train only (default); eval 43,054 slices split 8:1:1 by class",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 17259, "likes": 22,
        "repo_bytes": 10085025688,
        "source_url": "https://ccmusic-database.github.io/database/ctis.html",
        "paper": "Liang et al., Constructing a Multimedia Chinese Musical Instrument Database, CSMT 2019",
        "cite": "@inproceedings{10.1007/978-981-13-8707-4_5,\n  author    = {Xiaojing Liang and Zijin Li and Jingyu Liu and Wei Li and Jiaxing Zhu and Baoqiang Han},\n  title     = {Constructing a Multimedia Chinese Musical Instrument Database},\n  booktitle = {Proceedings of the 6th Conference on Sound and Music Technology (CSMT)},\n  pages     = {53--60}, publisher = {Springer}, year = {2019}\n}",
        "baseline": 0.980, "baseline_note": "regnet_y_32gf + CQT; swin_t mel 0.956; chroma 0.852",
        "updated": "2026-10",
    },
    {
        "name": "timbre_range",
        "cn": "音色音域数据集",
        "en": "Timbre and Range Dataset",
        "task": "timbre/range scoring and chromatic-scale clips (task definition unclear)",
        "task_en": "timbre & vocal-range scoring (task under-specified)",
        "labels": "timbre.csv: 32 rows of score1/score2/avg_score; range.csv: 729 rows id→0/1/2 (range band)",
        "n_src": 775, "n_src_raw": 775, "n_slice": 775,
        "duration_s": None, "sr": None,
        "slice_len": "clip / single note",
        "modality": "wav (timbre.zip 568 MB + range.zip 62 MB) + 2 csv files",
        "splits": "random 8:1:1 (by item)",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 666, "likes": 11,
        "repo_bytes": 660745598,
        "source_url": "— (the README gives no source link)",
        "paper": "no dedicated paper; the model card does not exist",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": None, "baseline_note": "no baseline, no evaluation",
        "updated": "2026-10",
    },
]

# ------------------------------------------------------------ quality rubric
DIMS = [
    ("scale", "scale and coverage", 0.15),
    ("labels", "label trustworthiness", 0.20),
    ("integrity", "data integrity and availability", 0.15),
    ("protocol", "evaluation-protocol reliability", 0.20),
    ("docs", "documentation and reproducibility", 0.10),
    ("legal", "reuse and compliance", 0.10),
    ("ecosystem", "ecosystem and baselines", 0.10),
]

# subjective 1-5 ratings, evidence in SCORES_EVIDENCE
SCORES = {
    "pianos":          dict(scale=3, labels=4, integrity=4, protocol=2, docs=4, legal=2, ecosystem=3),
    "music_genre":     dict(scale=4, labels=3, integrity=3, protocol=2, docs=4, legal=1, ecosystem=5),
    "chest_falsetto":  dict(scale=3, labels=3, integrity=4, protocol=1, docs=4, legal=2, ecosystem=3),
    "acapella":        dict(scale=2, labels=5, integrity=4, protocol=4, docs=3, legal=2, ecosystem=3),
    "CNPM":            dict(scale=3, labels=4, integrity=4, protocol=3, docs=4, legal=2, ecosystem=4),
    "Guzheng_Tech99":  dict(scale=5, labels=5, integrity=4, protocol=5, docs=3, legal=2, ecosystem=5),
    "instrument_timbre": dict(scale=2, labels=5, integrity=4, protocol=3, docs=2, legal=2, ecosystem=2),
    "song_structure":  dict(scale=4, labels=5, integrity=3, protocol=4, docs=1, legal=1, ecosystem=3),
    "GZ_IsoTech":      dict(scale=4, labels=3, integrity=4, protocol=3, docs=4, legal=2, ecosystem=4),
    "bel_canto":       dict(scale=4, labels=4, integrity=4, protocol=2, docs=4, legal=2, ecosystem=4),
    "erhu_playing_tech": dict(scale=3, labels=3, integrity=4, protocol=1, docs=4, legal=2, ecosystem=3),
    "CTIS":            dict(scale=5, labels=4, integrity=3, protocol=2, docs=3, legal=2, ecosystem=4),
    "timbre_range":    dict(scale=2, labels=2, integrity=2, protocol=1, docs=1, legal=2, ecosystem=1),
}

SCORES_EVIDENCE = {
    "pianos": "668 recordings in 8 classes, 54 min; 29 raters; the 0.18 s slices come from those 668 recordings → same recording across splits; NC-ND",
    "music_genre": "1,713 tracks in 16 classes; site style labels with no annotators; commercial-song copyright; 36,375 slices in a random 8:1:1 → strong leakage; highest download count",
    "chest_falsetto": "1,280 single notes; gender labels added afterwards; 0.25 s slices, and the card itself names CQT spectral leakage as a defect; 0.994 accuracy not credible",
    "acapella": "132 clips, 4 professional judges, 9 dimensions on a 10-point scale; no eval subset needed; but only 6 songs, so LOSO is required",
    "CNPM": "287 tracks, 858 min, four-axis annotation, ISMIR 2022 backing; label.csv is GBK; eval 2,724 slices split by class",
    "Guzheng_Tech99": "99 studio solo recordings, 63,352 per-note labels, 7 techniques, 79/10/10 by piece; the only song-level split",
    "instrument_timbre": "61 instruments × 16 semantic axes, subjective-experiment data; the card glossary disagrees with the CSV column names; no split",
    "song_structure": "300 tracks, 9 functional labels, annotated by conservatory teachers; the README is only 1.3k characters, with no schema table and no split",
    "GZ_IsoTech": "2,328 of the 2,824 clips come from virtual instrument sources; keeps 4:1; no performer/source grouping",
    "bel_canto": "203 professional recordings, 5.08 h; labels completed by hand; 9,910 eval slices from those 203 recordings → strong leakage",
    "erhu_playing_tech": "1,253 clips, 26 min; the original paper is not peer reviewed and the card itself says it is not reproducible; slice-level 6:2:2, 0.994",
    "CTIS": "209 classes / 219 labels / 32.6 h, all in an anechoic chamber with 32 B&K microphones; one session per instrument; 2 s slices leak, 0.980",
    "timbre_range": "775 clips documented but timbre.csv holds only 32 rows, the README is 703 characters, no baseline, no evaluation, no source link",
}

# measured leakage: computed directly on the eval/mel packages by code/ccmusic_zipindex.py + code/ccmusic_splits.py
# (slices, sources, K, leak_rate %. leak_rate = share of test slices whose same-source siblings also fall in training after the official split is reproduced)
AUDIT = [
    ("bel_canto", 9910, 203, 48.82, 100.0),
    ("pianos", 18352, 668, 27.47, 100.0),
    ("music_genre", 36375, 1713, 21.23, 100.0),
    ("CNPM", 2724, 287, 9.49, 99.25),
    ("CTIS", 43054, 4956, 8.69, 97.58),
    ("GZ_IsoTech", 2899, 2824, 1.03, 0.0),
    ("chest_falsetto", 1280, 1280, 1.0, 0.0),
    ("erhu_playing_tech", 1253, 1253, 1.0, 0.0),
    ("Guzheng_Tech99", 99, 99, 1.0, 0.0),
]


def composite(name):
    s = SCORES[name]
    return sum(s[k] * w for k, _, w in DIMS)


def main():
    os.makedirs(os.path.join(ROOT, "results", "ccmusic"), exist_ok=True)
    rows = []
    for d in DATASETS:
        r = dict(d)
        r["scores"] = SCORES[d["name"]]
        r["composite"] = round(composite(d["name"]), 2)
        r["evidence"] = SCORES_EVIDENCE[d["name"]]
        rows.append(r)
    with open(os.path.join(OUT, "facts.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": rows, "dims": DIMS, "audit": AUDIT}, f, ensure_ascii=False, indent=1)

    with open(os.path.join(ROOT, "results", "ccmusic", "ccmusic_scores.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "cn_name"] + [k for k, _, _ in DIMS] + ["composite"])
        for r in rows:
            w.writerow([r["name"], r["cn"]] + [r["scores"][k] for k, _, _ in DIMS] + [r["composite"]])

    print(f"{'dataset':20s} {'composite':>9s}  " + " ".join(f"{k:>9s}" for k, _, _ in DIMS))
    for r in sorted(rows, key=lambda x: -x["composite"]):
        print(f"{r['name']:20s} {r['composite']:9.2f}  " + " ".join(f"{r['scores'][k]:9d}" for k, _, _ in DIMS))


if __name__ == "__main__":
    main()
