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
        "task": "钢琴品牌/型号分类、音高检测、音质主观评分回归",
        "task_en": "piano brand classification, pitch detection, quality-score regression",
        "labels": "7 类品牌（默认）/8 类（含 Yamaha）；88/52 音高；bass/mid/treble/avg 四个连续评分",
        "n_src": 668, "n_src_raw": 1332, "n_slice": 18352,
        "duration_s": 3247.94, "sr": 44100,
        "slice_len": "0.18 s (eval) / 全曲（raw）",
        "modality": "wav + mel(.jpg)",
        "splits": "默认 580（8:1:1，按类）；eval 18,352 片（按类 8:1:1）",
        "split_level": "sclice",
        "license": "CC-BY-NC-ND",
        "downloads": 3627, "likes": 18,
        "repo_bytes": 645893739,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou1",
        "paper": "Zhou et al., A Holistic Evaluation of Piano Sound Quality, CSMT 2023 (arXiv:2310.04722)",
        "cite": "@inproceedings{zhou2023holistic,\n  title     = {A Holistic Evaluation of Piano Sound Quality},\n  author    = {Monan Zhou and Shangda Wu and Shaohua Ji and Zijin Li and Wei Li},\n  booktitle = {National Conference on Sound and Music Technology},\n  pages     = {3--17},\n  year      = {2023},\n  organization = {Springer}\n}",
        "baseline": None, "baseline_note": "模型卡仅给曲线图，未给数值",
        "updated": "2026-10",
    },
    {
        "name": "music_genre",
        "cn": "音乐流派数据集",
        "en": "Music Genre Dataset",
        "task": "三层级流派分类（2/9/16 类）",
        "task_en": "hierarchical genre classification",
        "labels": "一级 2 类（古典/非古典）；二级 9 类；三级 16 类",
        "n_src": 1713, "n_src_raw": 1713, "n_slice": 36375,
        "duration_s": None, "sr": 22050,
        "slice_len": "11.4 s (eval)",
        "modality": "mp3 + mel/cqt/chroma(.jpg)",
        "splits": "8:1:1（29,100 / 3,637 / 3,638）",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 13745, "likes": 20,
        "repo_bytes": 18309144980,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html",
        "paper": "Liu & Li, CCMUSIC DATASET, Zenodo 2021 (10.5281/zenodo.5676893)",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  month     = {nov},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": None, "baseline_note": "模型卡仅给曲线图；论文另有表",
        "updated": "2026-10",
    },
    {
        "name": "chest_falsetto",
        "cn": "真假声数据集",
        "en": "Chest Voice and Falsetto Dataset",
        "task": "真假声 + 性别四分类（m/f × chest/falsetto）",
        "task_en": "chest/falsetto + gender classification",
        "labels": "4 类（m_chest=0, f_chest=1, m_falsetto=2, f_falsetto=3）+ gender 2 类",
        "n_src": 1280, "n_src_raw": 1280, "n_slice": 1280,
        "duration_s": 640.05, "sr": 22050,
        "slice_len": "0.25–0.5 s",
        "modality": "wav + mel/cqt/chroma(.jpg) + 5120 个特征 csv",
        "splits": "默认 767/256/257（6:2:2，按类）",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 5100, "likes": 15,
        "repo_bytes": 122578898,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou3",
        "paper": "Zenodo 10.5281/zenodo.5676893（原始）；模型卡为集成版评测",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": 0.994, "baseline_note": "AlexNet + mel；均值 0.918；CQT 均值仅 0.610",
        "updated": "2026-10",
    },
    {
        "name": "acapella",
        "cn": "歌唱干声评价数据集",
        "en": "Acapella Evaluation Dataset",
        "task": "唱功九维评分回归 / 歌手排序",
        "task_en": "singing performance scoring (9 dimensions)",
        "labels": "pitch, rhythm, vocal_range, timbre, pronunciation, vibrato, dynamic, breath_control, overall_performance（10 分制）",
        "n_src": 132, "n_src_raw": 132, "n_slice": 132,
        "duration_s": 12493.83, "sr": None,
        "slice_len": "整段（主歌+副歌）",
        "modality": "wav + mel(.jpg) + 评分 xls/csv",
        "splits": "按歌曲 6 个 split（song1–song6），每首 22 位歌手",
        "split_level": "clip",
        "license": "CC-BY-NC-ND",
        "downloads": 4408, "likes": 18,
        "repo_bytes": 1385304433,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou2",
        "paper": "Li & Zhang, Singing-Voice Timbre Evaluations Based on Transfer Learning, Applied Sciences 2022",
        "cite": "@article{Li2022SingingVoiceTE,\n  title   = {Singing-Voice Timbre Evaluations Based on Transfer Learning},\n  author  = {Rongfeng Li and Mingtong Zhang},\n  journal = {Applied Sciences},\n  year    = {2022},\n  doi     = {10.3390/app12199931}\n}",
        "baseline": None, "baseline_note": "回归任务，无统一分类基线",
        "updated": "2026-10",
    },
    {
        "name": "CNPM",
        "cn": "中国民族五声调式数据集",
        "en": "Chinese National Pentatonic Mode Dataset",
        "task": "调式四轴标注：同宫系统(12)/主音(12)/样式(5)/种类(6)",
        "task_en": "pentatonic mode recognition (multi-label, 4 axes)",
        "labels": "system 12、tonic 12、pattern 5（宫商角徵羽）、type 6（五声/六声清角/六声变宫/七声雅乐/七声清乐/七声燕乐）",
        "n_src": 287, "n_src_raw": 287, "n_slice": 2724,
        "duration_s": 51517.97, "sr": 44100,
        "slice_len": "20 s（零填充）",
        "modality": "wav + mel/cqt/chroma(.jpg) + label.csv(GBK)",
        "splits": "仅 train（多标签）；eval 2,724 片按类 10/10/80",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 14587, "likes": None,
        "repo_bytes": 3205207050,
        "source_url": "https://ccmusic-database.github.io/database/csmtd.html#shou10",
        "paper": "Wang et al., Automatic Chinese National Pentatonic Modes Recognition Using CNN, ISMIR 2022",
        "cite": "@inproceedings{WangCYMLX022,\n  author    = {Zhaowen Wang and Mingjin Che and Yue Yang and Wenwu Meng and Qinyu Li and Fan Xia and Wei Li},\n  title     = {Automatic Chinese National Pentatonic Modes Recognition Using Convolutional Neural Network},\n  booktitle = {ISMIR 2022},\n  pages     = {345--352},\n  year      = {2022}\n}",
        "baseline": 0.859, "baseline_note": "vit_l_16 + CQT；vgg11_bn mel 0.807；chroma 最高仅 0.609",
        "updated": "2026-10",
    },
    {
        "name": "Guzheng_Tech99",
        "cn": "古筝真实演奏数据集",
        "en": "Guzheng Tech99 Dataset",
        "task": "逐帧多标签演奏技法检测（7 类）",
        "task_en": "frame-level multi-label playing technique detection",
        "labels": "chanyin/boxian/shanghua/xiahua/glissando/yaozhi/dianyin；每音符 onset/offset/pitch/tech",
        "n_src": 99, "n_src_raw": 99, "n_slice": None,
        "duration_s": 9064.61, "sr": 44100,
        "slice_len": "3 s / 88x258 CQT",
        "modality": "wav + mel + 逐音符 csv",
        "splits": "按曲目 79/10/10（train/valid/test）",
        "split_level": "song",
        "license": "CC-BY-NC-ND",
        "downloads": 26306, "likes": None,
        "repo_bytes": 683293260,
        "source_url": "https://ccmusic-database.github.io/database/csmtd.html#Tech99",
        "paper": "Li et al., Frame-Level Multi-Label Playing Technique Detection..., ICASSP 2023",
        "cite": "@inproceedings{LiCMWYXL23,\n  author    = {Dichucheng Li and Mingjin Che and Wenwu Meng and Yulun Wu and Yi Yu and Fan Xia and Wei Li},\n  title     = {Frame-Level Multi-Label Playing Technique Detection Using Multi-Scale Network and Self-Attention Mechanism},\n  booktitle = {ICASSP 2023},\n  pages     = {1--5},\n  year      = {2023}\n}",
        "baseline": 0.862, "baseline_note": "VGG19 + mel；ConvNeXt-B CQT 0.849；均值 mel 0.788",
        "updated": "2026-10",
    },
    {
        "name": "instrument_timbre",
        "cn": "乐器音色打分数据集",
        "en": "Instruments Timbre Evaluation",
        "task": "16 维语义音色评分回归 / 中西乐器对比",
        "task_en": "16-dimensional semantic timbre regression",
        "labels": "16 个语义轴 0–10 分（slim/bright/dim/sharp/thick/thin/solid/clear/dry/plump/rough/pure/hoarse/harmonious/soft/turbid）",
        "n_src": 61, "n_src_raw": 61, "n_slice": 61,
        "duration_s": 643.86, "sr": 44100,
        "slice_len": "整段",
        "modality": "wav + mel + Chinese.csv / Western.csv",
        "splits": "Chinese 37 / Western 24（无 train/test 划分）",
        "split_level": "item",
        "license": "CC-BY-NC-ND",
        "downloads": 2099, "likes": None,
        "repo_bytes": 106683039,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou4",
        "paper": "Jiang et al., Analysis and Modeling of Timbre Perception Features in Musical Sounds, Applied Sciences 2020",
        "cite": "@article{Jiang2020AnalysisAM,\n  title   = {Analysis and Modeling of Timbre Perception Features in Musical Sounds},\n  author  = {Wei Jiang and Jingyu Liu and Xiaoyi Zhang and Shuang Wang and Yujian Jiang},\n  journal = {Applied Sciences},\n  year    = {2020},\n  doi     = {10.3390/app10030789}\n}",
        "baseline": None, "baseline_note": "小样本感知建模，无公开基线",
        "updated": "2026-10",
    },
    {
        "name": "song_structure",
        "cn": "歌曲结构标注数据集",
        "en": "Structure Annotation Dataset of Songs",
        "task": "段落边界检测 + 功能标签（9 类）",
        "task_en": "music structure boundary detection + functional labelling",
        "labels": "intro/re-intro/verse/chorus/pre-chorus/post-chorus/bridge/interlude/ending",
        "n_src": 300, "n_src_raw": 300, "n_slice": None,
        "duration_s": None, "sr": 22050,
        "slice_len": "整曲",
        "modality": "mp3 + 边界/标签 txt + MSA_dataset.zip",
        "splits": "仅 train（无官方划分）",
        "split_level": "song",
        "license": "CC-BY-NC-ND",
        "downloads": 9425, "likes": None,
        "repo_bytes": 4558604575,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html",
        "paper": "无专文；模型卡用 MSAF + mir_eval（Harmonix 协议）",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": None, "baseline_note": "仅给 MSAF 流程，未给数值",
        "updated": "2026-10",
    },
    {
        "name": "GZ_IsoTech",
        "cn": "古筝演奏技法数据集",
        "en": "Guzheng IsoTech Dataset",
        "task": "片段级演奏技法八分类",
        "task_en": "clip-level playing technique classification",
        "labels": "vibrato/upward_portamento/downward_portamento/returning_portamento/glissando/tremolo/harmonics/plucks",
        "n_src": 2824, "n_src_raw": 2824, "n_slice": 2899,
        "duration_s": 3838.68, "sr": 44100,
        "slice_len": "~1.36 s（原始片段）",
        "modality": "wav + mel + 标签列（意大利语/中文/拼音）",
        "splits": "沿用原始 4:1 train/test；eval 2,899 片",
        "split_level": "clip",
        "license": "CC-BY-NC-ND",
        "downloads": 8746, "likes": None,
        "repo_bytes": 523903564,
        "source_url": "https://ccmusic-database.github.io/database/csmtd.html#GZTech",
        "paper": "Li et al., Playing Technique Detection by Fusing Note Onset Information in Guzheng Performance, ISMIR 2022",
        "cite": "@inproceedings{LiWLZ0X022,\n  author    = {Dichucheng Li and Yulun Wu and Qinyu Li and Jiahao Zhao and Yi Yu and Fan Xia and Wei Li},\n  title     = {Playing Technique Detection by Fusing Note Onset Information in Guzheng Performance},\n  booktitle = {ISMIR 2022},\n  pages     = {314--320},\n  year      = {2022}\n}",
        "baseline": 0.855, "baseline_note": "vit_l_16 + mel；resnet101 CQT 0.798",
        "updated": "2026-10",
    },
    {
        "name": "bel_canto",
        "cn": "美声民族唱法数据集",
        "en": "Bel Canto and Chinese Folk Song Singing Dataset",
        "task": "唱法 × 性别四分类",
        "task_en": "singing style x gender classification",
        "labels": "m_bel=0, f_bel=1, m_folk=2, f_folk=3 + singing_method + gender",
        "n_src": 203, "n_src_raw": 203, "n_slice": 9910,
        "duration_s": 18270.48, "sr": 22050,
        "slice_len": "1.6 s（去静音后切分）",
        "modality": "wav + mel/cqt/chroma(.jpg)",
        "splits": "默认 159/21/23（8:1:1）；eval 7,926/990/994",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 9092, "likes": None,
        "repo_bytes": 3310156921,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou9",
        "paper": "Zhou et al., CCMusic: An Open and Diverse Database for Chinese MIR Research, TISMIR 2025",
        "cite": "@article{Zhou-2025,\n  author  = {Monan Zhou and Shenyang Xu and Zhaorui Liu and Zhaowen Wang and Feng Yu and Wei Li and Baoqiang Han},\n  title   = {CCMusic: An Open and Diverse Database for Chinese Music Information Retrieval Research},\n  journal = {Transactions of the International Society for Music Information Retrieval},\n  volume  = {8}, number = {1}, pages = {22--38}, year = {2025},\n  doi     = {10.5334/tismir.194}\n}",
        "baseline": 0.948, "baseline_note": "GoogleNet + mel；Swin-S CQT 0.936；均值 mel 0.921",
        "updated": "2026-10",
    },
    {
        "name": "erhu_playing_tech",
        "cn": "二胡演奏技法数据集",
        "en": "Erhu Playing Technique Dataset",
        "task": "二胡技法 11 分类（另有 7 类/4 类层级子集）",
        "task_en": "erhu playing technique classification (11/7/4-class)",
        "labels": "vibrato/trill/tremolo/staccato/ricochet/pizzicato/percussive/legato_slide_glissando/harmonic/diangong/detache",
        "n_src": 1253, "n_src_raw": 1253, "n_slice": 1253,
        "duration_s": 1548.36, "sr": 44100,
        "slice_len": "1.24 s 均值（0.09–4.47 s）",
        "modality": "wav + mel + 标签列",
        "splits": "6:2:2（按类）；4/7 类子集 635 条",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 6977, "likes": None,
        "repo_bytes": 357201858,
        "source_url": "https://ccmusic-database.github.io/database/ccm.html#shou8",
        "paper": "Wang et al., Musical Instrument Playing Technique Detection Based on FCN, arXiv:1910.09021 (2019, 未同行评审)",
        "cite": "@article{abs-1910-09021,\n  author  = {Zehao Wang and Jingru Li and Xiaoou Chen and Zijin Li and Shicheng Zhang and Baoqiang Han and Deshun Yang},\n  title   = {Musical Instrument Playing Technique Detection Based on FCN: Using Chinese Bowed-Stringed Instrument as an Example},\n  journal = {CoRR}, volume = {abs/1910.09021}, year = {2019}\n}",
        "baseline": 0.994, "baseline_note": "Swin-T / ConvNeXt-T + mel；均值 mel 0.976",
        "updated": "2026-10",
    },
    {
        "name": "CTIS",
        "cn": "中国传统乐器音响数据集",
        "en": "Chinese Traditional Instrument Sound Dataset",
        "task": "209 类民族乐器识别（219 标签）",
        "task_en": "Chinese traditional instrument recognition (219 labels)",
        "labels": "209 种乐器 / 219 个标签（含变体），含中文名与拼音",
        "n_src": 4956, "n_src_raw": 4956, "n_slice": 43054,
        "duration_s": 117482.75, "sr": 44100,
        "slice_len": "2 s（循环填充）",
        "modality": "wav + mel/cqt/chroma(.jpg)",
        "splits": "仅 train（默认）；eval 43,054 片按类 8:1:1",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 17259, "likes": 22,
        "repo_bytes": 10085025688,
        "source_url": "https://ccmusic-database.github.io/database/ctis.html",
        "paper": "Liang et al., Constructing a Multimedia Chinese Musical Instrument Database, CSMT 2019",
        "cite": "@inproceedings{10.1007/978-981-13-8707-4_5,\n  author    = {Xiaojing Liang and Zijin Li and Jingyu Liu and Wei Li and Jiaxing Zhu and Baoqiang Han},\n  title     = {Constructing a Multimedia Chinese Musical Instrument Database},\n  booktitle = {Proceedings of the 6th Conference on Sound and Music Technology (CSMT)},\n  pages     = {53--60}, publisher = {Springer}, year = {2019}\n}",
        "baseline": 0.980, "baseline_note": "regnet_y_32gf + CQT；swin_t mel 0.956；chroma 0.852",
        "updated": "2026-10",
    },
    {
        "name": "timbre_range",
        "cn": "音色音域数据集",
        "en": "Timbre and Range Dataset",
        "task": "音色/音域评分与半音阶片段（任务定义不明确）",
        "task_en": "timbre & vocal-range scoring (task under-specified)",
        "labels": "timbre.csv: 32 条 score1/score2/avg_score；range.csv: 729 条 id→0/1/2（音域档）",
        "n_src": 775, "n_src_raw": 775, "n_slice": 775,
        "duration_s": None, "sr": None,
        "slice_len": "片段/单音",
        "modality": "wav（timbre.zip 568 MB + range.zip 62 MB）+ 2 个 csv",
        "splits": "8:1:1 随机（按条）",
        "split_level": "slice",
        "license": "CC-BY-NC-ND",
        "downloads": 666, "likes": 11,
        "repo_bytes": 660745598,
        "source_url": "—（README 未给原始出处链接）",
        "paper": "无专文；模型卡不存在",
        "cite": "@dataset{zhaorui_liu_2021_5676893,\n  author    = {Zhaorui Liu and Zijin Li},\n  title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},\n  year      = {2021},\n  publisher = {Zenodo},\n  version   = {1.1},\n  doi       = {10.5281/zenodo.5676893}\n}",
        "baseline": None, "baseline_note": "无基线、无评测",
        "updated": "2026-10",
    },
]

# ------------------------------------------------------------ quality rubric
DIMS = [
    ("scale", "规模与覆盖", 0.15),
    ("labels", "标注可信度", 0.20),
    ("integrity", "数据完整可用", 0.15),
    ("protocol", "评测协议可靠性", 0.20),
    ("docs", "文档与可复现", 0.10),
    ("legal", "复用与合规", 0.10),
    ("ecosystem", "生态与基线", 0.10),
]

# 1-5 主观评级，依据见 SCORES_EVIDENCE
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
    "pianos": "668 条 8 类、54 min 录音；29 人主观评分；0.18 s 切片来自 668 条录音 → 同录音跨 split；NC-ND",
    "music_genre": "1,713 首 16 类；站点风格标签无标注者；商用歌曲版权；36,375 片随机 8:1:1 → 强泄漏；下载量最高",
    "chest_falsetto": "1,280 单音；性别为后补标注；0.25 s 切片，CQT 频谱泄漏为卡片自述缺陷；0.994 精度不可信",
    "acapella": "132 段、4 位专业评委、9 维 10 分制；无 eval 子集需求；但仅 6 首歌曲、需 LOSO",
    "CNPM": "287 首、858 min、四轴标注、ISMIR 2022 背书；label.csv 为 GBK；eval 2,724 片按类切分",
    "Guzheng_Tech99": "99 首录音棚独奏、逐音符 63,352 标签、7 技法、按曲目 79/10/10；唯一 song-level 划分",
    "instrument_timbre": "61 件乐器 ×16 语义轴，主观实验数据；卡内术语表与 CSV 列名不一致；无划分",
    "song_structure": "300 首、9 类功能标签、音乐学院教师标注；README 仅 1.3k 字符、无 schema 表、无划分",
    "GZ_IsoTech": "2,824 片段中 2,328 来自虚拟音源；沿用 4:1；无演奏者/音源分组",
    "bel_canto": "203 段专业录音 5.08 h；标签人工补全；eval 9,910 片来自 203 段 → 强泄漏",
    "erhu_playing_tech": "1,253 片段 26 min；原始论文未同行评审且卡片自述不可复现；6:2:2 切片划分，0.994",
    "CTIS": "209 类/219 标签/32.6 h，全消声室 32 支 B&K 麦克风；每乐器单次录制；2 s 切片泄漏，0.980",
    "timbre_range": "775 片段但 timbre.csv 仅 32 条、README 703 字符、无基线无评测、无原始出处",
}

# 泄漏实测：由 code/ccmusic_zipindex.py + code/ccmusic_splits.py 在 eval/mel 包上直接算出
# (slices, sources, K, leak_rate%。leak_rate=复现官方划分后，测试切片中同源兄弟也落在训练集的比例)
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
