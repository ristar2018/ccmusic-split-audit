# -*- coding: utf-8 -*-
"""生成 D:/VSCode/Datasets/README.md

数据来源：本地实际文件（盘点）+ data/ccmusic/*.readme.md（官方说明与 BibTeX）
+ data/actual_stats.json（音频时长实测）。
BibTeX 一律从官方 README 原样摘出，不手工改写，避免引用信息失真。
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

# ---------------------------------------------------------------- 数据集定义
# key: ModelScope 仓库名
DS = [
    dict(
        key="pianos", zh="钢琴音质评价", task="品牌识别 / 音高识别 / 音质主观评分回归",
        docs="668 段钢琴录音，8 个品牌；每段按琴键切分，标注音高与 4 个维度的主观音质评分。",
        page=LIB + "/database/ccm.html#shou1",
        origin="CCMusic 多功能库（ccm.html#shou1），原始采集为品牌钢琴试音录音；论文见 Zhou et al. (2024)。",
        intro=[
            "面向「同一台琴、同一首曲子，不同品牌/型号听起来差在哪」的音质感知研究。",
            "默认子集按 8 个品牌分目录，音频为单键/短句片段；eval 子集把每段裁成 496×369 特征图。",
        ],
        extra="`data/mel/*/<id>.jpg` 为整段录音的 mel 长条（高 128、宽 482–854），是少数保留「整段上下文」的表示。",
    ),
    dict(
        key="music_genre", zh="音乐流派分类", task="流派分类（3 层：2 大类 / 9 类 / 16 细类）",
        docs="1,713 首，三层流派标签：classic vs non-classic → 9 类 → 16 细类。",
        page="https://doi.org/10.5281/zenodo.5676893",
        origin="CCMusic 多功能库（ccm.html，Zenodo 10.5281/zenodo.5676893）；无独立论文，随 CCMusic 主文献发布。",
        intro=[
            "特征图 109,125 张（仅次于 CTIS），下载量在 13 个数据集里最高（13,745 次）。",
            "三层标签使其可做层次分类与「细粒度错误是否集中在相邻类」的分析。",
            "本地只有 eval 特征图，没有原始音频——做音频域方法（增广、重采样、SSL 前端）需另下 `audio.zip`。",
        ],
        extra="层三并非每类都有：如 `11_rock` 下还有 3 个细类目录，而 `3_symphony` 直接就是切片图。",
    ),
    dict(
        key="chest_falsetto", zh="真假声识别", task="4 类唱法分类（男/女 × 胸声/假声）",
        docs="1,280 段，10.7 分钟，4 类。",
        page=LIB + "/database/ccm.html#shou3",
        origin="CCMusic 多功能库（ccm.html#shou3），随 Zenodo 10.5281/zenodo.5676893 发布。",
        intro=[
            "最小的任务之一，标签是演唱发声方式而非音高内容。",
            "段落极短（10.7 min / 1,280 段 ≈ 0.5 s/段），片段级特征图与片段一一对应，没有「一曲多片」的切片膨胀。",
        ],
        extra="因此它**不属于**切片泄漏的受害者：K=1.0，同一段录音只出现一次。",
    ),
    dict(
        key="acapella", zh="无伴奏唱功评价", task="9 维主观评分回归 / 唱功评价",
        docs="132 段、22 位歌手、6 首歌曲，3.47 小时，9 个维度 10 分制主观评分。",
        page=LIB + "/database/ccm.html#shou2",
        origin="CCMusic 多功能库（ccm.html#shou2）；论文 Li & Zhang (2022), Applied Sciences 12(19):9931。",
        intro=[
            "**本次未下载音频**：该数据集只有 `dataset_infos.json` 一个文件被取到，目录已在清理时移除。",
            "需要时重新执行：`python code/ccmusic_download.py --dataset acapella`（约 1.3 GB）。",
        ],
        extra="它是唯一带「人对人」主观评分的唱功数据（22 位歌手 × 9 维），做评分回归时样本量偏小，宜作外部验证集。",
    ),
    dict(
        key="CNPM", zh="中国五声调式识别", task="调式识别（四轴：宫系统 12 / 主音 12 / 调式型 5 / 类型 6）",
        docs="287 首，14.3 小时，四轴对齐标注；eval 为 20 s 切片。",
        page=LIB + "/database/csmtd.html#shou10",
        origin="中国音乐学院「中国五声调式」数据集，CCMusic 集成（csmtd.html#shou10）；论文 Wang et al., ISMIR 2022。",
        intro=[
            "四轴标注让「调式识别」可以拆成可诊断的子问题：是宫系统判错，还是主音判错。",
            "`data/label.csv` 为 **GBK** 编码（Excel 导出），11 列中英双语表头；直接按 UTF-8 读会得到乱码。",
        ],
        extra="音频混有 mp3 与 wav（196 + 91），采样率 44.1/48 kHz 不一致，预处理需统一重采样。",
    ),
    dict(
        key="Guzheng_Tech99", zh="古筝演奏技法（逐帧）", task="逐帧多标签技法检测（7 类）",
        docs="99 首古筝曲，151 分钟，7 类技法逐帧标注，63,352 个标签；官方按曲目 79/10/10 划分。",
        page=LIB + "/database/csmtd.html#Tech99",
        origin="CSMTD 古筝逐帧技法数据集，CCMusic 集成（csmtd.html#Tech99）；论文 Li et al., ICASSP 2023。",
        intro=[
            "13 个数据集里**唯一按曲目划分**的一个（train 79 / validation 10 / test 10），也是唯一提供逐帧标注的。",
            "每首曲子 1 个 flac + 1 个逐帧 CSV + 1 条 mel 长条（5,360–7,853 × 128），适合帧级序列建模。",
        ],
        extra="因此它的 0.862 逐帧 F1 不能与其它数据集 0.98+ 的切片级精度直接比较——两者评测协议不同。",
    ),
    dict(
        key="instrument_timbre", zh="乐器音色感知", task="16 维音色描述词评分回归",
        docs="61 件乐器（中国 37 / 西方 24），每件 16 个描述词的 9 分制评分。",
        page=LIB + "/database/ccm.html#shou4",
        origin="CCMusic 多功能库（ccm.html#shou4）；论文 Jiang et al. (2020), Applied Sciences 10(3):789。",
        intro=[
            "把「音色」从抽象嵌入落到 16 个可解释的描述词上，适合做表征-语义对齐。",
            "`data/mel/*/<id>.jpg` 为整段 mel 长条（417–566 × 128）；`Chinese.csv` / `Western.csv` 各 16 列分数。",
        ],
        extra="官方卡片里的 16 个术语与 CSV 实际列名有 12 个不一致（如卡片写 warm，CSV 写 soft），做标签映射时要照 CSV 来。",
    ),
    dict(
        key="song_structure", zh="歌曲结构分析", task="9 类功能段落标注（前奏/主歌/副歌/桥段…）",
        docs="300 首，9 类结构功能标签，4.25 GB。",
        page="https://doi.org/10.5281/zenodo.5676893",
        origin="CCMusic 多功能库（Zenodo 10.5281/zenodo.5676893），随 CCMusic 主文献发布。",
        intro=[
            "**本次未下载**：它不在 p1/p2/p3/p5/p7 任一研究预设内，`--project all` 也未包含。",
            "需要时：`python code/ccmusic_download.py --dataset song_structure`（4.25 GB，含音频）。",
        ],
        extra="结构标签是「时间轴上的段落」，做 P1 的切片泄漏审计时需要它的段落边界来定义「同一段落内的切片」。",
    ),
    dict(
        key="GZ_IsoTech", zh="古筝技法（片段级）", task="8 类片段级技法分类",
        docs="2,824 个片段（2,328 虚拟音源 + 496 真人演奏），8 类技法；官方给 train/test。",
        page=LIB + "/database/csmtd.html#GZTech",
        origin="CSMTD 古筝技法数据集，CCMusic 集成（csmtd.html#GZTech）；论文 Li et al., ISMIR 2022。",
        intro=[
            "与 Guzheng_Tech99 同源不同粒度：这里是「一个片段一个技法」，那里是「一帧一个技法」。",
            "虚拟音源占 82%，天然带一个 domain-shift 对照（合成 vs 真人），可以不做任何额外采集就做跨域实验。",
        ],
        extra="`data/audio/{train,test}/<8类>/` 已按官方划分放好；eval 2,899 张是片段级特征图。",
    ),
    dict(
        key="bel_canto", zh="美声/民族唱法", task="4 类唱法分类",
        docs="203 段、5.08 小时、4 类（美声/民族及其子类）。",
        page=LIB + "/database/ccm.html#shou9",
        origin="CCMusic 多功能库（ccm.html#shou9），随 CCMusic 主文献发布。",
        intro=[
            "专业性最强的演唱数据：美声与民族唱法的区分依赖共鸣方式与音色，而非旋律。",
            "本地只有 eval 特征图（29,730 张），没有音频；需要音频请补下 `audio.zip`。",
        ],
        extra="官方 README 的引用直接指向 CCMusic 主文献，没有单独的数据集论文。",
    ),
    dict(
        key="erhu_playing_tech", zh="二胡演奏技法", task="11 类技法分类（另有 4 类 / 7 类子集）",
        docs="1,253 段、25.8 分钟、11 类；4 类与 7 类子集各 635 段。",
        page=LIB + "/database/ccm.html#shou8",
        origin="CCMusic 多功能库（ccm.html#shou8）；论文 Wang et al., arXiv:1910.09021（FCN 技法检测）。",
        intro=[
            "`data/audio/` 下有三个平行目录：`11_classes`（1,253）、`4_classes`（635）、`7_classes`（635）。",
            "4/7 类是同一批录音的**重新打标**，所以本地音频文件数 2,523 ≠ 独立录音数 1,253——统计时长时会被算重。",
        ],
        extra="因此实测 0.83 h 是把三套目录相加的结果，按独立录音算约 25.8 min，与官方文档一致。",
    ),
    dict(
        key="CTIS", zh="中国传统乐器音响", task="乐器识别（219 类）/ 技法识别（文件名内嵌）",
        docs="4,956 条录音、32.6 小时、209 种乐器（219 个类别目录），消声室录制。",
        page=LIB + "/database/ctis.html",
        origin="中国传统乐器音响数据集（CTIS），CCMusic 集成；原始论文 Liang et al., CSMT 2019。",
        intro=[
            "唯一在标准消声室录制、可用作「干净参考信号」的子集（32 支 B&K 话筒，44.1/48/96 kHz 混用）。",
            "`data/audio/<C0xxx 乐器编号>/` 219 个目录，文件名内嵌演奏技法——技法标签要从文件名解析，没有独立 CSV。",
        ],
        extra="类别极不均衡（219 类 / 4,956 条 ≈ 22 条/类），是检验长尾与 few-shot 方法最自然的场地。",
    ),
    dict(
        key="timbre_range", zh="音色与音域", task="音色相似度 / 音域跨度分析（多标签）",
        docs="775 个片段、630 MB、9 位歌手；文档最不完整的一个子集。",
        page="https://doi.org/10.5281/zenodo.5676893",
        origin="CCMusic 多功能库（Zenodo 10.5281/zenodo.5676893），随 CCMusic 主文献发布。",
        intro=[
            "本地实为两个子目录：`data/timbre/audios/` 672 条 + `data/range/audios/` 726 条 = 1,398 条，比文档写的 775 多。",
            "`timbre.csv` 只有 32 行，`range.csv` 729 行（label ∈ {0,1,2}）——标签覆盖不全，只能当弱标注用。",
        ],
        extra="采样率混杂（16/44.1/48 kHz），且 `audios` 目录里混着同名 jpg，遍历时需按扩展名过滤。",
    ),
]

CITENOTE = {
    "pianos": "正式出版版本为 Springer LNEE 2024, pp. 3–17，DOI 10.1007/978-981-97-7962-8_1；官方 README 仍给 2023 会议版，投稿引用时以在线出版信息为准。",
    "music_genre": "该子集没有单独论文，引用 CCMUSIC Zenodo 数据集条目 + §3 的 CCMusic 主文献。",
    "chest_falsetto": "该子集没有单独论文，引用 CCMUSIC Zenodo 数据集条目 + §3 的 CCMusic 主文献。",
    "song_structure": "该子集没有单独论文，引用 CCMUSIC Zenodo 数据集条目 + §3 的 CCMusic 主文献。",
    "timbre_range": "该子集没有单独论文，引用 CCMUSIC Zenodo 数据集条目 + §3 的 CCMusic 主文献。",
    "bel_canto": "官方 README 直接引 CCMusic 主文献，没有子集论文——正文里要写清集成工作归属。",
    "instrument_timbre": "官方 BibTeX 缺卷期与 DOI；完整出版信息为 Applied Sciences 10(3): 789，DOI 10.3390/app10030789。",
    "acapella": "官方 BibTeX 缺卷期与 DOI；完整出版信息为 Applied Sciences 12(19): 9931，DOI 10.3390/app12199931，作者 Rongfeng Li & Mingtong Zhang。本地未下载音频。",
}

STATS = json.load(io.open(os.path.join(ROOT, "data", "actual_stats.json"), encoding="utf-8"))


# ---------------------------------------------------------------- 工具函数
def read_readme(key):
    p = os.path.join(README_DIR, key + ".readme.md")
    if not os.path.exists(p):
        return ""
    return io.open(p, encoding="utf-8", errors="replace").read()


def bibtex_of(key):
    """取官方 README 中 Cite/引用 附近的第一个 bibtex 块。"""
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
        return ["<未下载>"]
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
            out.append(f"{prefix}└─ …（另 {len(dirs) - 4:,} 个同类目录）")
        moref = len(files) > 3
        for j, n in enumerate(files[:3]):
            last = j == min(len(files), 3) - 1 and not dirs and not moref
            out.append(f"{prefix}{'└─ ' if last else '├─ '}{n}")
        if moref:
            out.append(f"{prefix}└─ …（另 {len(files) - 3:,} 个文件）")

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


# ---------------------------------------------------------------- 主流程
def main():
    rows, body = [], []
    for i, d in enumerate(DS, 1):
        key = d["key"]
        st = STATS.get(key, {})
        bits = []
        if st.get("audio_files"):
            bits.append(f"音频 {st['audio_files']:,} 条 / {st['duration_h']:.2f} h")
        else:
            bits.append("音频 0 条（本地只有特征图）")
        if os.path.isdir(os.path.join(DATA, key)):
            nimg = sum(1 for dp, _, fs in os.walk(os.path.join(DATA, key))
                       for f in fs if f.lower().endswith((".jpg", ".png", ".jpeg")))
            if nimg:
                bits.append(f"图像 {nimg:,} 张")
        base = os.path.join(DATA, key)
        vol = human(du(base)) if os.path.isdir(base) else "—"
        rows.append(f"| {i} | `{key}` | {d['zh']} | {d['task']} | {vol} |")

        lines = [f"### {i}. `{key}` — {d['zh']}", ""]
        lines += [f"- **任务**：{d['task']}", f"- **内容**：{d['docs']}"]
        lines += [f"- **本地实测**：{'；'.join(bits)}；合计 {count_files(base):,} 个文件，{vol}"
                  if os.path.isdir(base) else "- **本地实测**：未下载"]
        lines.append("")
        for t in d["intro"]:
            lines.append(t)
            lines.append("")
        if d.get("extra"):
            lines += [f"> {d['extra']}", ""]
        if os.path.isdir(os.path.join(DATA, key)):
            lines += ["```", f"{key}/data/"] + tree_lines(key) + ["```", ""]
        lines += [f"- **原始出处**：<{d['page']}> —— {d['origin']}",
                  f"- **集成版**：[ModelScope `ccmusic-database/{key}`](https://www.modelscope.cn/datasets/ccmusic-database/{key})",
                  f"- **本地获取**：`modelscope download --dataset ccmusic-database/{key} --local-dir D:\\VSCode\\Datasets\\{key} --cache-dir D:\\VSCode\\Datasets\\.cache`", ""]
        bib = bibtex_of(key)
        if bib:
            lines += ["- **引用（官方 README 原文）**：", "", "```bibtex", bib.strip(), "```", ""]
            if CITENOTE.get(key):
                lines += [f"> {CITENOTE[key]}", ""]
        else:
            lines += ["- **引用**：官方 README 未给出独立 BibTeX，按 CCMusic 主文献引用（见下方 §3）。", ""]
        body.append("\n".join(lines))

    head = f"""# CCMusic 数据集（本地副本，13 个）

> 来源：ModelScope 组织 [`ccmusic-database`]({MS_ORG})（CCMusic 多功能音乐数据库，中国音乐学院 / 复旦大学 FD-LAMT / 浙江大学 NEXT（依 CCMusic 论文作者单位））。
> 本地路径：`D:\\VSCode\\Datasets\\<dataset_id>\\data\\...`，保留 ModelScope 仓库内相对路径，可直接套用官方 loader。
> 下载与解压核验：2026-11-02。许可：**CC BY-NC-ND 4.0**（署名—非商业—禁演绎），见 §3。

## 0. 目录约定

```
D:\\VSCode\\Datasets\\
├─ README.md          本文件
├─ _manifest.json     下载清单：26 个文件、体积、sha256 前 16 位、所用引擎
├─ _extract.json      解压对账记录：21 个 zip 的条目数 / 字节数
├─ .cache\\            modelscope 缓存（可整目录删除）
└─ <dataset_id>\\data\\ 各数据集
```

四类内容，含义不同，不要混用：

| 目录 | 是什么 | 命名 |
| :-- | :-- | :-- |
| `data/audio/` | 原始音频 | wav / mp3 / flac，采样率不统一（16–96 kHz） |
| `data/eval/` | **预计算特征图**，不是音频切片文件 | 多为 496×369 JPEG（timbre_range 为 476×349），分 `chroma/`、`cqt/`、`mel/` 三套 |
| `data/mel/` | 整段录音的 mel 长条 | 高 128，宽随时长（417–7,853） |
| `*.csv` / `label/` | 标注 | 仅 5 个数据集有；CNPM 为 GBK 编码 |

因此「切片数」= `data/eval/` 下每个特征目录的图片数 ÷ 3，并不等于音频数。27 个 zip 均已删除，删除前逐个核对条目数与字节数一致（记录在 `_extract.json`）。

## 1. 总览

| # | 数据集 | 中文 | 任务 | 本地体积 |
| :-: | :-- | :-- | :-- | --: |
{chr(10).join(rows)}

**本地缺音频的三个**：`bel_canto`、`chest_falsetto`、`music_genre` 盘面只有特征图、音频 0 条；补下命令见各小节。

## 2. 逐数据集

"""
    tail = """
## 3. 统一引用与许可

**CCMusic 集成库主文献**（13 个数据集的集成、切片与 eval 划分都出自这项工作，引用数据集时必引）：

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

**CCMUSIC Zenodo 数据集**（music_genre / chest_falsetto / song_structure / timbre_range）：

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

- 各数据集若有独立论文，BibTeX 已在该小节内原样给出（来源：ModelScope 官方 README）。
- 集成版本（切片、特征图、eval 划分）由 CCMusic 团队完成；**原始采集**的署名见各小节的「原始出处」。转引时两级都要给，否则等于把集成工作算在原采集方头上。
- 许可 **CC BY-NC-ND 4.0**：可复制、可再分发，须署名、**不得商用、不得发布演绎作品**。以这些数据训练出的模型权重、特征、统计结果属于演绎作品，公开发布前需确认合规；学术论文内部使用不受影响。

## 4. 使用注意（都是实测踩出来的）

1. **文件编码**：`CNPM/data/label.csv` 是 GBK；用 UTF-8 读会得到 `����/Title` 这类乱码，`pandas.read_csv(..., encoding="gbk")` 即可。
2. **时长会被算重**：`erhu_playing_tech` 的 `11/4/7_classes` 是同一批录音的不同打标，直接累加目录得到 0.83 h，独立录音只有约 25.8 min。
3. **采样率不齐**：CTIS 44.1/48/96 kHz 混用，timbre_range 混到 16 kHz，pianos 44.1/48 kHz。任何 batch 化预处理都要先统一重采样。
4. **标签来源不统一**：只有 5 个数据集有 CSV（CNPM 1 个、Guzheng_Tech99 99 个逐帧、instrument_timbre 2 个、timbre_range 2 个），其余靠目录名或文件名解析（CTIS 的技法就写在文件名里）。
5. **官方文档与实际不符之处**：`timbre_range` 文档称 775 片段 / 9 位歌手，实际 1,398 条、`timbre.csv` 仅 32 行；`instrument_timbre` 卡片的 16 个描述词与 CSV 列名有 12 个不同；`pianos` 文档写 2023 CSMT，实际出版于 LNEE 2024。以文件与官方 README 为准。
6. **不要用 `data/eval` 的图片数当独立样本数**：同一段录音的多个切片共享内容，切片级随机划分会让测试集里 97–100% 的样本在训练集里有同源兄弟（详见 `音乐情感研究` 项目里的泄漏审计）。
"""
    txt = head + "\n".join(body) + tail
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(txt)
    print("wrote", OUT, len(txt), "chars")


if __name__ == "__main__":
    main()
