"""Render the report figures with Pillow (no matplotlib in this environment)."""
import collections
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figures")
S = 2  # supersampling factor
CN = "C:\\Windows\\Fonts\\msyh.ttc"
CN_B = "C:\\Windows\\Fonts\\msyhbd.ttc"

INK = (32, 36, 44)
GREY = (120, 128, 140)
GRID = (226, 230, 236)
BLUE = (47, 92, 168)
TEAL = (39, 138, 138)
ORANGE = (222, 125, 44)
RED = (183, 58, 62)
PURPLE = (110, 79, 158)

NAMES = ["Nostalgic", "Fresh", "Romantic", "Melancholic", "Healing", "Relaxing",
         "Lonely", "Touching", "Exciting", "Joyful", "Peaceful", "Missing"]


def font(size, bold=False):
    return ImageFont.truetype(CN_B if bold else CN, int(size * S))


class Canvas:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * S, h * S), "white")
        self.d = ImageDraw.Draw(self.img)

    def text(self, xy, s, size=14, bold=False, fill=INK, anchor="la"):
        self.d.text((xy[0] * S, xy[1] * S), s, font=font(size, bold), fill=fill, anchor=anchor)

    def line(self, pts, fill=GRID, width=1):
        self.d.line([(x * S, y * S) for x, y in pts], fill=fill, width=max(1, int(width * S)))

    def rect(self, box, fill=None, outline=None, width=1):
        self.d.rectangle([box[0] * S, box[1] * S, box[2] * S, box[3] * S],
                         fill=fill, outline=outline, width=max(1, int(width * S)))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.img.resize((self.w, self.h), Image.LANCZOS).save(path, quality=96)


def nice_max(v, ticks=5):
    """Smallest friendly axis maximum made of `ticks` equal steps that covers v."""
    raw = v / ticks
    exp = math.floor(math.log10(raw)) if raw > 0 else 0
    base = 10 ** exp
    for mul in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        step = base * mul
        if step >= raw:
            return step * ticks
    return base * 10 * ticks


def rot_text(cv, xy, s, size=9.5, angle=45, fill=INK):
    """Draw a label rotated by `angle` degrees, right end anchored at xy."""
    f = font(size)
    tmp = Image.new("RGBA", (int(size * S * len(s) * 0.75) + 24, int(size * S * 2.2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    d.text((tmp.width - 2, int(size * S * 0.5)), s, font=f, fill=fill + (255,), anchor="ra")
    tmp = tmp.rotate(angle, expand=True, resample=Image.BICUBIC)
    cv.img.paste(tmp, (int(xy[0] * S - tmp.width), int(xy[1] * S)), tmp)


def draw_axes(cv, box, values, title):
    x0, y0, x1, y1 = box
    left, right, top, bottom = 62, 16, 40, 78
    px0, py0, px1, py1 = x0 + left, y0 + top, x1 - right, y1 - bottom
    vmax = nice_max(max(values))
    cv.text((x0, y0 + 4), title, 15, bold=True)
    for k in range(6):
        v = vmax * k / 5
        y = py1 - (py1 - py0) * k / 5
        cv.line([(px0, y), (px1, y)], fill=GRID, width=0.8)
        cv.text((px0 - 6, y), f"{v:,.0f}", 10, fill=GREY, anchor="rm")
    cv.line([(px0, py1), (px1, py1)], fill=INK, width=1.1)
    return px0, py0, px1, py1, vmax


def bar_chart(cv, box, labels, values, title, color=BLUE, fmt="{:.0f}", ymax=None,
              annotate=True):
    px0, py0, px1, py1, vmax = draw_axes(cv, box, values, title)
    if ymax:
        vmax = ymax
        for k in range(6):
            y = py1 - (py1 - py0) * k / 5
            cv.rect((px0 - 46, y - 8, px0 - 4, y + 8), fill="white")
            cv.text((px0 - 6, y), f"{ymax * k / 5:,.0f}", 10, fill=GREY, anchor="rm")
    n = len(values)
    slot = (px1 - px0) / n
    bw = slot * 0.6
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = px0 + slot * (i + 0.5)
        yv = py1 - (py1 - py0) * (v / (ymax or vmax))
        cv.rect((cx - bw / 2, yv, cx + bw / 2, py1), fill=color)
        if annotate:
            cv.text((cx, yv - 6), fmt.format(v), 9.5, fill=INK, anchor="mb")
        for j, part in enumerate(lab.split("\n")):
            cv.text((cx, py1 + 8 + j * 15), part, 9.5, fill=INK, anchor="ma")


def vbar_chart(cv, box, labels, values, title, color=BLUE, fmt="{:.0f}", ymax=None,
               annotate=True, angle=45):
    px0, py0, px1, py1, vmax = draw_axes(cv, box, values, title)
    n = len(values)
    slot = (px1 - px0) / n
    bw = slot * 0.66
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = px0 + slot * (i + 0.5)
        yv = py1 - (py1 - py0) * (v / (ymax or vmax))
        cv.rect((cx - bw / 2, yv, cx + bw / 2, py1), fill=color)
        if annotate:
            cv.text((cx, yv - 5), fmt.format(v), 9, fill=INK, anchor="mb")
        rot_text(cv, (cx + 5, py1 + 8), lab, 9.5, angle=angle)


def heatmap(cv, box, matrix, labels, title, vmin, vmax):
    x0, y0, x1, y1 = box
    left, top = 92, 40
    px0, py0 = x0 + left, y0 + top
    side = min(x1 - px0 - 90, y1 - py0 - 60)
    cell = side / len(labels)
    cv.text((x0, y0 + 4), title, 15, bold=True)

    def color(v):
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
        if t < 0.5:
            a, b, u = (252, 250, 246), (232, 152, 74), t / 0.5
        else:
            a, b, u = (232, 152, 74), (140, 32, 36), (t - 0.5) / 0.5
        return tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))

    for i, lab in enumerate(labels):
        cv.text((px0 - 7, py0 + cell * (i + 0.5)), lab, 10, fill=INK, anchor="rm")
        cv.text((px0 + cell * (i + 0.5), py0 + side + 6), lab, 9, fill=INK, anchor="ra")
    for i in range(len(labels)):
        for j in range(len(labels)):
            cx = px0 + cell * (j + 0.04)
            cy = py0 + cell * (i + 0.04)
            cv.rect((cx, cy, cx + cell * 0.92, cy + cell * 0.92), fill=color(matrix[i][j]))
    cbx, cby = px0 + side + 46, py0
    for k in range(60):
        v = vmax - (vmax - vmin) * k / 59
        cv.rect((cbx, cby + k * (side / 60), cbx + 12, cby + (k + 1) * (side / 60)), fill=color(v))
    cv.text((cbx + 18, cby), f"{vmax:.2f}", 9.5, fill=GREY, anchor="lm")
    cv.text((cbx + 18, cby + side), f"{vmin:.2f}", 9.5, fill=GREY, anchor="lm")
    cv.text((cbx + 18, cby + side / 2), "Jaccard", 9.5, fill=GREY, anchor="lm")


def fig1():
    audit = json.load(open(os.path.join(RES, "audit.json"), encoding="utf-8"))
    leak = json.load(open(os.path.join(RES, "audit2.json"), encoding="utf-8"))
    cv = Canvas(1600, 1080)
    cv.text((40, 24), "图1  emo163 标签表的结构性问题：规模、多标签、歌单粒度与官方划分泄漏", 19, bold=True)
    counts = [audit["label_counts"][n] / 1000 for n in NAMES]
    vbar_chart(cv, (30, 66, 800, 540), NAMES, counts,
               "(a) 12 类情感标签的行数（千条，合计 39.59 万）", color=BLUE, fmt="{:.1f}")
    hist = audit["labels_per_song_hist"]
    ks = [str(k) for k in range(1, 13)]
    vals = [hist.get(str(k), 0) / 1000 for k in range(1, 13)]
    bar_chart(cv, (820, 66, 1580, 540), ks, vals,
              "(b) 同一首歌被赋予的标签个数分布（千首）", color=ORANGE, fmt="{:.0f}")
    sizes = json.load(open(os.path.join(RES, "playlist_sizes.json"), encoding="utf-8"))["sizes"]
    bins = [(1, 5), (6, 10), (11, 25), (26, 50), (51, 100), (101, 200), (201, 500), (501, 2000)]
    labels = [f"{a}–{b}" if b < 2000 else f"{a}+" for a, b in bins]
    cnt = [sum(1 for s0 in sizes if a <= s0 <= b) for a, b in bins]
    bar_chart(cv, (30, 566, 800, 1036), labels, cnt,
              "(c) 歌单规模（每歌单歌曲数）分布，中位数 31 首", color=TEAL, fmt="{:.0f}")
    base = leak["official_split_baselines"]
    labels2 = ["均匀猜测", "多数类", "歌单ID查表"]
    vals2 = [8.3, base["majority_acc"] * 100, base["playlist_lookup_acc"] * 100]
    bar_chart(cv, (820, 566, 1580, 1036), labels2, vals2,
              "(d) 官方随机划分下的“作弊”基线（测试集准确率 %）", color=RED, fmt="{:.1f}",
              ymax=100)
    cv.text((890, 610), "注：“歌单ID查表”只用歌单编号到训练集里查标签，不使用任何音频特征。",
            10.5, fill=GREY)
    cv.save(os.path.join(FIG, "fig1_overview.png"))


def fig2():
    co = json.load(open(os.path.join(RES, "cooccurrence.json"), encoding="utf-8"))
    order = ["Melancholic", "Lonely", "Missing", "Nostalgic", "Touching", "Healing",
             "Relaxing", "Peaceful", "Romantic", "Fresh", "Joyful", "Exciting"]
    M = [[co["jaccard"][a][b] if co["jaccard"][a][b] is not None else 1.0 for b in order]
         for a in order]
    cv = Canvas(1120, 880)
    cv.text((40, 24), "图2  12 类标签的共现结构（按聚类合并顺序重排）", 19, bold=True)
    heatmap(cv, (30, 56, 1100, 856), M, order,
            "同一首歌同时落在两个标签下的 Jaccard 系数（对角=1）", 0.05, 0.27)
    cv.save(os.path.join(FIG, "fig2_structure.png"))


def fig3():
    ev = json.load(open(os.path.join(RES, "eval_zip2.json"), encoding="utf-8"))
    mins = ev["audio_minutes_per_label"]
    cv = Canvas(1600, 540)
    cv.text((40, 22), "图3  校验子集（eval.zip）：38.4 小时、约 716 首、12 类各约 60 首", 18, bold=True)
    vbar_chart(cv, (30, 60, 790, 520), NAMES, [mins[n] for n in NAMES],
               "(a) 每类音频时长（分钟；建集时上限 192 分钟/类）", color=PURPLE, fmt="{:.0f}")
    stats = ev["segment_count_stats"]
    cv.text((860, 84), "(b) 校验子集的构成要点", 15, bold=True)
    lines = [
        f"5 秒片段总数（每类频谱各一支）：{ev['mel_images']:,}",
        f"可辨识歌曲数：约 {ev['estimated_songs']} 首（{ev['total_audio_minutes']:.1f} 分钟）",
        f"每首片段数：中位 {stats['median']}，均值 {stats['mean']}，范围 {stats['min']}–{stats['max']}",
        f"歌曲时长：中位 {ev['duration_stats_sec']['median']} 秒，"
        f"范围 {ev['duration_stats_sec']['min']}–{ev['duration_stats_sec']['max']} 秒",
        f"图像：496×369 RGB、JPEG，平均 {ev['avg_jpeg_bytes'] / 1024:.0f} KB/张",
        "mel 与 cqt 名称不完全配对：各 113 个片段只有单侧频谱",
        "文件名不含歌曲 ID：图像无法与标签表按歌曲对齐",
    ]
    y = 130
    for ln in lines:
        cv.text((870, y), "· " + ln, 12)
        y += 28
    cv.text((870, y + 16), "含义：片段级样本量约 2.7 万条，但歌曲级只有约 716 首；", 12, fill=GREY)
    cv.text((870, y + 44), "做歌曲级推断时，准确率的 95% 置信区间约 ±3.6 个百分点。", 12, fill=GREY)
    cv.save(os.path.join(FIG, "fig3_eval_subset.png"))


def grouped_bar(cv, box, categories, series, title, fmt="{:.1f}", ymax=None):
    """series: list of (name, values, color); a None value leaves a gap."""
    x0, y0, x1, y1 = box
    left, right, top, bottom = 92, 16, 104, 78
    px0, py0, px1, py1 = x0 + left, y0 + top, x1 - right, y1 - bottom
    flat = [v for _, vals, _ in series for v in vals if v is not None]
    vmax = ymax or nice_max(max(flat))
    cv.text((x0, y0 + 6), title, 25, bold=True)
    for k in range(6):
        v = vmax * k / 5
        y = py1 - (py1 - py0) * k / 5
        cv.line([(px0, y), (px1, y)], fill=GRID, width=0.9)
        cv.text((px0 - 10, y), f"{v:,.0f}", 17, fill=GREY, anchor="rm")
    cv.line([(px0, py1), (px1, py1)], fill=INK, width=1.4)

    n = len(categories)
    slot = (px1 - px0) / n
    k = len(series)
    bw = slot * 0.66 / k
    for i, cat in enumerate(categories):
        for j, (name, vals, color) in enumerate(series):
            v = vals[i]
            cx = px0 + slot * (i + 0.5) + (j - (k - 1) / 2) * bw
            if v is None:
                cv.text((cx, py1 - 26), "n/a", 15, fill=GREY, anchor="mb")
                continue
            yv = py1 - (py1 - py0) * (v / vmax)
            cv.rect((cx - bw * 0.44, yv, cx + bw * 0.44, py1), fill=color)
            cv.text((cx, yv - 9), fmt.format(v), 16, fill=INK, anchor="mb")
        cv.text((px0 + slot * (i + 0.5), py1 + 14), cat, 18, fill=INK, anchor="ma")
    # legend
    lx = px0 + 2
    for name, _, color in series:
        cv.rect((lx, y0 + 34, lx + 18, y0 + 52), fill=color)
        cv.text((lx + 26, y0 + 43), name, 17, fill=INK, anchor="lm")
        lx += 36 + 17 * len(name)


def fig4():
    cv = Canvas(1560, 730)
    cv.text((36, 18), "图4  三协议下的基线塌陷，以及发布去重对图信号的影响", 27, bold=True)
    grouped_bar(cv, (30, 60, 760, 690),
                ["官方随机", "按歌曲分组", "按歌单分组"],
                [("歌单ID查表", [87.79, 88.51, None], RED),
                 ("多数类", [10.72, 10.70, 10.14], BLUE)],
                "(a) 零音频基线的测试准确率（%）", ymax=100)
    grouped_bar(cv, (800, 60, 1530, 690),
                ["边保留率", "同质性", "标签传播 top-1"],
                [("真实图（平台侧）", [100.0, 37.0, 99.9], TEAL),
                 ("按(歌曲,标签)去重后", [11.3, 0.0, 0.1], ORANGE)],
                "(b) 合成语料：同一发布流程前后的三个量（%）", ymax=100)
    cv.save(os.path.join(FIG, "fig4_protocol_collapse.png"))


def main():
    os.makedirs(FIG, exist_ok=True)
    repo = os.path.join(ROOT, "data", "repo")
    sizes = collections.Counter()
    with open(os.path.join(repo, "data-processor", "songs.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                sizes[json.loads(line)["playlist"]] += 1
    with open(os.path.join(RES, "playlist_sizes.json"), "w", encoding="utf-8") as fh:
        json.dump({"sizes": list(sizes.values())}, fh)
    fig1()
    fig2()
    fig3()
    fig4()
    print("figures written")


if __name__ == "__main__":
    main()
