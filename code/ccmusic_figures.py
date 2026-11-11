"""Generate the three figures for the CCMusic review (pure Pillow, no matplotlib)."""
import json
import math
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report", "figures_ccmusic")
os.makedirs(FIG, exist_ok=True)

INK = (0x1F, 0x23, 0x2B)
ACCENT = (0x1F, 0x3B, 0x63)
GREY = (0x5A, 0x63, 0x70)
LIGHT = (0xE8, 0xEC, 0xF2)
RED = (0xB0, 0x3A, 0x2E)
GREEN = (0x2F, 0x6B, 0x4F)
AMBER = (0xB8, 0x86, 0x0B)
FONT = "C:/Windows/Fonts/msyh.ttc"
FONTB = "C:/Windows/Fonts/msyhbd.ttc"


def f(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def ctext(d, xy, s, font, fill=INK, anchor="mm"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def heat(v):
    """1 -> red, 3 -> amber, 5 -> green (interpolated)."""
    v = max(1.0, min(5.0, v))
    if v <= 3:
        t = (v - 1) / 2
        a, b = RED, AMBER
    else:
        t = (v - 3) / 2
        a, b = AMBER, GREEN
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ------------------------------------------------------------------ figure 1
def fig1(facts):
    base = {d["name"]: d for d in facts["datasets"]}
    rows = [
        ("erhu_playing_tech", 0.994, "切片随机"),
        ("chest_falsetto", 0.994, "切片随机"),
        ("CTIS", 0.980, "切片随机"),
        ("bel_canto", 0.948, "切片随机"),
        ("Guzheng_Tech99", 0.862, "按曲目"),
        ("CNPM", 0.859, "切片随机"),
        ("GZ_IsoTech", 0.855, "按片段"),
    ]
    rows.sort(key=lambda r: -r[1])
    W, H = 1500, 860
    L, R, T, B = 330, 190, 150, 90
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ctext(d, (L, 52), "CCMusic 各数据集公开基线精度与划分层级", f(38, True), ACCENT, "lm")
    ctext(d, (L, 100), "红色=随机划分（切片重复，或同一演奏者/音源跨集）；绿色=按曲目或按源划分", f(21), GREY, "lm")

    plot_w = W - L - R
    x0 = L
    for g in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = x0 + g * plot_w
        d.line([(x, T - 10), (x, H - B + 10)], fill=LIGHT, width=2)
        ctext(d, (x, H - B + 34), f"{g:.1f}", f(20), GREY)
    ctext(d, (x0 + plot_w / 2, H - 22), "测试精度 / F1（模型卡最佳值）", f(22), INK)

    rowh = (H - T - B) / len(rows)
    for i, (name, acc, kind) in enumerate(rows):
        y = T + i * rowh + rowh / 2
        col = RED if kind == "切片随机" else GREEN
        label = f"{name}"
        ctext(d, (L - 18, y), label, f(22, True), INK, "rm")
        if name == "Guzheng_Tech99":
            ctext(d, (L - 18, y + 22), "（逐帧 F1）", f(17), GREY, "rm")
        bar_w = acc * plot_w
        d.rounded_rectangle([x0, y - 20, x0 + bar_w, y + 20], radius=8, fill=col)
        ctext(d, (x0 + bar_w + 14, y), f"{acc:.3f}", f(23, True), col, "lm")
        ctext(d, (x0 + bar_w + 92, y), kind, f(17), GREY, "lm")
    img.save(os.path.join(FIG, "fig1_baseline_vs_split.png"))


# ------------------------------------------------------------------ figure 2
def fig2(facts):
    rows = facts["datasets"]
    dims = facts["dims"]
    cols = [k for k, _, _ in dims]
    heads = [h for _, h, _ in dims]
    order = sorted(rows, key=lambda r: -r["composite"])
    cw, ch = 140, 52
    lw, hh = 300, 168
    W = lw + cw * len(cols) + 210
    H = hh + ch * len(order) + 120
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ctext(d, (lw - 30, 48), "CCMusic 13 个数据集质量评估热力图（1 差 – 5 优）", f(34, True), ACCENT, "lm")

    for j, h in enumerate(heads):
        ctext(d, (lw + j * cw + cw / 2, hh - 36), h, f(20, True), INK)
    for i, r in enumerate(order):
        y = hh + i * ch
        ctext(d, (lw - 16, y + ch / 2), f"{r['name']}", f(20), INK, "rm")
        for j, k in enumerate(cols):
            v = r["scores"][k]
            x = lw + j * cw
            d.rectangle([x + 2, y + 2, x + cw - 2, y + ch - 2], fill=heat(v))
            ctext(d, (x + cw / 2, y + ch / 2), str(v), f(21, True), "white")
    # composite column
    xc = lw + cw * len(cols) + 40
    ctext(d, (xc + 70, hh - 34), "综合分", f(20, True), INK)
    for i, r in enumerate(order):
        y = hh + i * ch + ch / 2
        ctext(d, (xc + 70, y), f"{r['composite']:.2f}", f(22, True), ACCENT)

    # legend
    ly = H - 62
    ctext(d, (lw, ly), "1", f(19, True), INK)
    for i in range(120):
        v = 1 + 4 * i / 119
        d.line([(lw + 24 + i * 2.2, ly - 9), (lw + 24 + i * 2.2, ly + 9)], fill=heat(v), width=3)
    ctext(d, (lw + 24 + 120 * 2.2 + 12, ly), "5", f(19, True), INK)
    ctext(d, (lw + 330, ly), "维度权重：标注 20%、评测协议 20%、规模 15%、完整性 15%、文档 10%、合规 10%、生态 10%", f(17), GREY, "lm")
    img.save(os.path.join(FIG, "fig2_quality_heatmap.png"))


# ------------------------------------------------------------------ figure 3
def fig3(facts):
    """每个数据集 eval 包的切片数 / 独立源数 = K，以及实测同源泄漏率。"""
    rows = sorted(facts["audit"], key=lambda r: -r[3])
    W, H = 1500, 940
    L, R, T, B = 290, 330, 176, 116
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ctext(d, (L - 40, 48), "每个源录音被切成几片：K 与实测泄漏率", f(36, True), ACCENT, "lm")
    ctext(d, (L - 40, 104), "K = eval 切片数 / 独立源数；泄漏率 = 复现官方划分后，测试切片中同源兄弟也落在训练集的比例", f(20), GREY, "lm")

    pw = W - L - R
    kmax = 60.0
    kx = lambda k: L + (k / kmax) * pw

    for t in [0, 10, 20, 30, 40, 50, 60]:
        x = kx(t)
        d.line([(x, T - 12), (x, H - B + 12)], fill=LIGHT, width=1)
        ctext(d, (x, H - B + 34), str(t), f(19), GREY)
    ctext(d, (L + pw / 2, H - 48), "K（每条源录音的切片数）", f(22), INK)
    ctext(d, (W - R + 20, T - 40), "实测泄漏率", f(19, True), INK, "lm")

    rowh = (H - T - B) / len(rows)
    for i, (name, slices, sources, k, leak) in enumerate(rows):
        cy = T + i * rowh + rowh / 2
        col = RED if leak > 1 else GREEN
        ctext(d, (L - 18, cy), name, f(21, True), INK, "rm")
        bar = max(6, kx(k) - L)
        d.rounded_rectangle([L, cy - 17, L + bar, cy + 17], radius=7, fill=col)
        ctext(d, (L + bar + 10, cy), f"K={k:g}", f(19), col, "lm")
        txt = f"{leak:.0f}%" if leak in (0, 100) else f"{leak:.1f}%"
        ctext(d, (W - R + 40, cy), txt, f(23, True), col, "lm")
        ctext(d, (W - R + 150, cy), f"{sources:,} 源 / {slices:,} 片", f(16), GREY, "lm")
    ctext(d, (L, H - 14), "注：K≈1 只说明没有切片重复，不代表划分干净——同一演奏者/录音批次的片段仍可能跨集。", f(17), GREY, "lm")
    img.save(os.path.join(FIG, "fig3_slice_leakage.png"))
    for old in ("fig3_sample_inflation.png", "fig3_leakage.png"):
        p = os.path.join(FIG, old)
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    with open(os.path.join(ROOT, "data", "ccmusic", "facts.json"), encoding="utf-8") as fh:
        facts = json.load(fh)
    fig1(facts)
    fig2(facts)
    fig3(facts)
    for p in sorted(os.listdir(FIG)):
        print(p, os.path.getsize(os.path.join(FIG, p)))
