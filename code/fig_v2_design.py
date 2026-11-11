# -*- coding: utf-8 -*-
"""生成聚焦版方案的三臂设计示意图（Pillow，无 matplotlib）。"""
import os
from PIL import Image, ImageDraw, ImageFont

FONT = "C:/Windows/Fonts/msyh.ttc"
FONTB = "C:/Windows/Fonts/msyhbd.ttc"
INK = (32, 36, 44)
MUTED = (108, 116, 128)
TRAIN = (64, 108, 182)
TEST = (222, 126, 54)
POOL = (214, 216, 220)
EDGE = (150, 155, 162)
PANEL = (246, 247, 249)

W, H = 1860, 700
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)


def f(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


F_TITLE = f(30, True)
F_SUB = f(21, True)
F_TXT = f(19)
F_SMALL = f(17)
F_CAP = f(18)

PANELS = [
    (40, "A0  干净 · 匹配", "训练集不含目标录音的任何切片", None),
    (640, "A1  暴露 · 匹配", "训练集含目标录音的兄弟切片", "切片数与录音数与 A0 相同"),
    (1240, "A2  默认协议", "切片随机 80/10/10", None),
]
PW, PH, PY = 580, 400, 150

for px, title, sub, note in PANELS:
    d.rounded_rectangle([px, PY, px + PW, PY + PH], radius=14, fill=PANEL, outline=(226, 228, 232), width=2)
    d.text((px + 24, PY + 18), title, font=F_SUB, fill=INK)
    d.text((px + 24, PY + 52), sub, font=F_SMALL, fill=MUTED)
    if note:
        d.text((px + 24, PY - 34), note, font=F_SMALL, fill=(150, 92, 30))

# 卡片几何
CARD_W, CARD_H = 118, 250
CARD_Y = PY + 96
SQ, GAP = 34, 14


def card(px, idx, cells, target=False, label=None):
    x = px + 28 + idx * (CARD_W + 18)
    if target:
        d.rounded_rectangle([x - 4, CARD_Y - 4, x + CARD_W + 4, CARD_Y + CARD_H + 4], radius=12,
                            outline=TEST, width=4)
    else:
        d.rounded_rectangle([x, CARD_Y, x + CARD_W, CARD_Y + CARD_H], radius=10, outline=(206, 209, 214), width=2)
    for i, kind in enumerate(cells):
        sy = CARD_Y + 18 + i * (SQ + GAP)
        sx = x + (CARD_W - SQ) // 2
        fill = {"train": TRAIN, "test": TEST, "pool": (255, 255, 255)}[kind]
        d.rounded_rectangle([sx, sy, sx + SQ, sy + SQ], radius=5, fill=fill,
                            outline=EDGE if kind != "train" else TRAIN, width=2)
    d.text((x + CARD_W / 2, CARD_Y + CARD_H + 26), label, font=F_SMALL, fill=MUTED, anchor="mm")


# A0：3 个干净录音 + 1 个目标录音（2 测试 + 2 未用）
card(40, 0, ["train"] * 5, label="录音 C1")
card(40, 1, ["train"] * 5, label="录音 C2")
card(40, 2, ["train"] * 3, label="录音 C3")
card(40, 3, ["test", "test", "pool", "pool", "pool"], target=True, label="Q1")

# A1：2 个干净录音（各 5 片）+ 1 个目标录音（3 训练 + 2 测试），训练切片与录音数同 A0
card(640, 0, ["train"] * 5, label="录音 C1")
card(640, 1, ["train"] * 5, label="录音 C2")
card(640, 2, ["train", "train", "train", "test", "test"], target=True, label="Q1")

# A2：切片随机划分，同一录音的训练/测试切片混排
card(1240, 0, ["train", "train", "test", "train", "train"], label="录音 1")
card(1240, 1, ["train", "test", "train", "train", "test"], label="录音 2")
card(1240, 2, ["test", "train", "train", "train", "train"], label="录音 3")
card(1240, 3, ["train", "train", "test", "train", "pool"], target=True, label="Q1")

# 顶部说明
d.text((40, 34), "三臂设计：同一测试面板 Q_test，只改训练集与测试面板的源关系", font=F_TITLE, fill=INK)
d.text((40, 80), "m_c = 每类训练切片数，n_c = 每类训练录音数；A0 与 A1 在这两个量上逐类相等",
       font=F_SMALL, fill=MUTED)

# 图例
lx, ly = 40, PY + PH + 54
items = [("训练切片", TRAIN), ("Q_test 测试切片", TEST), ("Q_pool 未用切片", (255, 255, 255)), ("橙框 = 目标录音 Q", None)]
for text, color in items:
    if color is not None:
        d.rounded_rectangle([lx, ly, lx + 26, ly + 26], radius=5, fill=color, outline=EDGE, width=2)
    else:
        d.rounded_rectangle([lx, ly, lx + 26, ly + 26], radius=5, fill=(255, 255, 255), outline=TEST, width=3)
    d.text((lx + 36, ly + 13), text, font=F_SMALL, fill=INK, anchor="lm")
    lx += 36 + d.textlength(text, font=F_SMALL) + 46

d.text((40, H - 44), "每个面板：测试集恒为 Q_test（偶数哈希切片）；A1 用 Q_pool（奇数哈希切片）等量替换 A0 的干净训练切片；"
                     "A2 的训练与自测按切片随机划分，另在 Q_test 上评测以与其他臂可比。",
       font=F_CAP, fill=MUTED)

os.makedirs("report/fig", exist_ok=True)
out = "report/fig/fig_v2_design.png"
img.save(out, dpi=(200, 200))
print(out, os.path.getsize(out))
