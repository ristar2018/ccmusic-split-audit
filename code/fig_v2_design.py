# -*- coding: utf-8 -*-
"""Render the three-arm design schematic of the focused design note (Pillow only, no matplotlib)."""
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
    (40, "A0  clean · matched", "training set holds no slice of the target recording", None),
    (640, "A1  exposed · matched", "training set holds sibling slices of the target recording", "same slice and recording counts as A0"),
    (1240, "A2  default protocol", "slices split at random 80/10/10", None),
]
PW, PH, PY = 580, 400, 150

for px, title, sub, note in PANELS:
    d.rounded_rectangle([px, PY, px + PW, PY + PH], radius=14, fill=PANEL, outline=(226, 228, 232), width=2)
    d.text((px + 24, PY + 18), title, font=F_SUB, fill=INK)
    d.text((px + 24, PY + 52), sub, font=F_SMALL, fill=MUTED)
    if note:
        d.text((px + 24, PY - 34), note, font=F_SMALL, fill=(150, 92, 30))

# card geometry
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


# A0: 3 clean recordings + 1 target recording (2 test + 2 unused)
card(40, 0, ["train"] * 5, label="recording C1")
card(40, 1, ["train"] * 5, label="recording C2")
card(40, 2, ["train"] * 3, label="recording C3")
card(40, 3, ["test", "test", "pool", "pool", "pool"], target=True, label="Q1")

# A1: 2 clean recordings (5 slices each) + 1 target recording (3 train + 2 test); training slice and recording counts match A0
card(640, 0, ["train"] * 5, label="recording C1")
card(640, 1, ["train"] * 5, label="recording C2")
card(640, 2, ["train", "train", "train", "test", "test"], target=True, label="Q1")

# A2: slices split at random, train/test slices of the same recording interleaved
card(1240, 0, ["train", "train", "test", "train", "train"], label="recording 1")
card(1240, 1, ["train", "test", "train", "train", "test"], label="recording 2")
card(1240, 2, ["test", "train", "train", "train", "train"], label="recording 3")
card(1240, 3, ["train", "train", "test", "train", "pool"], target=True, label="Q1")

# top caption
d.text((40, 34), "Three-arm design: one shared test panel Q_test; only the source relation between training set and test panel changes", font=F_TITLE, fill=INK)
d.text((40, 80), "m_c = training slices per class, n_c = training recordings per class; A0 and A1 are equal class by class on both",
       font=F_SMALL, fill=MUTED)

# legend
lx, ly = 40, PY + PH + 54
items = [("train slice", TRAIN), ("Q_test test slice", TEST), ("Q_pool unused slice", (255, 255, 255)), ("orange box = target recording Q", None)]
for text, color in items:
    if color is not None:
        d.rounded_rectangle([lx, ly, lx + 26, ly + 26], radius=5, fill=color, outline=EDGE, width=2)
    else:
        d.rounded_rectangle([lx, ly, lx + 26, ly + 26], radius=5, fill=(255, 255, 255), outline=TEST, width=3)
    d.text((lx + 36, ly + 13), text, font=F_SMALL, fill=INK, anchor="lm")
    lx += 36 + d.textlength(text, font=F_SMALL) + 46

d.text((40, H - 44), "Each panel: the test set is always Q_test (even-hash slices); A1 replaces an equal number of A0 clean training slices with Q_pool (odd-hash) slices; "
                     "A2 splits training and self-test at random by slice and is also evaluated on Q_test so that it is comparable with the other arms.",
       font=F_CAP, fill=MUTED)

os.makedirs("report/fig", exist_ok=True)
out = "report/fig/fig_v2_design.png"
img.save(out, dpi=(200, 200))
print(out, os.path.getsize(out))
