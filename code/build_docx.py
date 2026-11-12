"""Build the DOCX report from the Markdown source.

Supports the subset used by report/emo163_调研分析报告.md:
headings, paragraphs, bullets, numbered items, pipe tables, images, code
fences, block quotes, inline bold/code.
"""
import os
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "report", "emo163_调研分析报告.md")
OUT = os.path.join(ROOT, "report", "emo163_调研分析报告.docx")
if len(sys.argv) >= 2:
    MD = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(ROOT, sys.argv[1])
    if len(sys.argv) >= 3:
        OUT = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(ROOT, sys.argv[2])
    else:
        OUT = os.path.splitext(MD)[0] + ".docx"

CJK = "微软雅黑"
LATIN = "Times New Roman"
MONO = "Consolas"
INK = RGBColor(0x1F, 0x23, 0x2B)
ACCENT = RGBColor(0x1F, 0x3B, 0x63)
GREY = RGBColor(0x5A, 0x63, 0x70)


def set_run_font(run, name=CJK, size=None, bold=None, color=None, italic=None):
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), CJK)
    rfonts.set(qn("w:ascii"), LATIN if name == CJK else name)
    rfonts.set(qn("w:hAnsi"), LATIN if name == CJK else name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade(element, hex_fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    element.append(shd)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9, color=GREY)


def setup_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = LATIN
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = INK
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    pf = normal.paragraph_format
    pf.line_spacing = 1.45
    pf.space_before = Pt(2)
    pf.space_after = Pt(4)

    for name, size, color, before, after in (
        ("Heading 1", 16, ACCENT, 16, 8),
        ("Heading 2", 13, ACCENT, 12, 6),
        ("Heading 3", 11.5, INK, 10, 4),
        ("Heading 4", 11, INK, 8, 3),
    ):
        st = doc.styles[name]
        st.font.name = LATIN
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        st = doc.styles[name]
        st.font.name = LATIN
        st.font.size = Pt(10.5)
        st.font.color.rgb = INK
        st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        st.paragraph_format.line_spacing = 1.35
        st.paragraph_format.space_after = Pt(2)


BOLD_RE = re.compile(
    r"(\*\*.+?\*\*|`[^`]+?`|\[[^\]]+\]\([^)]+\)|(?<!\*)\*(?!\*)[^*\n]+\*(?!\*))")


def add_link(paragraph, url, text, size=10.5):
    part = paragraph.part
    r_id = part.relate_to(url, RT.HYPERLINK, is_external=True)
    hl = OxmlElement("w:hyperlink")
    hl.set(qn("r:id"), r_id)
    run = paragraph.add_run(text)
    set_run_font(run, size=size, color=RGBColor(0x1F, 0x4E, 0x9A))
    run.font.underline = True
    hl.append(run._r)
    paragraph._p.append(hl)


def add_rich(paragraph, text, size=10.5, color=INK, base_bold=False):
    for piece in BOLD_RE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            add_rich(paragraph, piece[2:-2], size=size, color=color, base_bold=True)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            run = paragraph.add_run(piece[1:-1])
            set_run_font(run, name=MONO, size=size - 0.5, bold=base_bold,
                         color=RGBColor(0x8B, 0x2E, 0x2E))
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2 \
                and not piece.startswith("**"):
            run = paragraph.add_run(piece[1:-1])
            set_run_font(run, size=size, bold=base_bold, color=color, italic=True)
        elif piece.startswith("[") and "](" in piece and piece.endswith(")"):
            label, url = piece[1:-1].split("](", 1)
            add_link(paragraph, url, label, size=size)
        else:
            run = paragraph.add_run(piece)
            set_run_font(run, size=size, bold=base_bold, color=color)


def dwidth(text):
    """Display width: CJK glyphs count double."""
    text = text.replace("**", "").replace("`", "")
    return sum(2 if ord(ch) > 127 else 1 for ch in text)


def clean_heading(text):
    """Strip bold, strip inline-code backticks, collapse whitespace."""
    text = text.replace("**", "")
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def longest_ascii_token(text):
    text = text.replace("*", "").replace("`", "")
    toks = re.split(r"[^0-9A-Za-z_.\-]+", text)
    return max((len(t) for t in toks), default=0)


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    table = doc.add_table(rows=0, cols=ncol)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    table._tbl.tblPr.append(layout)

    weights, needs = [], []
    for c in range(ncol):
        col = [header[c]] + [r[c] if c < len(r) else "" for r in body]
        weights.append(max(dwidth(x) for x in col))
        # the longest unbreakable English identifier (e.g. erhu_playing_tech) sets the minimum column width
        needs.append(max(1.15, 0.165 * max(longest_ascii_token(x) for x in col) + 0.34))

    # weight that shrinks long text columns (exponent 0.6): keeps claim/description columns from eating the width of the evidence column
    weights = [max(w, 1.0) ** 0.6 for w in weights]
    total = sum(weights) or ncol
    avail = 16.4
    if sum(needs) >= avail:
        widths = [avail * n / sum(needs) for n in needs]
    else:
        rest = avail - sum(needs)
        widths = [n + rest * w / total for n, w in zip(needs, weights)]

    row = table.add_row()
    row_tr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    row_tr.append(cant)
    hdr = OxmlElement("w:tblHeader")
    row_tr.append(hdr)
    for i, text in enumerate(header):
        cell = row.cells[i]
        cell.width = Cm(widths[i])
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.keep_with_next = True
        add_rich(p, text, size=9, color=ACCENT, base_bold=True)
        shade(cell._tc.get_or_add_tcPr(), "E8EDF5")

    for r in body:
        row = table.add_row()
        # keep a row on one page: a single claim must not be split across two pages
        row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        for i in range(ncol):
            cell = row.cells[i]
            cell.width = Cm(widths[i])
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            add_rich(p, r[i] if i < len(r) else "", size=8.8)
    for i, wdt in enumerate(widths):
        table.columns[i].width = Cm(wdt)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_code(doc, lines):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        shade(p._p.get_or_add_pPr(), "F4F5F7")
        run = p.add_run(line if line else " ")
        set_run_font(run, name=MONO, size=8.6, color=RGBColor(0x22, 0x2A, 0x35))
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_picture(doc, path, caption=None):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        p = doc.add_paragraph()
        add_rich(p, f"[missing image: {path}]", size=9, color=GREY)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = bool(caption)
    p.add_run().add_picture(full, width=Cm(15.6))
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(10)
        add_rich(cap, caption, size=9, color=GREY)


def parse_table_row(line):
    body = line.strip().strip("|")
    body = body.replace("\\|", "\u0001PIPE\u0001")
    cells = [c.strip().replace("\u0001PIPE\u0001", "|") for c in body.split("|")]
    return cells


def is_separator(line):
    return bool(re.match(r"^\|[\s:\-\|]+\|?$", line.strip())) and "-" in line


def build():
    with open(MD, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.3)
    sec.right_margin = Cm(2.3)

    footer_p = sec.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer_p)

    i = 0
    pending_caption = None
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i].rstrip())
                i += 1
            i += 1
            add_code(doc, block)
            continue

        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(6)
            add_rich(p, stripped[2:], size=19, color=ACCENT, base_bold=True)
            i += 1
            continue
        if stripped.startswith("## "):
            doc.add_heading(clean_heading(stripped[3:]), level=1)
            i += 1
            continue
        if stripped.startswith("### "):
            doc.add_heading(clean_heading(stripped[4:]), level=2)
            i += 1
            continue
        if stripped.startswith("#### "):
            doc.add_heading(clean_heading(stripped[5:]), level=3)
            i += 1
            continue

        if stripped == "---":
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            pPr = p._p.get_or_add_pPr()
            borders = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "C9D2DE")
            borders.append(bottom)
            pPr.append(borders)
            i += 1
            continue

        img = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if img:
            caption = None
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                if len(nxt) > 2 and nxt.startswith("*") and nxt.endswith("*") \
                        and not nxt.startswith("**") and "](" not in nxt:
                    caption = nxt.strip("*").strip()
                    i = j
            add_picture(doc, img.group(2), caption)
            i += 1
            continue

        if len(stripped) > 2 and stripped.startswith("*") and stripped.endswith("*") \
                and not stripped.startswith("**") and "](" not in stripped:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(stripped.strip("*").strip())
            set_run_font(run, size=9, color=GREY, italic=True)
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not is_separator(lines[i]):
                    rows.append(parse_table_row(lines[i]))
                i += 1
            if rows:
                add_table(doc, rows)
            continue

        if stripped.startswith(">"):
            body = stripped[1:].strip()
            if not body:
                i += 1
                continue
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.space_after = Pt(6)
            add_rich(p, body, size=10, color=INK)
            i += 1
            continue

        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_rich(p, stripped[2:])
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
            add_rich(p, f"{m.group(1)}. {m.group(2)}")
            i += 1
            continue

        p = doc.add_paragraph()
        add_rich(p, stripped)
        i += 1

    title = "Dataset survey report"
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break
    doc.core_properties.title = title
    doc.core_properties.subject = "music emotion computing · weak supervision · dataset audit"
    doc.core_properties.author = "research report"
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    build()
