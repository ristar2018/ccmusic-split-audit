"""Sanity-check the generated DOCX: structure counts, leftovers, table widths."""
import re
import zipfile

from docx import Document

PATH = r"D:\VSCode\科研任务\音乐情感研究\report\emo163_调研分析报告.docx"

doc = Document(PATH)
paras = doc.paragraphs
tables = doc.tables
shapes = doc.inline_shapes

print("paragraphs:", len(paras))
print("tables:", len(tables))
print("inline images:", len(shapes))
print("sections:", len(doc.sections))
print("words approx:", sum(len(p.text) for p in paras))

heads = [(p.style.name, p.text[:50]) for p in paras if p.style.name.startswith("Heading")]
print("headings:", len(heads))
for h in heads[:40]:
    print("  ", h)

bad = []
for p in paras:
    t = p.text
    if "**" in t or (t.count("|") >= 2 and not t.startswith("|")) or re.search(r"!\[.*\]\(", t):
        bad.append(t[:90])
print("\nleftover-markup paragraphs:", len(bad))
for b in bad[:10]:
    print("  BAD:", b)

print("\ntable shapes:")
for i, t in enumerate(tables):
    print(f"  T{i+1}: rows={len(t.rows)} cols={len(t.columns)} first={t.rows[0].cells[0].text[:26]!r}")

with zipfile.ZipFile(PATH) as z:
    media = [n for n in z.namelist() if n.startswith("word/media/")]
    print("\nembedded media:", media)
    print("package size MB:", round(sum(i.file_size for i in z.infolist()) / 1e6, 2))
