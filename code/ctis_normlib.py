import re, unicodedata
def l1(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("\u3000", " ").replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[A-G]\s*调", "", s)
    s = re.sub(r"(?<=\S)\s*调$", "", s)
    s = re.sub(r"第\d+代", "", s)
    s = re.sub(r"(?<=[\u4e00-\u9fa5])\d$", "", s)
    s = re.sub(r"\s*[-—]\s*", "", s)
    return s.strip()

L2 = {
    "扬剧主胡（小西皮）": ("扬剧主胡", "tone-style qualifier: 小西皮 is a Yangju vocal style, not a separate instrument"),
    "雅筝": ("雅托嘎", "alias: 雅托嘎 is the Mongolian zither, i.e. 雅筝"),
    "伽倻琴（改良）": ("伽倻琴", "version marker: the improved type belongs with the prototype"),
    "奚琴（传统）": ("奚琴", "version marker"),
    "奚琴（改良）": ("奚琴", "version marker"),
    "中音奚琴（改良）": ("中音奚琴", "version marker"),
    "盖板（新）": ("盖板", "version marker"),
    "盖板（传统）": ("盖板", "version marker"),
    "陇剧陇胡（传统）": ("陇剧陇胡", "version marker"),
    "陇剧陇胡（改良）": ("陇剧陇胡", "version marker"),
    "短箫（传统）": ("短箫", "version marker"),
}
PENDING = {
    "萧": "probably a variant spelling of 「箫」; no 「箫」 anywhere in the inventory, kept separate",
    "中虎": "probably a mis-spelling of 「中胡」 (which has 36 further entries); unproven, so not merged",
    "二馨": "probably a mis-spelling of 「二弦」 or 「二胡」; unproven, so not merged",
    "福[新]": "probably the new version of 「宜春三星鼓 福鼓」; not merged",
    "禄[新]": "probably the new version of 「宜春三星鼓 禄鼓」; not merged",
    "寿[新]": "probably the new version of 「宜春三星鼓 寿鼓」; not merged",
    "铙": "whether it is the same instrument as 「铙钹」 is unproven; not merged",
    "南音三弦": "whether it is a regional variant of 「三弦」 is unproven; not merged",
}
L2N = {l1(k): v for k, v in L2.items()}

def canon(raw: str):
    a = l1(raw)
    if a in L2N:
        return L2N[a][0]
    return a

def canon_reason(raw: str):
    a = l1(raw)
    c = canon(raw)
    if c == raw:
        return ""
    return L2N[a][1] if a in L2N else "L1 mechanical canonicalization: whitespace / full-half-width punctuation / key names / generation / index"
