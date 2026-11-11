import os
import ssl
import urllib.parse
import urllib.request

BASE = "https://www.modelscope.cn/api/v1/datasets/Genius-Society/emo163/repo"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

FILES = [
    "data-processor/dld.py",
    "data-processor/playlists.json",
    "data-processor/playlists.py",
    "data-processor/process.py",
    "data-processor/songs.jsonl",
    "data-processor/songs.py",
    "data-processor/utils.py",
]
OUT = os.path.join("data", "repo")

for p in FILES:
    t = os.path.join(OUT, p.replace("/", os.sep))
    os.makedirs(os.path.dirname(t), exist_ok=True)
    if os.path.exists(t) and os.path.getsize(t) > 0:
        print("SKIP", p, os.path.getsize(t))
        continue
    u = f"{BASE}?Revision=master&FilePath={urllib.parse.quote(p)}"
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=CTX, timeout=600) as r, open(t, "wb") as f:
            d = r.read()
            f.write(d)
        print("OK", p, len(d))
    except Exception as e:  # noqa: BLE001
        print("FAIL", p, e)
