"""Download the small text/label files of the emo163 dataset repo from ModelScope."""
import json
import os
import ssl
import sys
import urllib.request

BASE = "https://www.modelscope.cn/api/v1/datasets/Genius-Society/emo163/repo"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "repo")

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

FILES = [
    "README.md",
    "dataset_infos.json",
    "quickstart.md",
    "emo163.py",
    ".gitattributes",
    "data/data.py",
    "data/songs-train.jsonl",
    "data/songs-valid.jsonl",
    "data/songs-test.jsonl",
]


def fetch(path: str) -> None:
    url = f"{BASE}?Revision=master&FilePath={urllib.parse.quote(path)}"
    target = os.path.join(OUT, path.replace("/", os.sep))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"SKIP {path} ({os.path.getsize(target)} bytes)")
        return
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=CTX, timeout=180) as resp, open(target, "wb") as fh:
        data = resp.read()
        fh.write(data)
    print(f"OK   {path} {len(data)} bytes")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for p in FILES:
        try:
            fetch(p)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {p}: {exc}", file=sys.stderr)
