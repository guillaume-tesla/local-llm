"""Download the raw English corpora into data/raw/ and verify their sha256.

Sources are NLTK data packages, mirrored on GitHub (nltk/nltk_data, gh-pages):

  gutenberg     18 public-domain books (Austen, Shakespeare, Melville, Bible, ...)
  brown         Brown Corpus, 1961 American English (500 files, POS-tagged)
  webtext       small web / forum / script text
  reuters       Reuters newswire articles (~10k files)
  movie_reviews 2000 movie reviews
  state_union   US State of the Union addresses
  abc           ABC Australia news (2 long files)

Usage:  python -m scripts.fetch_data [--raw data/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora"

SHA256 = {
    "abc": "129bb6001beb828049a90a59b7dd3c2f0594a47012e48fc5177dfae38e658565",
    "brown": "9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6",
    "gutenberg": "2d3c3ab548c653944310f37f536443ec85d0a0ad855fcae217a0c9efdce2d611",
    "movie_reviews": "a41211ae685019137410268134db6a1a14428c89b671eb83056151a878539008",
    "reuters": "9a59a43823f02a6e2777075c989a3dc454e4b6f68e0332ee3c0e8264075b62f5",
    "state_union": "366c1dc82b2abf896f42b2ec50ba802a0141a29f75d29ca48a7a243ce5bfbe8d",
    "webtext": "9e32dbae4879464b8f420a0dc721855bb26167b720d7695588d2ca2aeadf501a",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, raw_dir: Path) -> Path:
    dest = raw_dir / f"{name}.zip"
    if dest.exists() and sha256_of(dest) == SHA256[name]:
        print(f"[ok] {name} already present")
        return dest
    url = f"{BASE_URL}/{name}.zip"
    print(f"[get] {url}")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    digest = sha256_of(tmp)
    if digest != SHA256[name]:
        tmp.unlink()
        raise SystemExit(f"sha256 mismatch for {name}: got {digest}, expected {SHA256[name]}")
    tmp.replace(dest)
    print(f"[ok] {name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw", type=Path)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    for name in SHA256:
        fetch(name, args.raw)


if __name__ == "__main__":
    main()
