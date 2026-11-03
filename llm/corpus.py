"""Read the raw NLTK zips into clean, documented plain text.

Every document gets a stable id and a source name. Splits are assigned per
document by hashing the id, so the same document always lands in the same
split, and held-out documents never leak into training.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SPLIT_MOD = 20  # bucket 0 -> dev, bucket 1 -> test, everything else -> train (5% each)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    source: str
    text: str


_GUT_START = re.compile(r"\*\*\*\s*START OF", re.IGNORECASE)
_GUT_END = re.compile(r"\*\*\*\s*END OF", re.IGNORECASE)
_PUNCT_LEFT = re.compile(r" ([.,;:!?%)\]])")
_PUNCT_RIGHT = re.compile(r"([(\[]) ")


def normalize(text: str) -> str:
    """Unify newlines, drop BOMs and trailing spaces, collapse runs of blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg license header and footer when present."""
    lines = text.split("\n")
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if _GUT_START.search(line):
            start = i + 1
            break
    for i in range(start, len(lines)):
        if _GUT_END.search(lines[i]):
            end = i
            break
    return "\n".join(lines[start:end])


def brown_to_text(raw: str) -> str:
    """Turn POS-tagged Brown text (`The/at Fulton/np-tl`) into plain prose."""
    out = []
    for line in raw.split("\n"):
        words = []
        for tok in line.split():
            word = tok.rsplit("/", 1)[0] if "/" in tok else tok
            if word in ("``", "''"):
                word = '"'
            words.append(word)
        if words:
            s = " ".join(words)
            s = _PUNCT_LEFT.sub(r"\1", s)
            s = _PUNCT_RIGHT.sub(r"\1", s)
            out.append(s)
    return "\n".join(out)


def chunk_paragraphs(text: str, size: int = 30) -> list[str]:
    """Group blank-line-separated blocks into pseudo-documents of `size` blocks."""
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    return ["\n\n".join(blocks[i : i + size]) for i in range(0, len(blocks), size)]


def split_of(doc_id: str) -> str:
    bucket = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:8], 16) % SPLIT_MOD
    if bucket == 0:
        return "dev"
    if bucket == 1:
        return "test"
    return "train"


def _read(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def _emit(doc_id: str, source: str, text: str) -> Iterator[Doc]:
    text = normalize(text)
    if text:
        yield Doc(doc_id, source, text)


def iter_docs(raw_dir: Path) -> Iterator[Doc]:
    """Yield every document from every corpus zip, in a deterministic order."""

    def zip_path(name: str) -> Path:
        return raw_dir / f"{name}.zip"

    with zipfile.ZipFile(zip_path("gutenberg")) as zf:
        for n in sorted(zf.namelist()):
            if n.startswith("gutenberg/") and n.endswith(".txt"):
                yield from _emit(f"gutenberg/{Path(n).name}", "gutenberg",
                                 strip_gutenberg(_read(zf, n)))

    with zipfile.ZipFile(zip_path("brown")) as zf:
        for n in sorted(zf.namelist()):
            if n.startswith("brown/") and n.count("/") == 1 and Path(n).name != "README":
                yield from _emit(f"brown/{Path(n).name}", "brown", brown_to_text(_read(zf, n)))

    with zipfile.ZipFile(zip_path("webtext")) as zf:
        for n in sorted(zf.namelist()):
            if n.startswith("webtext/") and n.endswith(".txt"):
                yield from _emit(f"webtext/{Path(n).name}", "webtext", _read(zf, n))

    with zipfile.ZipFile(zip_path("reuters")) as zf:
        for n in sorted(zf.namelist()):
            if n.startswith(("reuters/training/", "reuters/test/")) and not n.endswith("/"):
                yield from _emit(f"reuters/{n}", "reuters", _read(zf, n))

    with zipfile.ZipFile(zip_path("movie_reviews")) as zf:
        for n in sorted(zf.namelist()):
            if n.startswith(("movie_reviews/pos/", "movie_reviews/neg/")) and n.endswith(".txt"):
                yield from _emit(f"movie_reviews/{n}", "movie_reviews", _read(zf, n))

    with zipfile.ZipFile(zip_path("state_union")) as zf:
        for n in sorted(zf.namelist()):
            if n.startswith("state_union/") and n.endswith(".txt"):
                yield from _emit(f"state_union/{Path(n).name}", "state_union", _read(zf, n))

    with zipfile.ZipFile(zip_path("abc")) as zf:
        for n in sorted(zf.namelist()):
            if n.startswith("abc/") and n.endswith(".txt"):
                body = normalize(_read(zf, n))
                for k, chunk in enumerate(chunk_paragraphs(body)):
                    yield from _emit(f"abc/{Path(n).name}#{k}", "abc", chunk)
