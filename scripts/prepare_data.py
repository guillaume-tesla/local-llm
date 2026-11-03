"""Turn raw corpora into tokenized, split-ready binaries.

Output (data/processed/tok<V>/):
  tokenizer.json           byte-level BPE trained on the train split only
  train.bin dev.bin test.bin   uint16 token ids, <|endoftext|> between documents
  *.src.npy                uint8 source id per token (for per-corpus metrics)
  meta.json                vocab, sources, per-split token/byte/doc counts
  docs.jsonl               one line per document: id, source, split, bytes, tokens

Usage:  python -m scripts.prepare_data --vocab 8192
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm.corpus import iter_docs, split_of
from llm.tokenizer import EOS, train_tokenizer

SPLITS = ("train", "dev", "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--vocab", type=int, default=8192)
    args = ap.parse_args()

    out_dir = args.out / f"tok{args.vocab}"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = [(d, split_of(d.doc_id)) for d in iter_docs(args.raw)]
    sources = sorted({d.source for d, _ in docs})
    src_index = {s: i for i, s in enumerate(sources)}
    print(f"{len(docs)} documents from {len(sources)} sources")

    tok = train_tokenizer((d.text for d, s in docs if s == "train"), args.vocab)
    tok.save(str(out_dir / "tokenizer.json"))
    eos = tok.token_to_id(EOS)
    print(f"tokenizer: {tok.get_vocab_size()} tokens, eos id {eos}")

    meta = {
        "vocab": tok.get_vocab_size(),
        "eos": eos,
        "sources": sources,
        "splits": {},
    }
    manifest = []

    for split in SPLITS:
        split_docs = [d for d, s in docs if s == split]
        encodings = tok.encode_batch([d.text for d in split_docs], add_special_tokens=False)

        ids_parts, src_parts = [], []
        per_source = defaultdict(lambda: {"tokens": 0, "bytes": 0, "docs": 0})
        total_bytes = 0
        for doc, enc in zip(split_docs, encodings):
            ids = np.asarray(enc.ids + [eos], dtype=np.uint16)
            nbytes = len(doc.text.encode("utf-8"))
            ids_parts.append(ids)
            src_parts.append(np.full(len(ids), src_index[doc.source], dtype=np.uint8))
            total_bytes += nbytes
            stats = per_source[doc.source]
            stats["tokens"] += len(ids)
            stats["bytes"] += nbytes
            stats["docs"] += 1
            manifest.append({"id": doc.doc_id, "source": doc.source, "split": split,
                             "bytes": nbytes, "tokens": len(ids)})

        ids_all = np.concatenate(ids_parts)
        ids_all.tofile(out_dir / f"{split}.bin")
        np.save(out_dir / f"{split}.src.npy", np.concatenate(src_parts))
        meta["splits"][split] = {
            "tokens": int(len(ids_all)),
            "bytes": int(total_bytes),
            "docs": len(split_docs),
            "by_source": dict(per_source),
        }
        print(f"{split:5s}: {len(split_docs):6d} docs  {len(ids_all):10,d} tokens  "
              f"{total_bytes:12,d} bytes  ({total_bytes / len(ids_all):.2f} bytes/token)")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with (out_dir / "docs.jsonl").open("w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
