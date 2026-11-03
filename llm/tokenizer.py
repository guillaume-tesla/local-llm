"""Byte-level BPE tokenizer (HuggingFace `tokenizers`).

Byte-level BPE is lossless for any UTF-8 text, so bits-per-byte is comparable
across tokenizers of different vocabulary sizes.
"""

from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS = "<|endoftext|>"


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def word_start_mask(tok: Tokenizer) -> "list[bool]":
    """True for tokens that begin a new word (byte-level BPE marks them with a leading space)."""
    return [tok.id_to_token(i).startswith("Ġ") for i in range(tok.get_vocab_size())]
