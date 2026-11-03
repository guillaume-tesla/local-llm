"""Continuation-ranking benchmark built from held-out text (no external dataset needed).

Each item: a 64-token prefix from the test split, plus 4 candidate 16-token continuations.
One is the real continuation; three are taken from other places in the test split.
A model scores each candidate by its mean log-probability given the prefix. Chance = 25%.
Items are generated from a fixed seed, so every model sees exactly the same questions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from llm.model import LLM


def build_items(
    tokens: np.ndarray,
    word_start: np.ndarray,
    eos: int,
    n_items: int = 1000,
    prefix_len: int = 64,
    cont_len: int = 16,
    n_choices: int = 4,
    seed: int = 1234,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    span = prefix_len + cont_len
    limit = len(tokens) - span - 1
    items: list[dict] = []

    def candidate(pos: int) -> np.ndarray | None:
        cand = tokens[pos + prefix_len : pos + span]
        if not word_start[tokens[pos + prefix_len]] or (cand == eos).any():
            return None
        return cand

    while len(items) < n_items:
        p = int(rng.integers(0, limit))
        if (tokens[p : p + span] == eos).any():
            continue
        gold = candidate(p)
        if gold is None:
            continue
        choices = [gold]
        while len(choices) < n_choices:
            q = int(rng.integers(0, limit))
            if abs(q - p) < 2 * span:
                continue
            cand = candidate(q)
            if cand is None or any(np.array_equal(cand, c) for c in choices):
                continue
            choices.append(cand)
        order = rng.permutation(n_choices)
        label = int(np.where(order == 0)[0][0])
        items.append({
            "prefix": tokens[p : p + prefix_len].astype(np.int64),
            "choices": [choices[i].astype(np.int64) for i in order],
            "label": label,
        })
    return items


@torch.no_grad()
def score_items(model: LLM, items: list[dict], device: torch.device, batch_items: int = 8) -> dict:
    model.eval()
    correct = 0
    for i in range(0, len(items), batch_items):
        chunk = items[i : i + batch_items]
        seqs = [np.concatenate([it["prefix"], ch]) for it in chunk for ch in it["choices"]]
        x = torch.from_numpy(np.stack(seqs)).to(device)
        P = len(chunk[0]["prefix"])
        logits, _ = model(x[:, :-1])
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        cont_lp = tok_lp[:, P - 1 :]  # log-probs of the candidate tokens only
        scores = cont_lp.mean(dim=1).view(len(chunk), -1)
        preds = scores.argmax(dim=1).cpu().numpy()
        correct += sum(int(p == it["label"]) for p, it in zip(preds, chunk))
    return {"accuracy": correct / len(items), "n_items": len(items), "chance": 1 / 4}
