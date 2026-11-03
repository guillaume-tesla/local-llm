"""Benchmark a trained run and write results/runs/<name>.json.

  python -m bench.evaluate --run runs/baseline

Metrics (lower BPB is better, higher accuracy is better):
  bpb.dev / bpb.test           bits per UTF-8 byte on held-out documents. Tokenizer-agnostic,
                               so runs with different vocab sizes compare directly.
  bpb.test_by_source           the same, split by corpus (gutenberg, brown, reuters, ...).
  cloze.accuracy               continuation ranking on test text (chance 0.25).
  generation.distinct_2        diversity of sampled text (unique bigrams / bigrams).
  efficiency                   parameters, training FLOPs, train and inference tokens/s, RAM.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from bench.cloze import build_items, score_items
from llm.common import load_meta, pick_device, sync, tune_malloc
from llm.model import LLM, ModelConfig
from llm.tokenizer import word_start_mask

LN2 = math.log(2)


def _windows(M: int, ctx: int, stride: int) -> list[tuple[int, int, int]]:
    """Sliding windows over M prediction targets. Returns (start, end, first_scored_target).

    Every target is scored exactly once, and each target sees at least ctx - stride tokens
    of context (or all of its preceding context if it is in the first window).
    """
    if M <= ctx:
        return [(0, M, 0)]
    starts = sorted(set(list(range(0, M - ctx, stride)) + [M - ctx]))
    out, prev = [], 0
    for s in starts:
        e = s + ctx
        out.append((s, e, prev))
        prev = e
    return out


@torch.no_grad()
def stream_nll(model: LLM, toks: np.ndarray, ctx: int, stride: int, device: torch.device,
               batch: int = 8) -> np.ndarray:
    """Per-target NLL in nats. Target i predicts toks[i + 1]."""
    M = len(toks) - 1
    nll = np.zeros(M, dtype=np.float64)
    wins = _windows(M, ctx, stride)
    for b in range(0, len(wins), batch):
        group = wins[b : b + batch]
        W = min(ctx, M)
        x = torch.from_numpy(np.stack([toks[s : s + W] for s, _, _ in group]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([toks[s + 1 : s + 1 + W] for s, _, _ in group]).astype(np.int64)).to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none").view(len(group), W).cpu().numpy()
        for row, (s, e, lo) in enumerate(group):
            lo = max(lo, s)
            nll[lo:e] = loss[row, lo - s : e - s]
    return nll


def split_bpb(model, data_dir: Path, meta: dict, split: str, device, ctx: int, stride: int) -> dict:
    toks = np.fromfile(data_dir / f"{split}.bin", dtype=np.uint16).astype(np.int64)
    src = np.load(data_dir / f"{split}.src.npy")
    nll = stream_nll(model, toks, ctx, stride, device)
    split_meta = meta["splits"][split]
    bpb = float(nll.sum() / LN2 / split_meta["bytes"])
    tgt_src = src[1:]
    by_source = {}
    for sid, name in enumerate(meta["sources"]):
        mask = tgt_src == sid
        if mask.any():
            by_source[name] = float(nll[mask].sum() / LN2 / split_meta["by_source"][name]["bytes"])
    return {"bpb": round(bpb, 4), "tokens": int(len(toks)), "bytes": split_meta["bytes"],
            "nll_per_token": round(float(nll.mean()), 4),
            "by_source": {k: round(v, 4) for k, v in by_source.items()}}


@torch.no_grad()
def sample(model: LLM, prompts: np.ndarray, n_new: int, device, temperature: float = 0.8,
           top_p: float = 0.95, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    ctx = model.cfg.ctx_len
    ids = torch.from_numpy(prompts).to(device)
    P = ids.size(1)
    for _ in range(n_new):
        logits, _ = model(ids[:, -ctx:])
        probs = F.softmax(logits[:, -1].float() / temperature, dim=-1)
        sorted_p, order = probs.sort(descending=True, dim=-1)
        keep = (sorted_p.cumsum(-1) - sorted_p) <= top_p
        sorted_p = sorted_p * keep
        sorted_p = sorted_p / sorted_p.sum(-1, keepdim=True)
        pick = torch.multinomial(sorted_p, 1)
        ids = torch.cat([ids, order.gather(-1, pick)], dim=1)
    return ids[:, P:].cpu().numpy()


def distinct_2(seqs: np.ndarray) -> float:
    grams = [tuple(s[i : i + 2]) for s in seqs for i in range(len(s) - 1)]
    return len(set(grams)) / max(1, len(grams))


@torch.no_grad()
def inference_tok_per_s(model: LLM, device, bs: int = 8, iters: int = 5) -> float:
    x = torch.randint(0, model.cfg.vocab_size, (bs, model.cfg.ctx_len), device=device)
    model(x)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    sync(device)
    return bs * model.cfg.ctx_len * iters / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="e.g. runs/baseline")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride", type=int, default=0, help="eval window stride (default ctx/2)")
    ap.add_argument("--cloze-items", type=int, default=1000)
    ap.add_argument("--results", type=Path, default=Path("results/runs"))
    args = ap.parse_args()

    tune_malloc()
    summary = json.loads((args.run / "summary.json").read_text())
    ck = torch.load(args.run / "ckpt.pt", map_location="cpu", weights_only=True)
    mcfg = ModelConfig(**ck["model_config"])
    device = pick_device(args.device)
    model = LLM(mcfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    data_dir = Path(summary["train_config"]["data"])
    meta = load_meta(data_dir)
    tok = Tokenizer.from_file(str(data_dir / "tokenizer.json"))
    ctx = mcfg.ctx_len
    stride = args.stride or ctx // 2
    print(f"[{args.run.name}] device={device} ctx={ctx} stride={stride}", flush=True)

    bpb = {split: split_bpb(model, data_dir, meta, split, device, ctx, stride) for split in ("dev", "test")}
    print("bpb:", {k: v["bpb"] for k, v in bpb.items()}, flush=True)

    test_tokens = np.fromfile(data_dir / "test.bin", dtype=np.uint16).astype(np.int64)
    mask = np.array(word_start_mask(tok), dtype=bool)
    items = build_items(test_tokens, mask, meta["eos"], n_items=args.cloze_items)
    cloze = score_items(model, items, device)
    print("cloze:", cloze, flush=True)

    starts = np.linspace(0, len(test_tokens) - 64 - 1, 16).astype(int)
    prompts = np.stack([test_tokens[s : s + 32] for s in starts])
    gen = sample(model, prompts, n_new=96, device=device)
    samples = [tok.decode(g.tolist()) for g in gen[:3]]
    gen_stats = {"distinct_2": round(distinct_2(gen), 4), "n_samples": int(len(gen)),
                 "new_tokens": int(gen.shape[1]), "examples": [s[:400] for s in samples]}

    result = {
        "name": args.run.name,
        "eval": {"device": str(device), "ctx": ctx, "stride": stride,
                 "bpb": {"dev": bpb["dev"]["bpb"], "test": bpb["test"]["bpb"]},
                 "bpb_detail": bpb,
                 "cloze": cloze,
                 "generation": gen_stats,
                 "infer_tok_per_s": round(inference_tok_per_s(model, device), 1)},
        "model": {"params": summary["params"], "params_non_embedding": summary["params_non_embedding"],
                  "config": summary["model_config"]},
        "train": {"steps": summary["steps"], "tokens_seen": summary["tokens_seen"],
                  "wall_s": summary["train_wall_s"], "tok_per_s": summary["train_tok_per_s"],
                  "flops_est": summary["train_flops_est"], "peak_rss_mb": summary["peak_rss_mb"],
                  "final_train_loss": summary["final_train_loss"], "config": summary["train_config"]},
        "env": {"device": summary["device"], "torch": summary["torch"], "platform": summary["platform"],
                "cpu_threads": summary["cpu_threads"]},
    }
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / f"{args.run.name}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
