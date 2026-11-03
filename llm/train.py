"""Train a model from a JSON config.

  python -m llm.train --config configs/baseline.json --name baseline

Writes runs/<name>/{ckpt.pt, log.jsonl, summary.json}. Benchmark with bench/evaluate.py.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from llm.common import cpu_threads, load_meta, peak_rss_mb, pick_device, set_seed, sync, tune_malloc
from llm.model import LLM, ModelConfig


@dataclass
class TrainConfig:
    data: str = "data/processed/tok8192"
    batch_size: int = 32
    seq_len: int = 256
    tokens_budget: int = 8_000_000
    lr: float = 2e-3
    min_lr_frac: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_windows: int = 64
    log_every: int = 25
    seed: int = 1337
    amp: str = "none"  # none | bf16 (only used on cuda / mps)
    compile: bool = True  # torch.compile: ~15% faster steps on CPU, ~30 s warmup


class Batcher:
    """Samples random contiguous windows from a flat token file."""

    def __init__(self, path: Path, seq_len: int, batch_size: int, seed: int) -> None:
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len, self.batch_size = seq_len, batch_size
        self.rng = np.random.default_rng(seed)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        x = np.stack([self.data[i : i + self.seq_len] for i in starts]).astype(np.int64)
        y = np.stack([self.data[i + 1 : i + 1 + self.seq_len] for i in starts]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def fixed_windows(path: Path, seq_len: int, n: int) -> torch.Tensor:
    """A fixed set of dev windows (seq_len + 1 tokens each) for cheap, comparable monitoring."""
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(data) - seq_len - 1, size=n)
    return torch.from_numpy(
        np.stack([data[i : i + seq_len + 1] for i in starts]).astype(np.int64)
    )


def make_optimizer(model: LLM, cfg: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        # Matrices get weight decay; embeddings, norms and other vectors do not.
        (decay if p.dim() >= 2 and "tok_emb" not in name else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8,
    )


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to min_lr_frac * lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    min_lr = cfg.lr * cfg.min_lr_frac
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def monitor_nll(model: LLM, windows: torch.Tensor, device: torch.device) -> float:
    """Mean per-token NLL (nats) on the fixed dev windows."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(windows), 16):
        chunk = windows[i : i + 16]
        x, y = chunk[:, :-1].to(device), chunk[:, 1:].to(device)
        _, loss = model(x, y)
        total += loss.item() * y.numel()
        count += y.numel()
    model.train(was_training)
    return total / count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--max-steps", type=int, default=0, help="debug: cap the number of steps")
    args = ap.parse_args()

    raw = json.loads(Path(args.config).read_text())
    tcfg = TrainConfig(**raw.get("train", {}))
    meta = load_meta(tcfg.data)
    mcfg = ModelConfig(vocab_size=meta["vocab"], **raw.get("model", {}))
    assert tcfg.seq_len <= mcfg.ctx_len, "train seq_len must fit in the model context"

    set_seed(tcfg.seed)
    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(cpu_threads())
        tune_malloc()

    model = LLM(mcfg).to(device)
    n_params = model.num_params()
    opt = make_optimizer(model, tcfg)
    fwd = torch.compile(model) if tcfg.compile else model
    amp = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
           if tcfg.amp == "bf16" and device.type in ("cuda", "mps")
           else contextlib.nullcontext())

    data = Path(tcfg.data)
    batcher = Batcher(data / "train.bin", tcfg.seq_len, tcfg.batch_size, tcfg.seed)
    monitor = fixed_windows(data / "dev.bin", tcfg.seq_len, tcfg.eval_windows)

    tokens_per_step = tcfg.batch_size * tcfg.seq_len
    total_steps = max(1, tcfg.tokens_budget // tokens_per_step)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    run_dir = args.out / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / "log.jsonl").open("w")

    def record(row: dict) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    print(f"{args.name}: {n_params / 1e6:.2f}M params, {total_steps} steps x {tokens_per_step} tokens "
          f"on {device}", flush=True)
    train_time, recent = 0.0, []
    model.train()
    for step in range(total_steps):
        t0 = time.perf_counter()
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total_steps, tcfg)
        x, y = batcher.get()
        x, y = x.to(device), y.to(device)
        with amp:
            _, loss = fwd(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val = loss.item()
        sync(device)
        train_time += time.perf_counter() - t0
        recent.append(loss_val)

        done = step + 1
        if done % tcfg.log_every == 0 or step == 0:
            record({"step": done, "loss": round(loss_val, 4), "lr": round(opt.param_groups[0]["lr"], 6),
                    "tokens": done * tokens_per_step,
                    "tok_per_s": round(done * tokens_per_step / train_time)})
        if done % tcfg.eval_every == 0 or done == total_steps:
            record({"step": done, "dev_nll_nats": round(monitor_nll(model, monitor, device), 4)})

    dev_nll = monitor_nll(model, monitor, device)
    torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                "train_config": asdict(tcfg)}, run_dir / "ckpt.pt")

    tokens_seen = total_steps * tokens_per_step
    summary = {
        "name": args.name,
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "params": n_params,
        "params_non_embedding": model.num_non_embedding_params(),
        "steps": total_steps,
        "tokens_seen": tokens_seen,
        "train_wall_s": round(train_time, 2),
        "train_tok_per_s": round(tokens_seen / train_time, 1),
        "train_flops_est": float(model.flops_per_token(tcfg.seq_len) * tokens_seen),
        "final_train_loss": round(float(np.mean(recent[-50:])), 4),
        "final_dev_nll_nats": round(dev_nll, 4),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "model_config": asdict(mcfg),
        "train_config": asdict(tcfg),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("name", "params", "tokens_seen", "train_wall_s",
                                              "train_tok_per_s", "final_dev_nll_nats")}), flush=True)


if __name__ == "__main__":
    main()
