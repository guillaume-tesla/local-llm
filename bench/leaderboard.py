"""Collect results/runs/*.json into results/LEADERBOARD.md, best test BPB first.

  python -m bench.leaderboard
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("results/runs")
OUT = Path("results/LEADERBOARD.md")


def bytes_per_token(data_dir: str) -> float:
    """Average UTF-8 bytes per token of the train split for this tokenizer."""
    split = json.loads((Path(data_dir) / "meta.json").read_text())["splits"]["train"]
    return split["bytes"] / split["tokens"]


def row(r: dict) -> str:
    e, m, t = r["eval"], r["model"], r["train"]
    cfg = m["config"]
    arch = f"L{cfg['n_layer']} d{cfg['d_model']} h{cfg['n_head']}/{cfg['n_kv_head']}"
    bpt = bytes_per_token(t["config"]["data"])
    text_mb = t["tokens_seen"] * bpt / 1e6
    return (f"| {r['name']} | {arch} | {m['params'] / 1e6:.2f} | {t['tokens_seen'] / 1e6:.2f} | "
            f"{text_mb:.1f} | "
            f"{e['bpb']['test']:.4f} | {e['bpb']['dev']:.4f} | {e['cloze']['accuracy'] * 100:.1f} | "
            f"{e['generation']['distinct_2']:.3f} | {t['flops_est'] / 1e12:.2f} | "
            f"{t['tok_per_s']:.0f} | {e['infer_tok_per_s']:.0f} | {t['peak_rss_mb']:.0f} |")


def main() -> None:
    results = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    results.sort(key=lambda r: r["eval"]["bpb"]["test"])
    lines = [
        "# Leaderboard",
        "",
        "Lower BPB is better. BPB = bits per UTF-8 byte on held-out documents (tokenizer-agnostic).",
        "Cloze = 4-way continuation ranking on test text (chance 25%).",
        "",
        "Budgets are fixed in tokens, but tokenizers differ in bytes per token. 'text seen' is the",
        "UTF-8 text the run actually trained on, so compare runs at equal text seen as well as",
        "equal tokens (and equal FLOPs) before crediting a tokenizer or architecture.",
        "",
        "| run | arch | params (M) | train tokens (M) | text seen (MB) | test BPB | dev BPB | cloze % | "
        "distinct-2 | train TFLOPs | train tok/s | infer tok/s | RSS (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [row(r) for r in results]
    lines.append("")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
