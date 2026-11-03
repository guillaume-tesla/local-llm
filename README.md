# local-llm

Small, efficient language models trained from scratch on a CPU-only box, with a benchmark
suite so every change is measured against the same held-out text. The point is to iterate:
change one thing, run the benchmark, keep it only if the numbers move.

This first step runs on a 4-vCPU Xeon (AVX-512, no GPU, 15 GB RAM). The same code is meant
to run unchanged on an Apple Silicon Mac with 24 GB (PyTorch uses MPS automatically).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python -m scripts.fetch_data                    # ~30 MB of English, sha256-verified
python -m scripts.prepare_data --vocab 8192     # BPE tokenizer + train/dev/test binaries

python -m llm.train --config configs/ablation/base.json --name base   # train
python -m bench.evaluate --run runs/base                              # benchmark
python -m bench.leaderboard                                           # results/LEADERBOARD.md

python -m unittest discover -s tests -t .       # unit tests
```

Sweeps: `PY=.venv/bin/python scripts/sweep.sh configs/ablation/*.json` runs train + benchmark for
each config, skips runs that already have a result, and refreshes the leaderboard.

## Data

| source | what | docs |
|---|---|---|
| gutenberg | 18 public-domain books (Austen, Shakespeare, Melville, KJV Bible, ...) | 18 |
| brown | Brown Corpus, 1961 American English (POS tags stripped) | 500 |
| reuters | Reuters newswire | ~10k |
| movie_reviews | movie reviews | 2000 |
| state_union | State of the Union addresses | 65 |
| abc | ABC Australia news (chunked into pseudo-documents) | ~ |
| webtext | forum, chat and script text | 7 |

Everything comes from NLTK data packages (`github.com/nltk/nltk_data`). Hashes are pinned in
`scripts/fetch_data.py`.

**Splits are by document**, decided by a stable hash of the document id: 5% dev, 5% test, 90%
train. A whole book or article is never split across sets. Dev is for choosing settings. Test
is for reporting milestones, so try not to tune on it.

Tokenizer: byte-level BPE trained on the train split only. Vocab size is a config choice.

| vocab | train tokens | bytes per token (train) |
|---|---|---|
| 8192 | 10.4 M | 3.69 |

## Model

`llm/model.py`: decoder-only Transformer in the Llama style: pre-norm RMSNorm, RoPE, SwiGLU,
optional grouped-query attention and QK-norm, tied embeddings, no biases. The causal test in
`tests/test_model.py` checks that future tokens cannot affect past predictions for every
variant.

## Training

`llm/train.py`. A JSON config holds the model shape and the training recipe. Training is set
by a **token budget**, not by steps or epochs, so runs with different batch sizes are compared
at the same amount of data. AdamW (β=0.9/0.95, weight decay 0.1 on matrices only), linear
warmup, cosine decay to 10% of peak, grad clip 1.0. Compiled with `torch.compile`.

## Benchmark

`bench/evaluate.py` writes `results/runs/<name>.json`. Metrics:

- **BPB (bits per byte), dev and test.** Negative log-likelihood of held-out text, divided by
  the UTF-8 byte count of that text. This does not depend on the tokenizer, so runs with
  different vocabularies are directly comparable. Lower is better. Scored with sliding windows
  (stride ctx/2), so every token sees at least half a context.
- **BPB by source.** The same, split by corpus, to show where a model is weak.
- **Cloze accuracy.** 1000 fixed items from the test split. Each has a 64-token prefix and four
  16-token continuations: the real one and three taken from elsewhere. The model picks the
  candidate with the highest mean log-probability. Chance is 25%.
- **distinct-2.** Diversity of 16 sampled 96-token continuations (unique bigrams / bigrams).
  A cheap check against degenerate repetition.
- **Efficiency.** Parameter count, training FLOPs estimate (6·N·tokens plus attention),
  training and inference tokens/s on the current device, peak RSS.

The leaderboard is in `results/LEADERBOARD.md`.

## Results

See `results/LEADERBOARD.md` for the full table. Summary of the first sweep is at the end of
this file.

## Layout

```
llm/        model, training loop, data and tokenizer code, device helpers
bench/      evaluation: BPB, cloze, generation, efficiency; leaderboard
scripts/    fetch_data, prepare_data, sweep
configs/    experiment configs (configs/ablation = first sweep)
results/    committed benchmark outputs (runs/*.json, LEADERBOARD.md)
tests/      unit tests (model causality and shapes, corpus parsing, split rules)
data/, runs/  generated, gitignored
```
