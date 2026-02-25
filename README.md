# word2vec

A from-scratch implementation of Word2Vec (SkipGram with negative sampling) in pure NumPy.

## Structure

```
config.py      — Hyperparameters (embed dim, window size, negatives, LR, etc.)
data.py        — Dataset download, tokenization, vocabulary, subsampling, negative sampling table, pair generator
model.py       — SkipGram model: two embedding matrices + forward/backward/SGD update
train.py       — Training loop with linear LR decay and per-epoch subsampling
evaluate.py    — Nearest-neighbour and analogy queries on learned embeddings
main.py        — entry point
```

## Running

Install dependencies and run with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Activate the virtual environment
```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

Run the main script
```bash
uv run main.py
```

The text8 dataset is downloaded automatically on first run (~100 MB).

**Common options:**

```bash
# Full training run with custom hyperparameters
uv run main.py --embed-dim 200 --window-size 5 --num-negatives 10 --epochs 5

# Save embeddings to a custom path
uv run main.py --output my_embeddings.txt
```

All CLI flags and their defaults:

| Flag | Default | Description |
|---|---|---|
| `--embed-dim` | 100 | Embedding dimensionality |
| `--window-size` | 5 | Max context window (actual window sampled from [1, N]) |
| `--num-negatives` | 5 | Negative samples per training pair |
| `--min-count` | 5 | Minimum word frequency to include in vocabulary |
| `--epochs` | 5 | Training epochs |
| `--lr` | 0.025 | Initial learning rate |
| `--max-tokens` | None | Cap token count for quick runs |
| `--output` | `embeddings.txt` | Path for saved embeddings |

## Notes

`Word2Vec_grads.pdf` contains derivations of the gradients used in the model, along with some implementation insights.

## Design choices

**Skip-gram architecture** — predicts surrounding context words from a center word. Two separate embedding matrices are maintained: `W_center` (the final word embeddings) and `W_context` (used only during training).

**Negative sampling** — instead of a full softmax over the vocabulary, each training step contrasts one positive (center, context) pair against `k` noise words. The noise distribution is `P(w) ∝ count(w)^{3/4}`, which flattens the frequency distribution and gives rare words a better chance of being sampled as negatives. A pre-built lookup table of 10M entries makes sampling O(1).

**Subsampling of frequent words** — very common words (e.g. "the", "a") are discarded from the token stream each epoch with probability `1 - sqrt(t / f(w))` where `t = 1e-5`. This speeds up training, and improves embedding quality for content words.

**Dynamic context window** — the actual window size for each center word is sampled uniformly from `[1, window_size]` rather than being fixed. Words closer to the center are implicitly seen as context more often.

**Linear learning-rate decay** — the learning rate decreases linearly from `lr_init` to `lr_min` over the total estimated number of training steps, across all epochs.

**Text8 dataset** — a cleaned, lowercased ~100 MB slice of Wikipedia used as a standard benchmark corpus for word embedding research.

**Pure NumPy** — the entire forward pass, gradient computation, and SGD update are implemented in numpy without autograd
