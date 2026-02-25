from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import Config
from data import Vocab, download_text8, tokenize
from evaluate import run_evaluation
from train import train


def save_embeddings(model, vocab: Vocab, path: Path) -> None:
    """Save embeddings in word2vec text format"""
    emb = model.W_center
    with open(path, "w") as f:
        f.write(f"{len(vocab)} {emb.shape[1]}\n")
        for i, word in enumerate(vocab.idx2word):
            vec_str = " ".join(f"{v:.6f}" for v in emb[i])
            f.write(f"{word} {vec_str}\n")
    print(f"Embeddings saved to {path}")


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description="Word2Vec Skip-Gram with Negative Sampling")
    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--num-negatives", type=int, default=defaults.num_negatives)
    parser.add_argument("--min-count", type=int, default=defaults.min_count)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr_init)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens, help="cap token count for quick runs")
    parser.add_argument("--output", type=str, default="embeddings.txt")
    args = parser.parse_args()

    cfg = Config(
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        min_count=args.min_count,
        epochs=args.epochs,
        lr_init=args.lr,
        max_tokens=args.max_tokens,
    )

    # 1. Load and tokenize
    print("Loading text8 dataset …")
    text = download_text8()
    tokens = tokenize(text)
    print(f"Tokens: {len(tokens):,}")

    # 2. Build vocabulary
    vocab = Vocab(tokens, cfg.min_count)
    print(f"Vocabulary: {len(vocab):,} words")

    # 3. Map tokens to indices
    token_ids = np.array(
        [vocab.word2idx[t] for t in tokens if t in vocab.word2idx], dtype=np.int32
    )
    if cfg.max_tokens is not None:
        token_ids = token_ids[: cfg.max_tokens]
    print(f"Token IDs (in-vocab): {len(token_ids):,}")

    # 4. Train
    model = train(token_ids, vocab, cfg)

    # 5. Evaluate
    run_evaluation(model, vocab)

    # 6. Save
    save_embeddings(model, vocab, Path(args.output))


if __name__ == "__main__":
    main()
