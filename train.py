from __future__ import annotations

import time

import numpy as np

from config import Config
from data import (
    Vocab,
    build_neg_table,
    generate_pairs,
    sample_negatives,
    subsample,
)
from model import SkipGram


def estimate_total_pairs(n_tokens: int, window_size: int) -> int:
    """Rough estimate of training pairs per epoch (for LR scheduling)."""
    # Average dynamic window = (1 + window_size) / 2, each center yields around 2*avg neighbours
    avg_window = (1 + window_size) / 2
    return int(n_tokens * 2 * avg_window)


def train(
    tokens: np.ndarray,
    vocab: Vocab,
    cfg: Config,
) -> SkipGram:
    """Train a SkipGram model on the given token-id sequence."""
    model = SkipGram(len(vocab), cfg.embed_dim)
    neg_table = build_neg_table(vocab, cfg.neg_table_size)

    est_pairs_per_epoch = estimate_total_pairs(len(tokens), cfg.window_size)
    total_steps = est_pairs_per_epoch * cfg.epochs

    global_step = 0
    running_loss = 0.0

    for epoch in range(1, cfg.epochs + 1):
        # Subsample frequent words each epoch
        sub_ids = subsample(tokens, vocab, cfg.subsample_t)
        print(f"Epoch {epoch}/{cfg.epochs}  |  tokens after subsampling: {len(sub_ids):,}")

        t0 = time.time()
        epoch_loss = 0.0
        epoch_steps = 0

        for center, context in generate_pairs(sub_ids, cfg.window_size):
            # Linear LR decay
            progress = global_step / total_steps
            lr = max(cfg.lr_init * (1.0 - progress), cfg.lr_min)

            negs = sample_negatives(neg_table, cfg.num_negatives, context)
            loss = model.train_pair(center, context, negs, lr)

            running_loss += loss
            epoch_loss += loss
            global_step += 1
            epoch_steps += 1

            if global_step % cfg.log_every == 0:
                avg = running_loss / cfg.log_every
                elapsed = time.time() - t0
                pairs_sec = epoch_steps / elapsed if elapsed > 0 else 0
                print(
                    f"  step {global_step:>10,}  |  lr {lr:.6f}  |  "
                    f"loss {avg:.4f}  |  {pairs_sec:,.0f} pairs/s"
                )
                running_loss = 0.0

        epoch_avg = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch} done — avg loss {epoch_avg:.4f}  ({time.time() - t0:.1f}s)\n")

    return model
