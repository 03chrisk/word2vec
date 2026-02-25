from dataclasses import dataclass


@dataclass
class Config:
    embed_dim: int = 100

    # Context window
    window_size: int = 5  # max window, actual sampled from [1, window_size]

    # Negative sampling
    num_negatives: int = 5

    # Vocabulary
    min_count: int = 5
    subsample_t: float = 1e-5

    # Training
    lr_init: float = 0.025
    lr_min: float = 1e-4
    epochs: int = 5
    log_every: int = 10_000_00

    # Dataset
    max_tokens: int | None = None  # Set to a smaller number for quick runs

    # Negative sampling table size
    neg_table_size: int = 10_000_000
