from __future__ import annotations

import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Generator

import numpy as np

from config import Config


# ---------------------------------------------------------------------------
# Loading & tokenization
# ---------------------------------------------------------------------------

DATA_URL = "http://mattmahoney.net/dc/text8.zip"
DATA_DIR = Path("data")


def download_text8() -> str:
    """Download and extract the text8 dataset, returning its text content."""
    DATA_DIR.mkdir(exist_ok=True)
    txt_path = DATA_DIR / "text8"
    if txt_path.exists():
        return txt_path.read_text()

    zip_path = DATA_DIR / "text8.zip"
    if not zip_path.exists():
        print("Downloading text8 …")
        urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    return txt_path.read_text()


def tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alpha, split on whitespace."""
    return re.findall(r"[a-z]+", text.lower())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """Word to index mapping with frequency counts."""

    def __init__(self, tokens: list[str], min_count: int):
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1

        # Keep only words meeting the minimum count
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.counts: list[int] = []

        for word, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count < min_count:
                continue
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.counts.append(count)

        self.counts_arr = np.array(self.counts, dtype=np.float64)
        self.total = self.counts_arr.sum()

    def __len__(self) -> int:
        return len(self.idx2word)


# ---------------------------------------------------------------------------
# Subsampling of frequent words
# ---------------------------------------------------------------------------
def subsample(token_ids: np.ndarray, vocab: Vocab, t: float) -> np.ndarray:
    """Discard frequent words with probability 1 - sqrt(t / f(w))."""
    freqs = vocab.counts_arr[token_ids] / vocab.total
    keep_prob = np.sqrt(t / freqs)
    rand = np.random.rand(len(token_ids))
    return token_ids[rand < keep_prob]

# ---------------------------------------------------------------------------
# Negative sampling table
# ---------------------------------------------------------------------------
def build_neg_table(vocab: Vocab, table_size: int) -> np.ndarray:
    """Build a large table for fast negative sampling using P_n(w) ∝ count^{3/4}."""
    power = np.power(vocab.counts_arr, 0.75)
    normalised = power / power.sum()
    table = np.zeros(table_size, dtype=np.int32)
    cum = np.cumsum(normalised)

    idx = 0
    for i in range(table_size):
        while idx < len(cum) - 1 and i / table_size > cum[idx]:
            idx += 1
        table[i] = idx
    return table


def sample_negatives(
    neg_table: np.ndarray, count: int, exclude: int
) -> np.ndarray:
    """Sample `count` negatives from the table, excluding `exclude`."""
    negs = np.empty(count, dtype=np.int32)
    drawn = 0
    while drawn < count:
        candidates = neg_table[np.random.randint(0, len(neg_table), size=count - drawn)]
        for c in candidates:
            if c != exclude and drawn < count:
                negs[drawn] = c
                drawn += 1
    return negs


# ---------------------------------------------------------------------------
# Training-pair generator
# ---------------------------------------------------------------------------
def generate_pairs(
    token_ids: np.ndarray, window_size: int
) -> Generator[tuple[int, int], None, None]:
    """Yield (center, context) pairs with dynamic window sizing."""
    n = len(token_ids)
    for i in range(n):
        # Dynamic window: sample actual window from [1, window_size]
        w = np.random.randint(1, window_size + 1)
        start = max(0, i - w)
        end = min(n, i + w + 1)
        center = token_ids[i]
        for j in range(start, end):
            if j != i:
                yield int(center), int(token_ids[j])
