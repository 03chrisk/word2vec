from __future__ import annotations

import numpy as np

from data import Vocab
from model import SkipGram


def get_embeddings(model: SkipGram) -> np.ndarray:
    """Return L2-normalised center embeddings."""
    emb = model.W_center.copy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms


def cosine_similarity(emb: np.ndarray, i: int, j: int) -> float:
    """Cosine similarity between two words (assuming embeddings are normalised)"""
    return float(emb[i] @ emb[j])


def nearest_neighbours(
    word: str, vocab: Vocab, emb: np.ndarray, k: int = 10
) -> list[tuple[str, float]]:
    """Return the k most similar words to `word`."""
    if word not in vocab.word2idx:
        print(f"'{word}' not in vocabulary")
        return []
    idx = vocab.word2idx[word]
    sims = emb @ emb[idx]  # cosine sim (embeddings already normalised)
    # Exclude the query word itself
    sims[idx] = -np.inf
    top = np.argsort(sims)[::-1][:k]
    return [(vocab.idx2word[i], float(sims[i])) for i in top]


def analogy(
    a: str, b: str, c: str, vocab: Vocab, emb: np.ndarray, k: int = 5
) -> list[tuple[str, float]]:
    """Computes: vec(b) - vec(a) + vec(c) and returns nearest words."""
    for w in (a, b, c):
        if w not in vocab.word2idx:
            print(f"'{w}' not in vocabulary")
            return []

    vec = emb[vocab.word2idx[b]] - emb[vocab.word2idx[a]] + emb[vocab.word2idx[c]]
    vec = vec / (np.linalg.norm(vec) + 1e-12)

    sims = emb @ vec
    exclude = {vocab.word2idx[a], vocab.word2idx[b], vocab.word2idx[c]}
    for idx in exclude:
        sims[idx] = -np.inf
    top = np.argsort(sims)[::-1][:k]
    return [(vocab.idx2word[i], float(sims[i])) for i in top]


def run_evaluation(model: SkipGram, vocab: Vocab) -> None:
    """Run a quick qualitative evaluation and print results."""
    emb = get_embeddings(model)

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Nearest neighbours for a few probe words
    probes = ["king", "computer", "france", "dog", "good"]
    for word in probes:
        neighbours = nearest_neighbours(word, vocab, emb, k=8)
        if neighbours:
            nns = ", ".join(f"{w} ({s:.3f})" for w, s in neighbours)
            print(f"\n  {word}  →  {nns}")

    # Analogy tests
    print("\nAnalogies:")
    analogy_tests = [
        ("king", "queen", "man"),     # king - queen + man ≈ woman
        ("paris", "france", "berlin"),  # paris - france + berlin ≈ germany
        ("good", "better", "bad"),      # good - better + bad ≈ worse
    ]
    for a, b, c in analogy_tests:
        results = analogy(a, b, c, vocab, emb, k=5)
        if results:
            ans = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"  {a} - {b} + {c}  →  {ans}")
    print()
