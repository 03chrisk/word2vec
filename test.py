import numpy as np
from types import SimpleNamespace
from evaluate import nearest_neighbours, analogy

# Load embeddings from file
words, vecs = [], []
with open("embeddings.txt") as f:
    n, dim = map(int, f.readline().split())
    for line in f:
        parts = line.split()
        words.append(parts[0])
        vecs.append(list(map(float, parts[1:])))

vocab = SimpleNamespace(word2idx={w: i for i, w in enumerate(words)}, idx2word=words)
emb = np.array(vecs, dtype=np.float32)
# Normalize each (row) embedding to unit length for cosine similarity
emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)

print("=" * 60)
print("EVALUATION (from embeddings.txt)")
print("=" * 60)

print("\nNearest neighbours:")
for w in ["king", "computer", "france", "dog", "good"]:
    results = nearest_neighbours(w, vocab, emb, k=8)
    if results:
        print(f"  {w} → " + ", ".join(f"{word} ({s:.3f})" for word, s in results))

print("\nAnalogies:")
for a, b, c in [("king", "queen", "man"), ("paris", "france", "berlin"), ("good", "better", "bad")]:
    results = analogy(a, b, c, vocab, emb, k=5)
    if results:
        print(f"  {a} - {b} + {c} → " + ", ".join(f"{w} ({s:.3f})" for w, s in results))