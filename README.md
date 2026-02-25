# word2vec
Implementation of the core training loop of word2vec in pure NumPy.

# Word2Vec: Skip-Gram with Negative Sampling (Pure NumPy)

> JetBrains Internship Task — Hallucination Detection  
> Implement the core training loop of word2vec in pure NumPy (no PyTorch/TensorFlow).

---

## Project Structure

```
word2vec/
├── data.py          # Vocab, subsampling, pair generation, negative sampling
├── model.py         # Embeddings, forward, loss, gradients, update
├── train.py         # Training loop, LR schedule, logging
├── evaluate.py      # Similarity, nearest neighbors, analogies
├── config.py        # Hyperparameters
├── main.py          # Entry point
└── README.md
```

---

## Dataset

**Text8** — cleaned Wikipedia (~17M words, single file).  

---

## Implementation Plan

### 1. Data Pipeline (`data.py`)

- [ ] **Tokenization**: lowercase, strip punctuation, split on whitespace
- [ ] **Vocabulary**: build word→index / index→word mappings with `min_count` threshold (e.g., 5)
- [ ] **Subsampling frequent words**: discard word $w_i$ with probability

$$P(\text{discard}) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

where $t \approx 10^{-5}$ and $f(w_i)$ is the word's frequency ratio.

- [ ] **Training pair generation**: slide window of size `window_size` over subsampled tokens, yield `(center_idx, context_idx)` pairs. Use dynamic window (sample actual size from `[1, window_size]`) to implicitly distance-weight context.
- [ ] **Negative sampling table**: precompute smoothed unigram distribution

$$P_n(w) \propto \text{count}(w)^{3/4}$$

Sample `k` negatives per positive pair (default `k=5`).

---

### 2. Model (`model.py`)

- [ ] **Two embedding matrices**:
  - `W_center`: shape `(vocab_size, embed_dim)` — center word embeddings
  - `W_context`: shape `(vocab_size, embed_dim)` — context word embeddings
  - Init: `np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, ...)`

- [ ] **Forward pass**: given center $c$, positive context $o$, negatives $\{n_1, \dots, n_k\}$:

$$s^+ = \mathbf{v}_c \cdot \mathbf{u}_o \qquad s^-_i = \mathbf{v}_c \cdot \mathbf{u}_{n_i}$$

- [ ] **Loss** (binary cross-entropy / negative log-likelihood):

$$\mathcal{L} = -\log \sigma(s^+) - \sum_{i=1}^{k} \log \sigma(-s^-_i)$$

- [ ] **Gradients** (analytical):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = (\sigma(s^+) - 1)\,\mathbf{u}_o + \sum_{i=1}^{k} \sigma(s^-_i)\,\mathbf{u}_{n_i}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o} = (\sigma(s^+) - 1)\,\mathbf{v}_c$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{n_i}} = \sigma(s^-_i)\,\mathbf{v}_c$$

- [ ] **Parameter update**: SGD with $\theta \leftarrow \theta - \alpha \cdot \nabla_\theta \mathcal{L}$

---

### 3. Training Loop (`train.py`)

- [ ] Iterate over training pairs (true SGD or mini-batch with vectorized ops)
- [ ] **Learning rate schedule**: linear decay from `lr_init` (0.025) to `lr_min` (1e-4) over total steps
- [ ] Log average loss every N steps
- [ ] Support multiple epochs (default: 5)
- [ ] Save trained embeddings to file

---

### 4. Evaluation (`evaluate.py`)

- [ ] **Cosine similarity** between word pairs
- [ ] **Nearest neighbors**: top-k most similar words to a query
- [ ] **Analogy tests** (optional): "king - man + woman ≈ queen" using Google's `questions-words.txt`
- [ ] Use `W_center` as final word embeddings

---

### 5. Config (`config.py`)

| Hyperparameter | Default | Notes |
|---|---|---|
| `embed_dim` | 100 | Embedding dimensionality |
| `window_size` | 5 | Max context window (dynamic sampling from `[1, window_size]`) |
| `num_negatives` | 5 | Negative samples per positive pair |
| `min_count` | 5 | Min word frequency to include in vocab |
| `subsample_t` | 1e-5 | Subsampling threshold |
| `lr_init` | 0.025 | Initial learning rate |
| `lr_min` | 1e-4 | Minimum learning rate |
| `epochs` | 5 | Number of passes over the corpus |

---