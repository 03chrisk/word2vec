from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: clip to prevent exp() overflow."""
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class SkipGram:
    """Two-matrix skip-gram model trained with negative sampling via SGD."""

    def __init__(self, vocab_size: int, embed_dim: int):
        # Center and context embeddings
        bound = 0.5 / embed_dim
        self.W_center = np.random.uniform(-bound, bound, (vocab_size, embed_dim))
        self.W_context = np.random.uniform(-bound, bound, (vocab_size, embed_dim))

    def train_pair(
        self,
        center: int,
        context: int,
        negatives: np.ndarray,
        lr: float,
    ) -> float:
        """Run forward + backward + SGD update for one (center, context) pair.
        Returns the scalar loss for this pair.
        """
        # center word embedding
        v_c = self.W_center[center]  # (D,)

        ctx_ids = np.empty(1 + len(negatives), dtype=np.int32)
        ctx_ids[0] = context
        ctx_ids[1:] = negatives
        
        # context embeddings for positive and negatives
        u = self.W_context[ctx_ids]  # (1+K, D)

        # ---- forward ----
        scores = u @ v_c  # (1+K,)
        sig = sigmoid(scores)

        # Labels: 1 for positive, 0 for negatives
        # grad coefficient = sigmoid(s) - label
        grad_coeff = sig.copy()
        grad_coeff[0] -= 1.0  # positive: sigmoid(s+) - 1

        # ---- loss ----
        # L = -log sigmoid(s+) - sum( log sigmoid(-s_i) )
        eps = 1e-12
        loss = -np.log(sig[0] + eps) - np.log(1.0 - sig[1:] + eps).sum()

        # ---- gradients ----
        # dL/dv_c = sum_j( grad_coeff_j * u_j )
        grad_center = grad_coeff @ u  # (D,)
        # dL/du_j = grad_coeff_j * v_c
        grad_context = grad_coeff[:, None] * v_c[None, :]  # (1+K, D)

        # ---- SGD update ----
        self.W_center[center] -= lr * grad_center
        self.W_context[ctx_ids] -= lr * grad_context

        return float(loss)
