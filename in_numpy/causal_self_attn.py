"""
- Compute self-attention
- Apply a mask
- Activate with softmax
- Compute weighted attention score by dot producting with V
"""

import numpy as np


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)


def softmax(scores):
    max_vals = np.max(scores, axis=-1, keepdims=True)
    x = scores - max_vals
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def masked_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # Compute self-attention
    attn_scores = (Q @ K.T) / np.sqrt(K.shape[-1])  # shape (seq_len, seq_len)

    # Apply a mask
    mask = (1 - np.tri(attn_scores.shape[-1])) * 1e-10  # shape (seq_len, seq_len)
    masked_scores = attn_scores - mask  # # shape (seq_len, seq_len)
    activations = softmax(masked_scores)  # shape (seq_len, seq_len)
    weighted_attn_scores = activations @ V  # # shape (seq_len, d_model)
    return weighted_attn_scores


np.random.seed(42)
X = np.arange(48).reshape(6, 8)
X = np.random.permutation(X.flatten()).reshape(6, 8)
mask = np.triu(np.ones((6, 6)) * (-np.inf), k=1)
W_q = np.random.randint(0, 4, size=(8, 8))
W_k = np.random.randint(0, 5, size=(8, 8))
W_v = np.random.randint(0, 6, size=(8, 8))
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
print(masked_attention(Q, K, V, mask))
