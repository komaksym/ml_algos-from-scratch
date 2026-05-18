from typing import Tuple

import numpy as np
import torch


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Itorchut matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    return X @ W_q, X @ W_k, X @ W_v  # (seq_len, d_model)


def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """

    # Your code here
    def softmax(x):
        max_v = torch.max(x, dim=-1, keepdim=True)[0]
        x -= max_v
        return torch.exp(x) / torch.sum(torch.exp(x), dim=-1, keepdim=True)

    d_k = K.shape[-1]
    K = K.T if K.ndim == 2 else torch.transpose(K, dim0=1, dim1=2)

    attn_weights = softmax(
        (Q @ K) / torch.sqrt(torch.tensor(d_k))
    )  # (n_heads, seq_len, seq_len)
    return (
        attn_weights @ V
    )  # (heads, seq_len, seq_len) @ (heads, seq_len, d_k) = (heads, seq_len, d_k)


def multi_head_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int
) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0
    d_k = d_model // n_heads

    Q = torch.transpose(Q.reshape(seq_len, n_heads, d_k), dim0=0, dim1=1)
    K = torch.transpose(K.reshape(seq_len, n_heads, d_k), dim0=0, dim1=1)
    V = torch.transpose(V.reshape(seq_len, n_heads, d_k), dim0=0, dim1=1)

    out = self_attention(Q, K, V)
    return torch.transpose(out, dim0=0, dim1=1).reshape(seq_len, d_model)


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.permutation(np.arange(16)).reshape(4, 4)
    W_q = np.random.randint(0, 4, size=(4, 4))
    W_k = np.random.randint(0, 5, size=(4, 4))
    W_v = np.random.randint(0, 6, size=(4, 4))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    result = multi_head_attention(Q, K, V, n_heads=2)
    print(np.round(result).astype(int).tolist())
