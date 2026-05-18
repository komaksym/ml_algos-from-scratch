import numpy as np


def softmax(x):
    max_score = np.max(x, axis=1, keepdims=True)
    x -= max_score
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)


def masked_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    scores += mask
    attn_weights = softmax(scores)
    return attn_weights @ V


if __name__ == "__main__":
    np.random.seed(42)
    X = np.arange(48).reshape(6, 8)
    X = np.random.permutation(X.flatten()).reshape(6, 8)
    mask = np.triu(np.ones((6, 6)) * (-np.inf), k=1)
    W_q = np.random.randint(0, 4, size=(8, 8))
    W_k = np.random.randint(0, 5, size=(8, 8))
    W_v = np.random.randint(0, 6, size=(8, 8))
    Q, K, V = compute_qkv(X, W_q, W_k, W_v)
    print(masked_attention(Q, K, V, mask))
