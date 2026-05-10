import numpy as np


def softmax(scores):
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    return np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)


class MHA:
    def __init__(self, X, W_q, W_k, W_v, n_heads):
        self.Q = X @ W_q  # (seq_len, qkv_dim)
        self.K = X @ W_k  # (seq_len, qkv_dim)
        self.V = X @ W_v  # (seq_len, qkv_dim)
        self.n_heads = n_heads

    def self_attention(self, Q, K, V):
        return softmax((Q @ K.T) / np.sqrt(K.shape[-1])) @ V  # (seq_len, qkv_dim)

    def multi_head_attention(self):
        Q_heads = np.hsplit(
            self.Q, self.n_heads
        )  # [(seq_len, qkv_dim / n_heads)] * n_heads
        K_heads = np.hsplit(
            self.K, self.n_heads
        )  # [(seq_len, qkv_dim / n_heads)] * n_heads
        V_heads = np.hsplit(
            self.V, self.n_heads
        )  # [(seq_len, qkv_dim / n_heads)] * n_heads

        attn_out = np.hstack(
            [self.self_attention(Q, K, V) for Q, K, V in zip(Q_heads, K_heads, V_heads)]
        )  # (seq_len, qkv_dim)
        return attn_out


x = np.random.rand(4, 6)
W_q = np.random.rand(6, 8)
W_k = np.random.rand(6, 8)
W_v = np.random.rand(6, 8)
n_heads = 2

print(MHA(x, W_q, W_k, W_v, n_heads).multi_head_attention())
