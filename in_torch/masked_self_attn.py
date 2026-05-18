import torch


def compute_qkv(
    X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor
):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)


def masked_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked self-attention.
    """
    scores = Q @ K.T / K.shape[-1] ** 0.5
    scores += mask
    attn_weights = torch.softmax(scores, dim=-1)
    return attn_weights @ V
