import numpy as np


def softmax(x):
    max_v = np.max(x, axis=-1, keepdims=True)
    x = x - max_v
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def moe_topk_routing(
    router_logits: np.ndarray, expert_outputs: np.ndarray, k: int
) -> np.ndarray:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.

    For each token:
    1. Select the top-k experts based on router_logits
    2. Compute softmax weights over only the selected experts
    3. Return weighted combination of the selected expert outputs

    Args:
        router_logits: Shape (batch_size, num_experts)
                      Raw scores from the router for each expert
        expert_outputs: Shape (batch_size, num_experts, hidden_dim)
                       Output from each expert for each input
        k: Number of experts to select per token

    Returns:
        Shape (batch_size, hidden_dim) - weighted combination of expert outputs
    """
    top_k_logits_idxs = np.argsort(router_logits, axis=-1)[..., ::-1][:, :k].squeeze()
    batch_idxs = np.arange(router_logits.shape[0])[:, None]
    probs = softmax(router_logits[batch_idxs, top_k_logits_idxs])
    weighted_sum = probs * expert_outputs[:, top_k_logits_idxs, :]
    return weighted_sum.squeeze().sum(axis=1)


if __name__ == "__main__":
    import numpy as np

    router_logits = np.array([[2.0, 1.0, 0.5, 0.1]])
    expert_outputs = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]])
    result = moe_topk_routing(router_logits, expert_outputs, k=2)
    print(np.round(result, 4).tolist())
