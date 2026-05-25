import torch
import torch.nn.functional as F


def moe_topk_routing(
    router_logits: torch.Tensor, expert_outputs: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.
    """
    # Stable descending sort -> deterministic tie-breaking (lower index wins).
    # torch.topk gives no ordering guarantee on ties, which breaks all-equal cases.
    sorted_indices = torch.argsort(router_logits, dim=1, descending=True, stable=True)
    top_k_indices = sorted_indices[:, :k]  # (B, k)

    # Gather the logits for the selected experts, then softmax over just those.
    top_k_logits = torch.gather(router_logits, dim=1, index=top_k_indices)  # (B, k)
    routing_weights = F.softmax(top_k_logits, dim=1)  # (B, k)

    # Gather the corresponding expert outputs.
    hidden_dim = expert_outputs.shape[2]
    gather_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)  # (B, k, H)
    top_k_outputs = torch.gather(expert_outputs, dim=1, index=gather_indices)

    # Weighted combination.
    return (routing_weights.unsqueeze(-1) * top_k_outputs).sum(dim=1)  # (B, H)
