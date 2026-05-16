import torch


def gru_cell(
    x: torch.Tensor,
    h_prev: torch.Tensor,
    W_z: torch.Tensor,
    U_z: torch.Tensor,
    b_z: torch.Tensor,
    W_r: torch.Tensor,
    U_r: torch.Tensor,
    b_r: torch.Tensor,
    W_h: torch.Tensor,
    U_h: torch.Tensor,
    b_h: torch.Tensor,
) -> torch.Tensor:
    """
    Implements a single GRU cell forward pass.

    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input of shape (hidden_size, input_size)
        U_z, U_r, U_h: Weight matrices for hidden state of shape (hidden_size, hidden_size)
        b_z, b_r, b_h: Bias vectors of shape (hidden_size,)

    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    x = torch.unsqueeze(x, dim=0) if x.ndim == 1 else x
    h = h_prev

    for x_t in x:
        z_t = torch.sigmoid(W_z @ x_t + U_z @ h + b_z)
        r_t = torch.sigmoid(W_r @ x_t + U_r @ h + b_r)
        cand = torch.tanh(W_h @ x_t + U_h @ (r_t * h) + b_h)
        h = (1 - z_t) * h + z_t * cand
    return h
