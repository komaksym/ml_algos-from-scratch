import numpy as np


def apply_rope(
    x: np.ndarray, positions: np.ndarray, base: float = 10000.0
) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.

    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)

    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    d = x.shape[-1]
    assert d % 2 == 0
    i = np.arange(d // 2, dtype=np.int32)  # (d//2)
    thetas = 1 / (base ** ((2 * i) / d))  # (d//2)

    positions = np.arange(x.shape[0])  # (n_rows)
    angles = positions[:, None] * thetas[None, :]  # (n_rows, d//2)
    x_even = x[:, ::2]  # (n_rows, d//2)
    x_odd = x[:, 1::2]  # (n_rows, d//2)
    cos = np.cos(angles)  # (n_rows, d//2)
    sin = np.sin(angles)  # (n_rows, d//2)

    new_even = np.empty((cos.shape))
    new_odd = np.empty((cos.shape))
    new_even[:, i] = x_even[:, i] * cos[:, i] - x_odd[:, i] * sin[:, i]
    new_odd[:, i] = x_even[:, i] * sin[:, i] - x_even[:, i] * cos[:, i]

    new_x = np.empty(x.shape, dtype=x.dtype)
    new_x[:, ::2] = new_even
    new_x[:, 1::2] = new_odd

    return new_x


if __name__ == "__main__":
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    positions = np.array([0])
    result = apply_rope(x, positions)
    print(np.round(result, 4).tolist())
