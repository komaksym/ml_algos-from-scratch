import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gru_cell(
    x: np.ndarray,
    h_prev: np.ndarray,
    W_z: np.ndarray,
    U_z: np.ndarray,
    b_z: np.ndarray,
    W_r: np.ndarray,
    U_r: np.ndarray,
    b_r: np.ndarray,
    W_h: np.ndarray,
    U_h: np.ndarray,
    b_h: np.ndarray,
) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.

    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input
        U_z, U_r, U_h: Weight matrices for hidden state
        b_z, b_r, b_h: Bias vectors

    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    x = np.expand_dims(x, 0) if x.ndim == 1 else x
    h = h_prev

    for x_t in x:
        z_t = sigmoid(W_z @ x_t + U_z @ h + b_z)
        r_t = sigmoid(W_r @ x_t + U_r @ h + b_r)
        cand = np.tanh(
            W_h @ x_t + U_h @ (r_t * h) + b_h
        )  # not sure about Uh * or @ here
        h = (1 - z_t) * h + z_t * cand
    return h


if __name__ == "__main__":
    x = np.array([1.0, 0.5])
    h_prev = np.zeros(3)
    W_z = np.ones((3, 2)) * 0.1
    U_z = np.ones((3, 3)) * 0.1
    b_z = np.zeros(3)
    W_r = np.ones((3, 2)) * 0.1
    U_r = np.ones((3, 3)) * 0.1
    b_r = np.zeros(3)
    W_h = np.ones((3, 2)) * 0.2
    U_h = np.ones((3, 3)) * 0.2
    b_h = np.zeros(3)
    result = gru_cell(x, h_prev, W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h)
    print(np.round(result, 4))
