import numpy as np

"""
- Declare a list of losses
- Declare weights and bias
- Iterate over num of iterations
	- Compute y_hat
	- Compute Loss
	- Append loss to the list of losses
	- Compute grads of Loss w.r.t. params
	- Update grads
- Return ([weights, bias], losses)
"""


def train_logreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    losses = []
    W = np.zeros(X.shape[-1])
    b = 0

    for _ in range(iterations):
        # Compute logits
        z = np.dot(X, W) + b  # shape (X.shape[0])

        # Compute y_hat
        y_hat = 1 / (1 + np.exp(-z))  # shape (X.shape[0])

        # Compute Loss
        loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))  # shape (1)

        # Append loss
        losses.append(round(loss, 4))

        # Compute grads
        dW = X.T @ (y_hat - y)  # shape (W.shape)
        db = np.sum(y_hat - y)  # shape (X.shape[0])

        # Update grads
        W -= learning_rate * dW
        b -= learning_rate * db

    coefficients = [round(b, 4)] + W.flatten().round(4).tolist()

    return (coefficients, losses)


# Input
print(
    train_logreg(
        np.array([[1.0, 0.5], [-0.5, -1.5], [2.0, 1.5], [-2.0, -1.0]]),
        np.array([1, 0, 1, 0]),
        0.01,
        20,
    )
)
