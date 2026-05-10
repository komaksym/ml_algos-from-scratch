import numpy as np


def softmax(scores):
    max_scores = np.max(scores, axis=-1, keepdims=True)
    scores = scores - max_scores
    return np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)


def train_softmaxreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    m = X.shape[0]
    num_classes = np.max(y) + 1
    losses = []
    W = np.zeros((X.shape[-1], num_classes))
    b = np.zeros(num_classes)

    # Set up One hot encoded y
    y_OHE = np.zeros((m, y.max() + 1))
    y_OHE[np.arange(m), y] = 1

    for _ in range(iterations):
        # Compute logits
        z = np.dot(X, W) + b  # shape (X.shape[0], y.shape[0])

        # Compute p_hat
        p_hat = softmax(z)  # shape (X.shape[0])

        # Compute Loss
        loss = -np.sum(y_OHE * np.log(p_hat))

        # Append loss
        losses.append(round(loss, 4))

        # Compute grads
        dW = X.T @ (p_hat - y_OHE)  # shape (W.shape)
        db = np.sum(p_hat - y_OHE, axis=0)

        # Update grads
        W -= learning_rate * dW
        b -= learning_rate * db

    coeff = np.hstack(
        (np.expand_dims(b, axis=-1).round(4).tolist(), W.T.round(4).tolist())
    ).tolist()

    return (coeff, losses)


# Input
print(
    train_softmaxreg(
        np.array([[0.5, -1.2], [-0.3, 1.1], [0.8, -0.6]]), np.array([0, 1, 2]), 0.01, 10
    )
)
# print(train_softmaxreg(np.array([[2.5257, 2.3333, 1.7730, 0.4106, -1.6648], [1.5101, 1.3023, 1.3198, 1.3608, 0.4638], [-2.0969, -1.3596, -1.0403, -2.2548, -0.3235], [-0.9666, -0.6068, -0.7201, -1.7325, -1.1281], [-0.3809, -0.2485, 0.1878, 0.5235, 1.3072], [0.5482, 0.3315, 0.1067, 0.3069, -0.3755], [-3.0339, -2.0196, -0.6546, -0.9033, 2.8918], [0.2860, -0.1265, -0.5220, 0.2830, -0.5865], [-0.2626, 0.7601, 1.8409, -0.2324, 1.8071], [0.3028, -0.4023, -1.2955, -0.1422, -1.7812]]), np.array([2, 3, 0, 0, 1, 3, 0, 1, 2, 1]), 0.03, 10))
