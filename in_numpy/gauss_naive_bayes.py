import math

import numpy as np


def gaussian_naive_bayes(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray
) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.

    Args:
            X_train: Training features (shape: N_train x D)
            y_train: Training labels (shape: N_train)
            X_test: Test features (shape: N_test x D)

    Returns:
            Predicted class labels for X_test (shape: N_test)
    """

    # Your code here
    def train(X_train, y_train):
        num_samples, num_features = X_train.shape
        classes, counts = np.unique(
            y_train, return_counts=True, equal_nan=False, sorted=True
        )
        num_classes = classes.shape[0]
        priors = counts / num_samples  # (num_features,)
        eps = 1e-9

        means = np.zeros((num_classes, num_features))
        variances = np.zeros((num_classes, num_features))

        for idx, c in enumerate(range(num_classes)):
            mask = y_train == c

            num_samples_c = np.sum(mask)

            means[idx] = np.sum(X_train[mask], axis=0) / num_samples_c
            variances[idx] = (
                np.sum((X_train[mask] - means[idx]) ** 2, axis=0) / num_samples_c + eps
            )

        return priors, means, variances

    def predict(X_test, priors, means, variances):
        likelihoods = np.array(
            [
                1
                / np.sqrt(2 * math.pi * variances)
                * np.exp(-((sample - means) ** 2) / (2 * variances))
                for sample in X_test
            ]
        )  # (n_test, num_clases, num_features)

        preds = np.log(np.clip(priors, 1e-12, None)) + np.sum(
            np.log(np.clip(likelihoods, 1e-12, None)), axis=2
        )  # (n_test, n_classes, )
        y_hat = np.argmax(preds, axis=1)  # (n_samples,)
        return y_hat.squeeze()

    priors, means, variances = train(X_train, y_train)
    y_hats = predict(X_test, priors, means, variances)
    return y_hats


print(
    gaussian_naive_bayes(
        np.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [6.0, 7.0],
                [7.0, 8.0],
                [8.0, 9.0],
            ]
        ),
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([[2.5, 3.5], [6.5, 7.5]]),
    )
)
