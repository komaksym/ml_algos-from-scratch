import numpy as np

# DO NOT CHANGE SEED
np.random.seed(42)


# DO NOT CHANGE LAYER CLASS
class Layer(object):
    def set_input_shape(self, shape):

        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
        self.optimizer = None

    def initialize(self, optimizer):
        self.optimizer = optimizer
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(
            low=-limit,
            high=limit,
            size=(*self.input_shape, self.n_units),  # (2, 3)
        )
        self.w0 = np.zeros(shape=(1, self.n_units))

    def forward_pass(self, X):
        assert X.shape[-1] == self.input_shape[0]
        self.layer_input = X
        return X @ self.W + self.w0  # (1, 2) @ (2, 3) = (1, 3)

    def backward_pass(self, accum_grad):
        if self.trainable:
            W_prev = self.W.copy()  # (2, 3)
            self.W = self.optimizer.update(
                self.W, self.layer_input.T @ accum_grad
            )  # (2, 1) @ (1, 3)  = (2, 3)
            self.w0 = self.optimizer.update(self.w0, accum_grad)  # (1, 3)
        return accum_grad @ W_prev.T  # (1, 3) @ (2, 3) = (1, 2)

    def parameters(self):
        return self.W.size + self.w0.size


if __name__ == "__main__":
    dense_layer = Dense(n_units=3, input_shape=(2,))

    class MockOptimizer:
        def update(self, weights, grad):
            return weights - 0.01 * grad

    optimizer = MockOptimizer()
    dense_layer.initialize(optimizer)

    X = np.array([[1, 2]])
    output = dense_layer.forward_pass(X)

    accum_grad = np.array([[0.1, 0.2, 0.3]])
    back_output = dense_layer.backward_pass(accum_grad)
    print(np.round(back_output, 5).tolist())
