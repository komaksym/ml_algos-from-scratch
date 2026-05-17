from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F

# DO NOT CHANGE SEED (used for deterministic init)
np.random.seed(42)


def _to_tensor(
    x: torch.Tensor | np.ndarray,
    *,
    requires_grad: bool = False,
) -> torch.Tensor:
    tensor = (
        x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    )
    if not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    if requires_grad:
        tensor = tensor.detach().clone()
        tensor.requires_grad_(True)
    return tensor


class Layer:
    def set_input_shape(self, shape: Tuple[int, ...]):
        self.input_shape = shape

    def layer_name(self) -> str:
        return self.__class__.__name__

    def parameters(self) -> int:
        return 0

    def forward_pass(self, X, training: bool = True):
        raise NotImplementedError

    def backward_pass(self, accum_grad):
        raise NotImplementedError

    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_units: int, input_shape: Tuple[int, ...] | None = None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None  # torch.Tensor with requires_grad=True
        self.w0 = None  # torch.Tensor with requires_grad=True
        self._opt = None
        self.outputs = None

    def initialize(self, optimizer) -> None:
        """Initialize weights with a uniform distribution and biases with zeros.
        Hint: use numpy to create deterministic arrays, then convert to torch tensors.
        Also store the provided optimizer for updates.
        """
        self._opt = optimizer
        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = _to_tensor(
            np.random.uniform(
                low=-limit,
                high=limit,
                size=(*self.input_shape, self.n_units),  # (2, 3)
            ),
            requires_grad=True,
        )
        self.w0 = _to_tensor(np.zeros(shape=(1, self.n_units)), requires_grad=True)

    def parameters(self) -> int:
        """Return total number of trainable parameters in W and b."""
        if self.W is None or self.w0 is None:
            return 0
        return self.W.numel() + self.w0.numel()

    def number_of_parameters(self) -> int:
        return self.parameters()

    def forward_pass(self, X, training: bool = True):
        """Use torch.nn.functional.linear for the forward pass (X @ W + b)."""
        self.layer_input = X
        self.outputs = F.linear(_to_tensor(X), self.W.T, self.w0)
        return self.outputs

    def backward_pass(self, accum_grad):
        """Use torch.autograd.grad to get dL/dW, dL/db given upstream gradient.
        Then update params with the provided optimizer and return grad w.r.t. input.
        Hint: grad_input can be computed with a matmul against W.T.
        """
        prev_W = self.W.detach().clone()
        accum_grad = _to_tensor(accum_grad)
        dL_dW, dL_db = torch.autograd.grad(
            self.outputs,
            inputs=(self.W, self.w0),
            grad_outputs=accum_grad,
        )
        if self.trainable:
            self.W = _to_tensor(self._opt.update(self.W, dL_dW), requires_grad=True)
            self.w0 = _to_tensor(self._opt.update(self.w0, dL_db), requires_grad=True)
        return torch.matmul(accum_grad, prev_W.T)

    def output_shape(self) -> Tuple[int, ...]:
        return (self.n_units,)


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
    print(np.round(back_output.detach().numpy(), 5).tolist())
