import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """
        Forward pass through the RNN for a given sequence of inputs.
        """
        self.n_time_steps = len(x)
        h_prev = np.zeros(
            (self.hidden_size, 1)
        )  # (hidden_size, 1) ### NOT SURE IF LAST DIM IS 1 HERE

        self.h_T = np.zeros((self.n_time_steps, self.hidden_size, 1))
        self.yhat_T = np.zeros((self.n_time_steps, self.output_size, 1))

        for t in range(self.n_time_steps):
            # Forward prop
            h_t = np.tanh(
                self.W_xh @ x[t][:, np.newaxis] + self.W_hh @ h_prev + self.b_h
            )  # (hidden_size, input_size)
            y_t = self.W_hy @ h_t + self.b_y  # (output_size, input_size)

            # Storing for backprop
            self.h_T[t] = h_t
            self.yhat_T[t] = y_t

            # Updating the h_prev
            h_prev = h_t

        return self.yhat_T

    def backward(self, x, y, learning_rate):
        """
        Backpropagation through time to adjust weights based on error gradient.
        """
        # Grads
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)

        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(self.n_time_steps)):
            # Compute the gradient of the loss with respect to the outputs
            dy_t = self.yhat_T[t] - y[t].reshape(-1, 1)  # (output_size, input_size)

            # Compute the gradients for the output layer weights and biases
            dW_hy += dy_t @ self.h_T[t].T  # (output_size, hidden_size)
            db_y += dy_t  # (output_size, input_size)

            # Backpropagate the gradients through the hidden layers
            dh_t = self.W_hy.T @ dy_t + dh_next  # (hidden_size, input_size)
            dh_raw = dh_t * (1 - self.h_T[t] ** 2)  # (hidden_size, input_size)

            # Compute the gradients for the hidden layer weights and biases
            dW_xh += dh_raw @ x[t][np.newaxis, :]  # (hidden_size, 1) or scalar ???
            dW_hh += (
                dh_raw @ self.h_T[t - 1].T
                if t > 0
                else dh_raw @ np.zeros((self.hidden_size, 1)).T
            )  # (hidden_size, hidden_size)
            db_h += dh_raw

            # Update prev dh_t (dh_t+1)
            dh_next = self.W_hh.T @ dh_raw  # (hidden_size, input_size)

        # Updating Weights
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y


np.random.seed(42)
input_sequence = np.array([[1.0, 2.0], [7.0, 2.0], [1.0, 3.0], [12.0, 4.0]])
expected_output = np.array([[2.0, 1.0], [3.0, 7.0], [4.0, 8.0], [5.0, 10.0]])
rnn = SimpleRNN(input_size=2, hidden_size=10, output_size=2)
for epoch in range(50):
    output = rnn.forward(input_sequence)
    rnn.backward(input_sequence, expected_output, learning_rate=0.01)
print(output)
