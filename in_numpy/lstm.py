import numpy as np


def sigmoid(x):
    a = 1 / (1 + np.exp(-x))
    return a


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """
        hidden_states = np.zeros((x.shape[0], initial_hidden_state.shape[-1]))
        x = np.array(x)
        h = np.array(initial_hidden_state)  # (1, 1)
        c = np.array(initial_cell_state)  # (1, 1)

        for i in range(x.shape[0]):
            x_t = x[i]  # (1,)
            cur = np.concatenate([h, x_t.reshape(x.shape[1], 1)], axis=0)
            f_t = sigmoid(self.Wf @ cur + self.bf)

            i_t = sigmoid(self.Wi @ cur + self.bi)
            cand_t = np.tanh(self.Wc @ cur + self.bc)

            c = f_t * c + i_t * cand_t

            o_t = sigmoid(self.Wo @ cur + self.bo)
            h = o_t * np.tanh(c)

            hidden_states[i] = h

        return hidden_states, h, c


if __name__ == "__main__":
    input_sequence = np.array([[1.0], [2.0], [3.0]])
    initial_hidden_state = np.zeros((1, 1))
    initial_cell_state = np.zeros((1, 1))

    lstm = LSTM(input_size=1, hidden_size=1)
    # Set weights and biases for reproducibility
    lstm.Wf = np.array([[0.5, 0.5]])
    lstm.Wi = np.array([[0.5, 0.5]])
    lstm.Wc = np.array([[0.3, 0.3]])
    lstm.Wo = np.array([[0.5, 0.5]])
    lstm.bf = np.array([[0.1]])
    lstm.bi = np.array([[0.1]])
    lstm.bc = np.array([[0.1]])
    lstm.bo = np.array([[0.1]])

    outputs, final_h, final_c = lstm.forward(
        input_sequence, initial_hidden_state, initial_cell_state
    )

    print(final_h)
