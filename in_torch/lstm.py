import torch


class LSTM:
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases as float64 tensors
        self.Wf = torch.randn(
            hidden_size, input_size + hidden_size, dtype=torch.float64
        )
        self.Wi = torch.randn(
            hidden_size, input_size + hidden_size, dtype=torch.float64
        )
        self.Wc = torch.randn(
            hidden_size, input_size + hidden_size, dtype=torch.float64
        )
        self.Wo = torch.randn(
            hidden_size, input_size + hidden_size, dtype=torch.float64
        )

        self.bf = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bi = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bc = torch.zeros(hidden_size, 1, dtype=torch.float64)
        self.bo = torch.zeros(hidden_size, 1, dtype=torch.float64)

    def forward(
        self,
        x: torch.Tensor,
        initial_hidden_state: torch.Tensor,
        initial_cell_state: torch.Tensor,
    ):
        """
        Processes a sequence of inputs and returns the hidden states,
        final hidden state, and final cell state.

        Args:
            x: Input tensor of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)

        Returns:
            outputs: Tensor of hidden states at each time step
            h: Final hidden state tensor
            c: Final cell state tensor
        """
        hidden_states = torch.empty((x.shape[0], self.hidden_size, 1))
        x = torch.tensor(x)
        h = torch.tensor(initial_hidden_state)  # (1, 1)
        c = torch.tensor(initial_cell_state)  # (1, 1)

        for i in range(x.shape[0]):
            x_t = x[i]  # (1,)
            cur = torch.cat((h, x_t.reshape(x.shape[1], 1)), dim=0)
            f_t = torch.nn.functional.sigmoid(self.Wf @ cur + self.bf)

            i_t = torch.nn.functional.sigmoid(self.Wi @ cur + self.bi)
            cand_t = torch.tanh(self.Wc @ cur + self.bc)

            c = f_t * c + i_t * cand_t

            o_t = torch.nn.functional.sigmoid(self.Wo @ cur + self.bo)
            h = o_t * torch.tanh(c)

            hidden_states[i] = h

        return hidden_states, h, c
