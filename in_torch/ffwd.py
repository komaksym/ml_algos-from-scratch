import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNBlock(nn.Module):
    def __init__(self, d_model, d_hidden, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p

        self.W1 = nn.Linear(d_model, d_hidden)
        self.W2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        ffwd = F.dropout(self.W2(F.relu(self.W1(x))), self.dropout_p) + x
        return ffwd


if __name__ == "__main__":
    x = torch.tensor([1.0, -1.0])

    b = FFNBlock(4, 4)
    print(b.forward(x))
