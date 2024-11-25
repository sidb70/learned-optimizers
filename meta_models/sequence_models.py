import torch
import torch.nn as nn
import torch.nn.functional as F


class UnrolledRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(UnrolledRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len

        # RNN weights
        self.W_xh = nn.Linear(input_size, hidden_size)  # Input to hidden weights
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)  # Hidden to hidden weights
        self.W_out = nn.Linear(hidden_size, output_size)  # Hidden to output weights

    def forward(self, x, h0=None):
        """
        Forward pass through the unrolled RNN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            h0: Initial hidden state, of shape (batch_size, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, output_size).
        """
        # print('h0.size', h0.size())
        # print("x.size", x.size())
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size).to(x.device)

        h_t = h0
        outputs = []

        # if length of x is less than seq_len, pad with zeros
        if x.size(1) < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.seq_len - x.size(1)))


        for t in range(self.seq_len):
            x_t = x[:, t, :]  # Input at time step t

            # Compute the hidden state
            h_t = torch.tanh(self.W_xh(x_t) + self.W_hh(h_t))

            # Compute the output
            y_t = self.W_out(h_t)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # Stack outputs along the sequence dimension
        # return only the last output
        last = outputs[:, -1, :]
        return last