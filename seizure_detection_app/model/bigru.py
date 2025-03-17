import torch
import torch.nn as nn

class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = True

        # Define the bidirectional GRU layer
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )

        # Define a fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_layers * 2,  # Multiply by 2 for bidirectional
            x.size(0),
            self.hidden_size
        ).to(x.device)

        # GRU layer
        out, _ = self.gru(x, h0)

        # Get the last time step's output from both directions
        # Forward direction: out[:, -1, :hidden_size]
        # Backward direction: out[:, 0, hidden_size:]
        out_forward = out[:, -1, :self.hidden_size]
        out_backward = out[:, 0, self.hidden_size:]
        out = torch.cat((out_forward, out_backward), dim=1)

        # Fully connected layer
        out = self.fc(out)

        return out
