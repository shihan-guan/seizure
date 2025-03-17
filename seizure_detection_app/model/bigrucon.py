import torch
import torch.nn as nn

class BiGRUModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.0,
        num_channels=12,
    ):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        # Define the BiGRU layer for per-channel processing
        self.gru = nn.GRU(
            input_size=input_size,  # Each time step has one feature per channel
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )

        # Fully connected layer for final classification
        # If concatenating outputs, input size is hidden_size * num_directions * num_channels
        # If averaging or summing outputs, input size is hidden_size * num_directions
        self.fc = nn.Linear(hidden_size * self.num_directions * num_channels, output_size)

    def forward(self, x):
        # x shape: (batch_size, num_channels, sequence_length)
        batch_size, num_channels, sequence_length = x.size()

        # Reshape to process each channel independently
        # New shape: (batch_size * num_channels, sequence_length, input_size)
        x = x.view(batch_size * num_channels, sequence_length, -1)

        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size * num_channels,
            self.hidden_size
        ).to(x.device)

        # GRU layer: process each channel's time series
        out, _ = self.gru(x, h0)
        # out shape: (batch_size * num_channels, sequence_length, hidden_size * num_directions)

        # Get the last time step's output from both directions
        # For bidirectional GRU, concatenate the outputs from forward and backward passes
        out_forward = out[:, -1, :self.hidden_size]
        out_backward = out[:, 0, self.hidden_size:]
        out = torch.cat((out_forward, out_backward), dim=1)
        # out shape: (batch_size * num_channels, hidden_size * num_directions)

        # Reshape back to (batch_size, num_channels, hidden_size * num_directions)
        out = out.view(batch_size, num_channels, -1)
        # Now, out shape: (batch_size, num_channels, hidden_size * num_directions)

        # Combine outputs from all channels
        # Option 1: Concatenate along the channel dimension
        out = out.view(batch_size, -1)  # Flatten the channels
        # out shape: (batch_size, num_channels * hidden_size * num_directions)

        # Option 2: Average over channels
        # out = out.mean(dim=1)
        # out shape: (batch_size, hidden_size * num_directions)

        # Option 3: Sum over channels
        # out = out.sum(dim=1)
        # out shape: (batch_size, hidden_size * num_directions)

        # Fully connected layer
        out = self.fc(out)
        # out shape: (batch_size, output_size)

        return out
