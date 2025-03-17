import torch
import torch.nn as nn

class BiGRUModelChannelAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.5,
        num_channels=12,
        num_heads=1
    ):
        super(BiGRUModelChannelAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        # Define the bidirectional GRU layer for per-channel processing
        self.gru = nn.GRU(
            input_size=input_size,  # Each time step has 1 feature per channel
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )

        # Update embed_dim for attention layer
        self.embed_dim = hidden_size * self.num_directions

        # Define self-attention over channel representations
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True  # Ensures input/output tensors are of shape (batch_size, seq_length, embed_dim)
        )

        # Define a fully connected layer for the final output
        self.fc = nn.Linear(self.embed_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)


    def forward(self, x):
        # x shape: (batch_size, num_channels, sequence_length)
        batch_size, num_channels, sequence_length = x.size()

        # Reshape to process each channel independently
        # New shape: (batch_size * num_channels, sequence_length, input_size)
        x = x.view(batch_size * num_channels, sequence_length, 1)

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
        # For bidirectional GRU, concatenate the last output from forward and backward passes
        out_forward = out[:, -1, :self.hidden_size]    # Forward direction last time step
        out_backward = out[:, 0, self.hidden_size:]    # Backward direction last time step
        out = torch.cat((out_forward, out_backward), dim=1)  # Shape: (batch_size * num_channels, hidden_size * num_directions)

        # Reshape back to (batch_size, num_channels, hidden_size * num_directions)
        out = out.view(batch_size, num_channels, self.embed_dim)

        # Self-attention over channels
        # The channels are treated as the sequence dimension for self-attention
        out_attn, _ = self.attention(out, out, out)
        # out_attn shape: (batch_size, num_channels, embed_dim)

        # Aggregate the outputs (e.g., mean over channels)
        out = out_attn.mean(dim=1)  # Shape: (batch_size, embed_dim)

        # Fully connected layer
        out = self.fc(out)  # Shape: (batch_size, output_size)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
