import torch
import torch.nn as nn

class BiGRUModelChannelAttention(nn.Module):
    def __init__(
        self,
        seq_length,
        output_size,
        hidden_size,
        num_layers=1,
        dropout=0.5,
        num_channels=12,
        num_heads=1,
          # Adjust based on your data
    ):
        super(BiGRUModelChannelAttention, self).__init__()
        self.num_layers = num_layers
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1

        self.hidden_size = 64  # Hidden size for GRU

        # Convolutional layers processing all channels together
        self.conv1 = nn.Conv1d(
            in_channels=num_channels, out_channels=64, kernel_size=3, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=512, kernel_size=3, padding=1
        )
        self.relu3 = nn.ReLU()
        # No pooling after conv3

        self.conv4 = nn.Conv1d(
            in_channels=512, out_channels=1024, kernel_size=3, padding=1
        )
        self.relu4 = nn.ReLU()
        # No pooling after conv4

        # Adaptive pooling to get a fixed output length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Compute the input size for fc_cnn
        with torch.no_grad():
            # Use a dummy input to compute the size after convolutions
            dummy_input = torch.zeros(1, num_channels, seq_length)
            x = dummy_input

            x = self.conv1(x)
            x = self.relu1(x)
            x = self.pool1(x)

            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool2(x)

            x = self.conv3(x)
            x = self.relu3(x)

            x = self.conv4(x)
            x = self.relu4(x)

            x = self.adaptive_pool(x)
            x = x.view(1, -1)
            fc_cnn_input_size = x.size(1)

        # Now define the fully connected layer with the computed input size
        self.fc_cnn = nn.Linear(fc_cnn_input_size, 128)

        # BiGRU with hidden size 64
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self.bidirectional
        )

        # Update embed_dim for attention layer
        self.embed_dim = self.hidden_size * self.num_directions

        # Define self-attention over time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Define a fully connected layer for the final output
        self.fc = nn.Linear(self.embed_dim, output_size)

    def forward(self, x):
        # x shape: (batch_size, num_channels, sequence_length)
        batch_size = x.size(0)  # Dynamically get the batch size

        # Pass through CNN layers
        x = self.conv1(x)  # Shape: (batch_size, 64, seq_len)
        x = self.relu1(x)
        x = self.pool1(x)  # Shape: (batch_size, 64, seq_len / 2)

        x = self.conv2(x)  # Shape: (batch_size, 128, seq_len / 2)
        x = self.relu2(x)
        x = self.pool2(x)  # Shape: (batch_size, 128, seq_len / 4)

        x = self.conv3(x)  # Shape: (batch_size, 512, seq_len / 4)
        x = self.relu3(x)
        # No pooling after conv3

        x = self.conv4(x)  # Shape: (batch_size, 1024, seq_len / 4)
        x = self.relu4(x)
        # No pooling after conv4

        # Adaptive pooling to get fixed output length (output size = 1)
        x = self.adaptive_pool(x)  # Shape: (batch_size, 1024, 1)

        # Flatten the features
        x = x.view(batch_size, -1)  # Shape: (batch_size, fc_cnn_input_size)

        # Fully connected layer to reduce to 128-dimensional representation
        x = self.fc_cnn(x)  # Shape: (batch_size, 128)

        # Prepare input for GRU
        x = x.unsqueeze(1)  # Shape: (batch_size, sequence_length=1, 128)

        # Initialize hidden state
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ).to(x.device)

        # Pass through GRU
        out, _ = self.gru(x, h0)  # Shape: (batch_size, 1, hidden_size * num_directions)

        # Since sequence_length=1, we can squeeze the sequence dimension
        out = out.squeeze(1)  # Shape: (batch_size, hidden_size * num_directions)

        # Fully connected layer to produce the final output
        out = self.fc(out)  # Shape: (batch_size, output_size)

        return out
