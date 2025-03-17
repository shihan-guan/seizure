import torch
import torch.nn as nn

class MultiChannelCNNLSTM(nn.Module):
    def __init__(
        self,
        num_channels=12,
        window_size=256,
        stride=50,
        cnn_encode_size=64,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        output_size=2,
        mlp_hidden_sizes=[32],
        cnn_kernel_size=3,
        cnn_stride=1,
        cnn_padding=1
    ):
        """
        Initializes the MultiChannelCNNLSTM model with CNN-based encoding and channel-wise LSTMs.

        Parameters:
        - num_channels (int): Number of EEG channels.
        - window_size (int): Number of time steps per window for encoding.
        - stride (int): Step size between windows.
        - cnn_encode_size (int): Size of the encoded feature per window via CNN.
        - lstm_hidden_size (int): Number of features in the hidden state of each LSTM.
        - lstm_num_layers (int): Number of recurrent layers in each LSTM.
        - output_size (int): Number of output classes.
        - mlp_hidden_sizes (list of int): Sizes of hidden layers in the MLP.
        - cnn_kernel_size (int): Kernel size for CNN layers.
        - cnn_stride (int): Stride for CNN layers.
        - cnn_padding (int): Padding for CNN layers.
        """
        super(MultiChannelCNNLSTM, self).__init__()
        self.num_channels = num_channels
        self.window_size = window_size
        self.stride = stride
        self.cnn_encode_size = cnn_encode_size

        # Shared CNN encoder for all channels
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=window_size, out_channels=cnn_encode_size,
                      kernel_size=cnn_kernel_size, stride=cnn_stride, padding=cnn_padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Optional: Adjust based on desired encoding
        )

        # Initialize a separate LSTM for each channel
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=cnn_encode_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                bidirectional=False,
                batch_first=True
            ) for _ in range(num_channels)
        ])

        # Calculate the input size for the MLP
        mlp_input_size = lstm_hidden_size * num_channels

        # Define the MLP
        layers = []
        prev_size = mlp_input_size
        for hidden in mlp_hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))  # Dropout for regularization
            prev_size = hidden
        layers.append(nn.Linear(prev_size, output_size))  # Final output layer

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_len)

        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, output_size)
        """
        batch_size, num_channels, seq_len = x.size()

        # Calculate the number of windows
        num_windows = 1 + (seq_len - self.window_size) // self.stride
        if num_windows <= 0:
            raise ValueError(f"Sequence length ({seq_len}) is shorter than the window size ({self.window_size}).")

        # Extract sliding windows
        # Resulting shape: (batch_size, num_channels, num_windows, window_size)
        windows = x.unfold(dimension=2, size=self.window_size, step=self.stride)
        # windows: (batch_size, num_channels, num_windows, window_size)

        # Reshape to process all windows across all channels
        # New shape: (batch_size * num_channels, num_windows, window_size)
        windows = windows.contiguous().view(batch_size * num_channels, num_windows, self.window_size)

        # Permute to (batch_size * num_channels, window_size, num_windows)
        # Required for CNN input: (N, C, L)
        windows = windows.permute(0, 2, 1)  # (batch_size * num_channels, window_size, num_windows)

        # Pass through CNN
        # CNN expects input shape: (batch_size * num_channels, window_size, num_windows)
        # After CNN: (batch_size * num_channels, cnn_encode_size, L_out)
        cnn_out = self.cnn(windows)

        # Permute to (batch_size * num_channels, L_out, cnn_encode_size) for LSTM
        cnn_out = cnn_out.permute(0, 2, 1).contiguous()

        # Reshape to (batch_size, num_channels, L_out, cnn_encode_size)
        L_out = cnn_out.size(1)
        cnn_out = cnn_out.view(batch_size, num_channels, L_out, self.cnn_encode_size)

        # Prepare to pass through channel-wise LSTMs
        # Initialize a list to hold the last hidden states from each LSTM
        lstm_last_hidden = []

        for channel in range(num_channels):
            # Extract data for the current channel: (batch_size, L_out, cnn_encode_size)
            channel_data = cnn_out[:, channel, :, :]  # (batch_size, L_out, cnn_encode_size)

            # Pass through the corresponding LSTM
            lstm_out, _ = self.lstm_layers[channel](channel_data)  # lstm_out: (batch_size, L_out, lstm_hidden_size)

            # Extract the last hidden state
            last_hidden = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)

            lstm_last_hidden.append(last_hidden)

        # Concatenate all last hidden states: (batch_size, num_channels * lstm_hidden_size)
        lstm_last_hidden = torch.cat(lstm_last_hidden, dim=1)

        # Pass through MLP
        out = self.mlp(lstm_last_hidden)  # (batch_size, output_size)

        return out
