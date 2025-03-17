import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalBiLSTMAttention(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, output_size, num_channels=12, dropout=0.5, chunk_size=256):
        super(HierarchicalBiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.chunk_size = chunk_size
        self.num_channels = num_channels

        # Calculate the number of chunks
        self.num_chunks = seq_len // chunk_size

        # Lower-level BiLSTM (processes chunks)
        self.lower_bilstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        # Upper-level BiLSTM (processes chunk representations)
        self.upper_bilstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: (batch_size, num_channels, seq_len)
        batch_size, num_channels, seq_len = x.size()

        # Ensure the sequence length is divisible by chunk_size
        assert seq_len % self.chunk_size == 0, "Sequence length must be divisible by chunk size"

        # Reshape x to (batch_size * num_chunks, num_channels, chunk_size)
        x = x.view(batch_size, num_channels, self.num_chunks, self.chunk_size)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_chunks, num_channels, chunk_size)
        x = x.reshape(batch_size * self.num_chunks, num_channels, self.chunk_size)

        # Permute to (batch_size * num_chunks, chunk_size, num_channels)
        x = x.permute(0, 2, 1)

        # Lower-level BiLSTM
        lower_lstm_out, _ = self.lower_bilstm(x)  # (batch_size * num_chunks, chunk_size, hidden_size * 2)
        lower_lstm_out = self.dropout(lower_lstm_out)

        # Take the last hidden state from each chunk
        chunk_representation = lower_lstm_out[:, -1, :]  # (batch_size * num_chunks, hidden_size * 2)

        # Reshape to (batch_size, num_chunks, hidden_size * 2)
        chunk_representation = chunk_representation.view(batch_size, self.num_chunks, -1)

        # Upper-level BiLSTM
        upper_lstm_out, _ = self.upper_bilstm(chunk_representation)  # (batch_size, num_chunks, hidden_size * 2)
        upper_lstm_out = self.dropout(upper_lstm_out)

        # Attention mechanism
        attention_scores = torch.tanh(self.attention(upper_lstm_out))  # (batch_size, num_chunks, 1)
        attention_weights = F.softmax(attention_scores, dim=1)         # (batch_size, num_chunks, 1)
        context_vector = torch.sum(attention_weights * upper_lstm_out, dim=1)  # (batch_size, hidden_size * 2)

        # Output layer
        out = self.fc(context_vector)
        return out
