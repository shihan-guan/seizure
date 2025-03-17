import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMAttention(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers, output_size, num_channels=12, dropout=0.5):
        super(BiLSTMAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer applied to the sequence
        self.embedding = nn.Linear(seq_len, 512)  # Embedding sequence into 20 units (20 seconds)

        # BiLSTM layer
        self.bilstm = nn.LSTM(num_channels, hidden_size, num_layers,
                              bidirectional=True, batch_first=True)
        # Attention layer
        self.attention = nn.Linear(hidden_size * 2, 1)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: (batch_size, num_channels, seq_len)
        # Embed the sequence dimension
        # x = self.embedding(x)  # x: (batch_size, num_channels, 20)
        # x = F.relu(x)
        # x = self.dropout(x)

        # Permute to (batch_size, seq_len, num_channels)
        x = x.permute(0, 2, 1)

        # BiLSTM
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = self.dropout(bilstm_out)

        # Attention mechanism
        attention_scores = torch.tanh(self.attention(bilstm_out))
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * bilstm_out, dim=1)
        out = self.fc(context_vector)
        return out

