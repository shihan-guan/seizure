# model/crnn.py
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, input_channels, num_classes, hidden_size, num_layers=1, dropout=0.5):
        super(CRNN, self).__init__()
        # 1D Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Recurrent Layer (GRU)
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Transpose to [batch_size, sequence_length, channels] for GRU
        x = x.permute(0, 2, 1)

        # GRU layer
        h0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        x, _ = self.gru(x, h0)

        # Fully connected layer
        x = x[:, -1, :]  # Get the last output of the GRU
        x = self.dropout(x)
        x = self.fc(x)
        return x
