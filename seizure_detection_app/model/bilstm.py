import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 because of bidirectional

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # x = x.permute(0, 2, 1)  # Now x has shape (batch_size, features, channels)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size*2)
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size*2)
        out = self.fc(lstm_out)
        return out

def initialize_bilstm_model(input_size, hidden_size, num_layers, output_size):
    return BiLSTM(input_size, hidden_size, num_layers, output_size)
