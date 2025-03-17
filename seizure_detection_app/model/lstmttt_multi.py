import torch.nn as nn

class LSTMModelMultiTTT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModelMultiTTT, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

        # Classification Head
        self.classification_head = nn.Linear(hidden_size, output_size)
        # Pretext Task Heads
        self.time_shift_head = nn.Linear(hidden_size, 2)  # Binary classification: shuffled or not
        self.amplitude_scaling_head = nn.Linear(hidden_size, 2)  # Binary classification: scaled or not
        self.temporal_inversion_head = nn.Linear(hidden_size, 2)  # Binary classification: inverted or not

    def forward(self, x, task='classification'):
        # x = x.permute(0, 2, 1)  # [batch_size, time_steps, channels]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step

        if task == 'classification':
            logits = self.classification_head(out)
            return logits
        elif task == 'time_shift':
            logits = self.time_shift_head(out)
            return logits
        elif task == 'amplitude_scaling':
            logits = self.amplitude_scaling_head(out)
            return logits
        elif task == 'temporal_inversion':
            logits = self.temporal_inversion_head(out)
            return logits
        else:
            raise ValueError("Invalid task specified.")

