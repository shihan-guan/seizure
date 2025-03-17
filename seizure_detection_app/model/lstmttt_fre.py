import torch.nn as nn

class LSTMModelTTT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pretext_output_size=30720):
        super(LSTMModelTTT, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

        # Classification Head
        self.classification_head = nn.Linear(hidden_size, output_size)
        # Pretext Task Head (Regression for frequency prediction)
        self.pretext_head = nn.Linear(hidden_size, pretext_output_size)

    def forward(self, x, task='classification'):
        # x shape: [batch_size, channels, time_steps]
        # x = x.permute(0, 2, 1)  # Now x shape: [batch_size, time_steps, channels]

        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output from the last time step

        if task == 'classification':
            logits = self.classification_head(out)
        elif task == 'pretext':
            preds = self.pretext_head(out)
            return preds
        else:
            raise ValueError("Task must be 'classification' or 'pretext'")
        return logits

def initialize_lstm_model_ttt(input_size, hidden_size, num_layers, output_size, pretext_output_size):
    return LSTMModelTTT(input_size, hidden_size, num_layers, output_size, pretext_output_size)
