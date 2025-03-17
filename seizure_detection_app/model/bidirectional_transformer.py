# model/bidirectional_transformer.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create constant 'pe' matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If d_model is odd, pad div_term to match
            div_term = div_term[:-1]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class BidirectionalTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads=8, num_layers=3, dropout=0.5):
        super(BidirectionalTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        x = self.embedding(x)  # Shape: [batch_size, seq_len, hidden_size]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, hidden_size]

        # Transformer encoding and decoding
        memory = self.transformer_encoder(x)  # memory shape: [seq_len, batch_size, hidden_size]
        output = self.transformer_decoder(x, memory)

        output = output.permute(1, 0, 2)  # Back to [batch_size, seq_len, hidden_size]
        output = output.mean(dim=1)  # Global average pooling over the sequence
        output = self.fc(output)
        return output
