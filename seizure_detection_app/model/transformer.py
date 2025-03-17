import torch
import torch.nn as nn
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, num_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Linear projection layer
        self.proj = nn.Linear(num_channels * patch_size, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, n_channels, seq_length)
        batch_size, channels, seq_length = x.size()
        # Calculate number of patches
        num_patches = seq_length // self.patch_size
        # Reshape and permute to (batch_size, num_patches, n_channels, patch_size)
        x = x[:, :, :num_patches * self.patch_size]
        x = x.view(batch_size, channels, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_patches, n_channels, patch_size)
        # Flatten patches
        x = x.reshape(batch_size, num_patches, -1)  # (batch_size, num_patches, n_channels * patch_size)
        # Project patches to embeddings
        x = self.proj(x)  # (batch_size, num_patches, embed_dim)
        return x


class EEGTransformer(nn.Module):
    def __init__(self, num_channels, seq_length, patch_size, embed_dim, num_heads, num_layers, num_classes,
                 dropout=0.1):
        super(EEGTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(num_channels, patch_size, embed_dim)
        num_patches = seq_length // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, n_channels, seq_length)
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        batch_size, num_patches, _ = x.size()
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        # Add positional embeddings
        x = x + self.pos_embed[:, :num_patches + 1, :]
        x = self.pos_drop(x)
        # Transformer expects input of shape (sequence_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (num_patches + 1, batch_size, embed_dim)
        x = self.transformer_encoder(x)  # (num_patches + 1, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, num_patches + 1, embed_dim)
        x = self.norm(x)
        # Classification using the CLS token representation
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)
        logits = self.fc(cls_output)  # (batch_size, num_classes)
        return logits
