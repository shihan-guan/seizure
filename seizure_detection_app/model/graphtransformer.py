import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




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



class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.adj = adj  # Normalized adjacency matrix

    def forward(self, x):
        # x shape: (batch_size, n_channels, seq_length)
        adj = self.adj.to(x.device)  # adj shape: (n_channels, n_channels)
        adj = adj.unsqueeze(0)  # adj shape: (1, n_channels, n_channels)
        x = torch.matmul(adj, x)  # x shape: (batch_size, n_channels, seq_length)
        x = x.permute(0, 2, 1)  # x shape: (batch_size, seq_length, n_channels)
        x = self.fc(x)  # Linear layer over 'n_channels' dimension
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # x shape: (batch_size, out_channels, seq_length)
        return x




class EEGTransformer(nn.Module):
    def __init__(self, num_channels, seq_length, patch_size, embed_dim, num_heads, num_layers, num_classes,
                 adj_matrix, dropout=0.1):
        super(EEGTransformer, self).__init__()
        self.gcn = GCNLayer(in_channels=num_channels, out_channels=num_channels, adj=adj_matrix)
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
        # x shape: (batch_size, num_channels, seq_length)
        x = self.gcn(x)  # Apply GCN over channels
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        batch_size, num_patches, _ = x.size()

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embeddings
        x = x + self.pos_embed[:, :num_patches + 1, :]
        x = self.pos_drop(x)

        # Transformer Encoder
        x = x.permute(1, 0, 2)  # (sequence_length, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, embed_dim)
        x = self.norm(x)

        # Classification Head
        cls_output = x[:, 0, :]
        logits = self.fc(cls_output)
        return logits

