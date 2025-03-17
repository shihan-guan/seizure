import torch
import torch.nn as nn


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

    def get_patch_dim(self):
        return self.num_channels * self.patch_size  # Dimension of a patch before projection


class EEGTransformer(nn.Module):
    def __init__(self, num_channels, seq_length, patch_size, embed_dim, num_heads, num_layers, num_classes,
                 dropout=0.1):
        super(EEGTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(num_channels, patch_size, embed_dim)
        self.patch_size = patch_size
        num_patches = seq_length // patch_size
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder for masked reconstruction
        patch_dim = self.patch_embed.get_patch_dim()
        self.decoder = nn.Linear(embed_dim, patch_dim)

    def forward(self, x, task='classification', mask=None):
        # x shape: (batch_size, n_channels, seq_length)
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        batch_size, num_patches, _ = x.size()

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :num_patches + 1, :]
        x = self.pos_drop(x)

        if task == 'pretext' and mask is not None:
            # Apply mask to embeddings (e.g., set masked positions to zero)
            # mask shape: (batch_size, num_patches)
            # Extend mask to include CLS token
            mask_with_cls = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device), mask], dim=1)
            x[mask_with_cls] = 0  # Alternatively, you can use a learnable mask embedding

        # Transformer expects input of shape (sequence_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (num_patches + 1, batch_size, embed_dim)
        x = self.transformer_encoder(x)  # (num_patches + 1, batch_size, embed_dim)
        x = x.permute(1, 0, 2)  # (batch_size, num_patches + 1, embed_dim)
        x = self.norm(x)

        if task == 'classification':
            # Classification using the CLS token representation
            cls_output = x[:, 0, :]  # (batch_size, embed_dim)
            out = self.fc(cls_output)
        elif task == 'pretext':
            # Use the outputs at the masked positions to predict the masked patches
            # x[:, 1:, :] shape: (batch_size, num_patches, embed_dim)
            masked_embeddings = x[:, 1:, :][mask]  # (total_masked_patches, embed_dim)
            # Decode to reconstruct the masked patches
            reconstructed_patches = self.decoder(masked_embeddings)  # (total_masked_patches, patch_dim)
            out = reconstructed_patches
        else:
            raise ValueError("Task must be 'classification' or 'pretext'")

        return out
